#!/usr/bin/env python3
"""Summarize the completed fixed qev study, preserving development/final separation."""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import struct
import sys
import numpy as np
from threadpoolctl import threadpool_limits


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


study = module('qev_study', Path(__file__).with_name('qev-study.py'))


def group_id(item):
    return re.sub(r'-(complete|missing)$', '', item['id'].replace('game_q-', 'game-').replace('game_s-', 'game-'))


def paired_intervals(prepared, baseline, candidate):
    items = json.loads((prepared/'items.json').read_text())
    rows = json.loads((prepared/'rows.json').read_text())
    temps = json.loads((prepared/'manifest.json').read_text())['temperatures']
    a, b = study.load_logits(baseline), study.load_logits(candidate)
    skills = sorted({x['skill'] for x in items if x['skill'] != 'rules/missing_facts'})
    groups = sorted({group_id(x) for x in items})
    gi, si = {g:i for i,g in enumerate(groups)}, {s:i for i,s in enumerate(skills)}
    counts = np.zeros((len(groups),len(skills)))
    hit_delta = np.zeros_like(counts)
    brier_delta = np.zeros_like(counts)
    for row in rows:
        item = items[row['item']]
        target = item['targets'].get(row['question'])
        if item['skill'] not in si or not target:
            continue
        names = row['rendered']['names']
        names = [('true' if x else 'false') if isinstance(x,bool) else str(x) for x in names]
        spec = item['questions'][row['question']]
        if spec['type'] == 'score':
            levels = [str(level) for level in spec['criteria']]
            names = [str(levels.index(name)) if name in levels else name for name in names]
        temp = temps[row['rendered']['type']]
        pa = study.adapter.distribution(a[row['id']][:len(names)],temp)
        pb = study.adapter.distribution(b[row['id']][:len(names)],temp)
        gold = max(target,key=target.get)
        def hit(p):
            top = [n for n,x in zip(names,p) if x >= max(p)-1e-9]
            return (1/len(top)) if gold in top else 0
        def brier(p):
            return sum((x-target.get(n,0))**2 for n,x in zip(names,p))
        g,s = gi[group_id(item)],si[item['skill']]
        counts[g,s] += 1
        hit_delta[g,s] += hit(pb)-hit(pa)
        brier_delta[g,s] += brier(pb)-brier(pa)
    rng = np.random.default_rng(20261021)
    accuracy, briers = [], []
    with threadpool_limits(limits=2):
        for _ in range(10):
            weights = rng.multinomial(len(groups), np.full(len(groups),1/len(groups)),size=100)
            den = weights@counts
            accuracy.extend(np.mean((weights@hit_delta)/den,axis=1).tolist())
            briers.extend(np.mean((weights@brier_delta)/den,axis=1).tolist())
    return dict(method='paired cluster bootstrap, grouped variants/requests; macro mean across scored skills',
                resamples=1000,seed=20261021,
                raw_accuracy_delta_percentage_points=float(np.mean(hit_delta.sum(axis=0)/counts.sum(axis=0))*100),
                accuracy_delta_percentage_points_ci95=[float(x*100) for x in np.quantile(accuracy,[.025,.975])],
                macro_brier_delta_ci95=[float(x) for x in np.quantile(briers,[.025,.975])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    root,out = args.root,args.out
    out.mkdir(parents=True,exist_ok=True)
    selection = json.loads((root/'frozen-final-selection.json').read_text())
    def get(name):
        return json.loads((root/name).read_text())
    reports = dict(development={},final={})
    for panel in reports:
        prepared = root/'panels'/panel
        for tag,path in [('bf16',root/'current-bf16-gpu.tsv'),('q8',root/'current-q8-gpu.tsv')]:
            score_path = root/f'{tag}-{panel}-score.json'
            cmd=[sys.executable,'scripts/qev-study.py','score','--prepared',str(prepared),'--upstream',str(root/'upstream'),
                 '--logits',str(path),'--allow-superset','--out',str(score_path)]
            if tag!='bf16':cmd+=['--reference',str(root/'current-bf16-gpu.tsv')]
            import subprocess
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            reports[panel][tag]=get(score_path.name)
        reports[panel]['uncalibrated']=get(f'uncalibrated-{panel}-score.json')
    candidates = get('development-comparison.json')
    for candidate in candidates:
        name = candidate['name']
        reports['development'][name]=get(name+'-development-score.json')
    for kind in ('pure', 'mixed'):
        reports['final'][selection[kind]['name']]=get(f'selected-{kind}-final-score.json')
    sizes={'bf16':1505783040,'q8':800881920,'uncalibrated':424934656,
           **{candidate['name']:tensor_bytes(Path(candidate['weights'])) for candidate in candidates}}
    file_sizes={'bf16':1516746912,'q8':811845792,'uncalibrated':424964864,
                **{candidate['name']:Path(candidate['weights']).stat().st_size for candidate in candidates}}
    records=[]
    for panel,variants in reports.items():
        for name,x in variants.items():
            metrics=[v for s,v in x['skills'].items() if s!='rules/missing_facts']
            fidelity=x.get('to_bf16',{})
            records.append(dict(panel=panel,checkpoint=name,backend='llama.cpp CUDA' if name in ('bf16','q8') else 'native CPU',
                tensor_bytes=sizes[name],file_bytes=file_sizes[name],items=x['items'],questions=x['rows'],macro_accuracy=x['pooled_all'],
                macro_brier=statistics.mean(v['brier'] for v in metrics),
                mean_option_kl=fidelity.get('mean_kl',0),mean_option_tv=fidelity.get('mean_tv',0),
                argmax_flips=fidelity.get('argmax_flips',0),max_option_delta=fidelity.get('max_option_delta',0)))
    records.sort(key=lambda r:(r['panel'],-r['tensor_bytes'],r['mean_option_kl']))
    with (out/'quality.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    a=reports['final']['uncalibrated']
    outcomes = {}
    for kind in ('pure', 'mixed'):
        selected = selection[kind]
        b = reports['final'][selected['name']]
        candidate_logits = root/f'selected-{kind}-final.tsv'
        outcomes[kind] = dict(selected=selected,
            accuracy_gain_percentage_points=100*(b['pooled_all']-a['pooled_all']),
            mean_kl_reduction_percent=100*(1-b['to_bf16']['mean_kl']/a['to_bf16']['mean_kl']),
            argmax_flips_before=a['to_bf16']['argmax_flips'],argmax_flips_after=b['to_bf16']['argmax_flips'],
            vs_uncalibrated=paired_intervals(root/'panels/final',root/'uncalibrated-final.tsv',candidate_logits),
            vs_bf16=paired_intervals(root/'panels/final',root/'current-bf16-gpu.tsv',candidate_logits),
            vs_q8=paired_intervals(root/'panels/final',root/'current-q8-gpu.tsv',candidate_logits))
    summary=dict(selection=selection,development_comparison=candidates,final_outcomes=outcomes,
        panels=get('panels/manifest.json'),study_plan=get('study-plan.json'),
        dataset_revision=get('dataset-info.json')['sha'],cpu_gpu_validation=get('cpu-gpu-validation.json'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'per-skill-final.json').write_text(json.dumps({k:{name:v for name,v in x.items() if name in ('skills','extras','pooled_by_category','to_bf16')} for k,x in reports['final'].items()},indent=2)+'\n')
    (out/'smoke-development.json').write_text(json.dumps(smoke_results(root, candidates),indent=2)+'\n')
    (out/'probability-drift-detail.json').write_text(json.dumps(probability_drift(root),indent=2)+'\n')
    print(json.dumps(outcomes,indent=2))


def tensor_bytes(path):
    """Read payload sizes without loading weights or relying on quantization labels."""
    with path.open('rb') as f:
        if f.read(8) != b'Q35H128\0' or struct.unpack('<4I',f.read(16)) != (1,0x01020304,96,64):
            raise ValueError('Unsupported artifact header')
        count,directory=struct.unpack('<2Q',f.read(16))
        f.seek(directory)
        total=0
        for _ in range(count):
            length,rank,_,_,_,_,_,_,_,size,_=struct.unpack('<6I5Q',f.read(64))
            f.seek(length+rank*8,1)
            total+=size
        return total


def probability_drift(root):
    """Measure option-probability tails on the same frozen final questions."""
    prepared = root/'panels/final'
    rows = json.loads((prepared/'rows.json').read_text())
    temps = json.loads((prepared/'manifest.json').read_text())['temperatures']
    reference = study.load_logits(root/'current-bf16-gpu.tsv')
    files = [('q8', 'current-q8-gpu.tsv'), ('uncalibrated', 'uncalibrated-all.tsv'),
             ('pure', 'selected-pure-final.tsv'), ('mixed', 'selected-mixed-final.tsv')]
    results = {}
    for name, filename in files:
        values = study.load_logits(root/filename)
        deltas, tvs, flips = [], [], 0
        for row in rows:
            count = len(row['rendered']['names'])
            temp = temps[row['rendered']['type']]
            a = np.asarray(study.adapter.distribution(reference[row['id']][:count],temp))
            b = np.asarray(study.adapter.distribution(values[row['id']][:count],temp))
            differences = np.abs(a-b)*100
            deltas.append(float(differences.max()))
            tvs.append(float(differences.sum()/2))
            flips += int(a.argmax() != b.argmax())
        results[name] = dict(rows=len(rows),mean_tv_percentage_points=float(np.mean(tvs)),
            mean_max_option_delta=float(np.mean(deltas)),
            median_max_option_delta=float(np.median(deltas)),
            p90_max_option_delta=float(np.quantile(deltas,.9)),
            p95_max_option_delta=float(np.quantile(deltas,.95)),
            p99_max_option_delta=float(np.quantile(deltas,.99)),max_option_delta=max(deltas),
            rows_over_10pp=sum(x>10 for x in deltas),rows_over_20pp=sum(x>20 for x in deltas),
            argmax_flips=flips)
    return dict(panel='final',reference='BF16, llama.cpp CUDA',
                metric='Maximum absolute option-probability difference per question; percentage points',
                variants=results)


def smoke_results(root, candidates):
    """Keep the inspected smoke cases separate from independent final metrics."""
    questions=[]
    for line in Path('configs/qev-smoke.jsonl').read_text().splitlines():
        if not line.strip():
            continue
        request=json.loads(line)
        for key,spec in request['questions'].items():
            questions.append((request['id'],key,study.adapter.render_question(spec),request['expected'][key]))
    files=[('bf16',root/'current-smoke-bf16.tsv'),('q8',root/'current-smoke-q8.tsv'),('uncalibrated',root/'smoke-native.tsv')]
    files += [(x['name'],root/f"smoke-{x['name']}.tsv") for x in candidates]
    result={}
    for name,path in files:
        values=study.load_logits(path)
        if len(values)!=len(questions):
            raise ValueError('Incomplete smoke output: '+str(path))
        matches,failures,snake=0,[],None
        for i,(ident,key,rendered,want) in enumerate(questions):
            logits=values[f'smoke-{i:03d}'][:len(rendered['names'])]
            probs=study.adapter.distribution(logits,dict(choice=1.7,noul=.9,score=1.25)[rendered['type']])
            got=rendered['names'][max(range(len(probs)),key=probs.__getitem__)]
            matches+=got==want
            if got!=want:
                failures.append(dict(request=ident,question=key,expected=want,actual=got))
            if ident=='snake':
                snake=dict(zip(rendered['names'],probs))
        result[name]=dict(questions=len(questions),argmax_matches=matches,failures=failures,snake=snake)
    return result


if __name__=='__main__':
    main()
