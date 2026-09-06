#!/usr/bin/env python3
"""Compare a Q8 experiment against the current calibrated Q4 standard."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np
from evaluation_common import sha, source_identity

p=argparse.ArgumentParser()
p.add_argument('--label',required=True,choices=['q8-gates','q8-head','q8-gates-head'])
args=p.parse_args()
label=args.label
root=Path('benchmarks')
out=Path('docs/results/q8-experiments-2026-09-06')/label
out.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
meta=read(root/f'{label}-speed/metadata.json')
assert meta['runs']==3 and meta['warmup_runs']==1 and meta['affinity']==21845
performance=[]
for i,case in enumerate(meta['matrix']):
    name,b,t,prompt,n=case['name'].rsplit('-',4)
    b,t,prompt,n=[int(x[1:]) for x in (b,t,prompt,n)]
    profiles=[read(root/f'{label}-speed/case-{i}-run-{r}.json') for r in (1,2,3)]
    assert all(x['cpu_batch']==b and x['prompt_tokens']==b*prompt and x['generated_tokens']==b*n
        and x['decode_forwards']==b*(n-1) and x['cpu_kv_cache']=='fp16'
        and not x['cached_prefix_tokens'] and not x['greedy_lm_head_batches']
        and not x.get('operation_instrumentation',False) for x in profiles)
    row=dict(candidate=name,batch=b,threads=t,prompt=prompt,output=n)
    for field,key in [('prefill','prefill_forward_tokens_per_second'),('decode','tokens_per_second'),('peak_rss_bytes','peak_rss_bytes')]:
        values=[x[key] for x in profiles]
        row.update({field+'_median':statistics.median(values),field+'_min':min(values),field+'_max':max(values)})
    performance.append(row)
for row in performance:
    baseline=next(r for r in performance if r['candidate']=='calibrated' and all(r[k]==row[k] for k in ('batch','threads','prompt','output')))
    row['prefill_ratio']=row['prefill_median']/baseline['prefill_median']
    row['decode_ratio']=row['decode_median']/baseline['decode_median']
    row['speed_pass']=row['prefill_ratio']>=.97 and (row['output']==2 or row['decode_ratio']>=.97)

quality={}
for suite,baseline_root,candidate_root in [('regression',root/'plan-calibrated-full',root/f'{label}-full'),
                                          ('heldout',root/'plan-heldout-calibrated',root/f'{label}-heldout')]:
    baseline,candidate=read(baseline_root/'summary.json'),read(candidate_root/'summary.json')
    bm,cm=read(baseline_root/'commands.json'),read(candidate_root/'commands.json')
    assert bm['teacher_sha256']==cm['teacher_sha256'] and len(bm['windows'])==len(cm['windows'])
    documents=[]
    for bw,cw in zip(bm['windows'],cm['windows']):
        assert all(bw[k]==cw[k] for k in ('window','prompt_sha256','targets_sha256','scoring_mask'))
        def positions(folder):
            with (folder/str(cw['window'])/'BF16-vs-native.csv').open() as f:return list(csv.DictReader(f))
        before,after=positions(baseline_root),positions(candidate_root)
        assert len(before)==len(after)==cw['scored_tokens']
        assert all(a['target_token']==b['target_token'] for a,b in zip(before,after))
        documents.append([len(after),sum(float(b['target_nll_candidate'])-float(a['target_nll_candidate']) for a,b in zip(before,after)),
            sum(float(b['kld_teacher_candidate'])-float(a['kld_teacher_candidate']) for a,b in zip(before,after))])
    sums=np.array(documents)
    rng=np.random.default_rng(1234)
    samples=sums[rng.integers(len(sums),size=(2000,len(sums)))].sum(axis=1)
    quality[suite]=dict(baseline_ppl=baseline['candidate_ppl'],candidate_ppl=candidate['candidate_ppl'],
        baseline_kl=baseline['candidate_kl'],candidate_kl=candidate['candidate_kl'],
        ppl_ratio=candidate['candidate_ppl']/baseline['candidate_ppl'],
        kl_delta=candidate['candidate_kl']-baseline['candidate_kl'],
        paired_document_ppl_ratio_95=np.exp(np.percentile(samples[:,1]/samples[:,0],[2.5,97.5])).tolist(),
        paired_document_kl_delta_95=np.percentile(samples[:,2]/samples[:,0],[2.5,97.5]).tolist(),
        quality_pass=candidate['candidate_ppl']<baseline['candidate_ppl'] and candidate['candidate_kl']<baseline['candidate_kl'],
        positions=candidate['positions'],candidate_details=candidate)
arithmetic=read(root/f'{label}-arithmetic/summary.json')
arithmetic_pass=all(r['free_pass'] and r['prefix_and_answer_pass'] for r in arithmetic['results'])
speed_pass=all(r['speed_pass'] for r in performance if r['candidate']==label)
quality_pass=all(r['quality_pass'] for r in quality.values())
audit=read(root/f'{label}-audit.json')
result=dict(candidate=label,quality=quality,performance=performance,audit=audit,arithmetic=arithmetic,
    gates=dict(speed_tolerance=.03,quality_pass=quality_pass,speed_pass=speed_pass,arithmetic_pass=arithmetic_pass,
               promotion_screen_pass=quality_pass and speed_pass and arithmetic_pass))
with (out/'performance.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(performance[0]),lineterminator='\n');writer.writeheader();writer.writerows(performance)
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
files=[]
for folder in [root/f'{label}-full',root/f'{label}-heldout']:
    files += [f for f in folder.rglob('*') if f.is_file() and f.suffix in ('.json','.csv','.py')]
for folder in [root/f'{label}-speed',root/f'{label}-speed-inputs',root/f'{label}-arithmetic',root/'q8-baseline-invariance']:
    files += [f for f in folder.glob('*') if f.is_file() and f.suffix!='.logits']
files += [root/f'{label}-audit.json',root/f'{label}-pack.log',root/f'{label}-scheduler.log',root/f'{label}-prefix.log',root/'q8-ctest.log']
checkpoint=Path('models/qwen3.5-0.8b')/f'model-calibrated-{label}.q35h'
files += [Path(str(checkpoint)+suffix) for suffix in ('.precision.json','.quantization.json','.calibration.json')]
if label=='q8-head':
    files.append(root/'q8-head-q4km-inventory.json')
with zipfile.ZipFile(out/'raw-results.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for f in sorted(set(files)):
        assert f.is_file() and f.suffix not in ('.exe','.dll','.q35h','.gguf','.logits')
        archive.write(f,f.as_posix())
manifest=dict(source=source_identity(Path('.')),binary_sha256={f.name:sha(f) for f in (root/'q8-tools').glob('*.exe')},
    files={f.name:sha(f) for f in out.iterdir() if f.is_file() and f.name not in ('manifest.json','SHA256SUMS')})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
(out/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(out.iterdir()) if f.is_file() and f.name!='SHA256SUMS'),encoding='utf-8',newline='\n')
print(json.dumps(dict(quality={s:{k:v for k,v in q.items() if k!='candidate_details'} for s,q in quality.items()},gates=result['gates']),indent=2))
