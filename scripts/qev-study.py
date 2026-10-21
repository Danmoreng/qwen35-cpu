#!/usr/bin/env python3
"""Prepare and score pinned Jev Skills data using resident CPU option-logit tools."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


adapter = module('qev_adapter', Path(__file__).with_name('qev-cpu.py'))


def prepare(args):
    import pyarrow.parquet as pq
    args.out.mkdir(parents=True, exist_ok=False)
    tok = adapter.TokenizerAdapter(args.model_dir)
    cfg = json.loads((args.model_dir / 'qev_config.json').read_text())
    letters = adapter.label_table(tok)[1]
    if letters != cfg['letter_token_ids']:
        raise ValueError('Option token mismatch')
    (args.out / 'letters.txt').write_text(','.join(map(str, letters)))
    items = []
    for path in sorted((args.upstream / 'data').glob('*.parquet')):
        for item in pq.read_table(path).to_pylist():
            item['questions'] = json.loads(item['questions'])
            item['targets'] = json.loads(item['targets'])
            items.append(item)
    if len(items) != len({x['id'] for x in items}):
        raise ValueError('Duplicate dataset IDs')
    rows, jobs, excluded = [], [], []
    token_dir = args.out / 'tokens'
    token_dir.mkdir()
    for index, item in enumerate(items):
        for key, question in item['questions'].items():
            ids, rendered = adapter.encode(tok, item['state'], question)
            if len(ids) >= args.max_context:
                excluded.append(dict(item=item['id'], question=key, tokens=len(ids)))
                continue
            ident = f'row-{len(rows):06d}'
            path = token_dir / (ident + '.txt')
            path.write_text(','.join(map(str, ids)))
            row = dict(id=ident, item=index, question=key, tokens=len(ids), rendered=rendered)
            rows.append(row)
            jobs.append(f'{ident}\t{path.resolve()}\n')
    (args.out / 'items.json').write_text(json.dumps(items, ensure_ascii=False))
    (args.out / 'rows.json').write_text(json.dumps(rows, ensure_ascii=False))
    (args.out / 'jobs.tsv').write_text(''.join(jobs))
    manifest = dict(encoding='per_question', items=len(items), rows=len(rows), excluded=excluded,
                    tokens=sum(r['tokens'] for r in rows), max_tokens=max(r['tokens'] for r in rows),
                    tokenizer_sha256=adapter.sha(args.model_dir / 'tokenizer.json'),
                    temperatures=cfg['temperatures'],
                    inputs={str(p): adapter.sha(p) for p in sorted((args.upstream / 'data').glob('*.parquet'))})
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


def load_logits(path):
    values = {}
    with path.open() as stream:
        for line in stream:
            fields = line.rstrip('\n').split('\t')
            if fields[0] in values:
                raise ValueError('Duplicate output ID')
            xs = [float(x) for x in fields[1:]]
            if len(xs) != 255 or not all(math.isfinite(x) for x in xs):
                raise ValueError('Invalid option-logit output')
            values[fields[0]] = xs
    return values


def evaluate(args):
    prepared = args.prepared
    items = json.loads((prepared / 'items.json').read_text())
    rows = json.loads((prepared / 'rows.json').read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    logits = load_logits(args.logits)
    reference = load_logits(args.reference) if args.reference else None
    expected = {r['id'] for r in rows}
    if not expected.issubset(logits) or (not args.allow_superset and set(logits) != expected):
        raise ValueError('Incomplete outputs; refusing to score a partial run')
    if reference is not None and not expected.issubset(reference):
        raise ValueError('Incomplete reference outputs')
    answers = [{} for _ in items]
    comparisons = []
    for r in rows:
        names = r['rendered']['names']
        temp = manifest['temperatures'][r['rendered']['type']]
        p = adapter.distribution(logits[r['id']][:len(names)], temp)
        if r['rendered']['type'] == 'noul':
            answer = {'noul': p[1]}
        else:
            answer = {'probabilities': {str(k): v for k, v in zip(names, p, strict=True)}}
        answers[r['item']][r['question']] = answer
        if reference:
            q = adapter.distribution(reference[r['id']][:len(names)], temp)
            a, b = max(range(len(p)), key=p.__getitem__), max(range(len(q)), key=q.__getitem__)
            ordered = sorted(q, reverse=True)
            comparisons.append(dict(row=r['id'], item=items[r['item']]['id'], skill=items[r['item']]['skill'],
                                    kl=sum(x * math.log(x / y) for x, y in zip(q, p, strict=True)),
                                    tv=sum(abs(x-y) for x,y in zip(q,p,strict=True))/2,
                                    max_delta=max(abs(x-y) for x,y in zip(q,p,strict=True)),
                                    flip=a != b, teacher_margin=ordered[0]-ordered[1]))
    scorer = module('jev_scorer', args.upstream / 'code/score.py')
    result = scorer.score(items, answers)
    result.update(encoding='per_question', temperatures=manifest['temperatures'], rows=len(rows),
                  items=len(items), excluded=manifest['excluded'], logits_sha256=adapter.sha(args.logits))
    if reference:
        result['to_bf16'] = dict(mean_kl=statistics.mean(r['kl'] for r in comparisons),
                                 mean_tv=statistics.mean(r['tv'] for r in comparisons),
                                 max_option_delta=max(r['max_delta'] for r in comparisons),
                                 argmax_flips=sum(r['flip'] for r in comparisons), rows=len(rows),
                                 by_skill={skill:dict(mean_kl=statistics.mean(r['kl'] for r in comparisons if r['skill']==skill),
                                                     flips=sum(r['flip'] for r in comparisons if r['skill']==skill))
                                           for skill in sorted({r['skill'] for r in comparisons})})
        args.out.with_suffix('.comparisons.json').write_text(json.dumps(comparisons))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('pooled_all','pooled_by_category','to_bf16','rows','items')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-context', type=int, default=32768)
    p = sub.add_parser('score')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--logits', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--allow-superset', action='store_true', help='Score a complete panel from a larger completed output')
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
