#!/usr/bin/env python3
"""Collect measured head evidence without treating weight SSE as model quality."""
import csv
import json
from pathlib import Path
import statistics
import math
import numpy as np

out = Path('docs/results/head-experiments-2026-09-06')
out.mkdir(parents=True, exist_ok=True)
quality = {}
paths = {
    'unweighted-development': 'plan-heldout-mse16',
    'unweighted-regression': 'plan-mse16-full',
    'calibrated-development': 'plan-heldout-calibrated',
    'calibrated-regression': 'plan-calibrated-full',
    'h128-development': 'head-h128-heldout',
    'g16-development': 'head-g16-heldout',
    'g16-regression': 'head-g16-full',
    'h128-g16-development': 'head-h128-g16-heldout',
    'unweighted-audit': 'head-audit-mse16',
    'g16-audit': 'head-audit-g16-head',
    'h128-g16-audit': 'head-audit-h128-g16-head',
}
for name, folder in paths.items():
    source = Path('benchmarks')/folder/'summary.json'
    if source.exists():
        quality[name] = json.loads(source.read_text())
        (out/(name+'.json')).write_bytes(source.read_bytes())

timing = {}
for folder in ['head-speed-final', 'head-speed-extended', 'head-speed-attribution']:
    source = Path('benchmarks')/folder/'results.csv'
    if not source.exists():
        continue
    rows = list(csv.DictReader(source.open(encoding='utf-8-sig')))
    for name in dict.fromkeys(r['name'] for r in rows):
        measured = [r for r in rows if r['name'] == name and r['warmup'].lower() == 'false']
        numeric = {}
        for key in measured[0]:
            if 'tokens_per_second' not in key and 'tok_s' not in key and 'ms' not in key and 'bytes' not in key:
                continue
            try:
                values = [float(r[key]) for r in measured]
            except (ValueError, TypeError):
                continue
            numeric[key] = dict(runs=values, median=statistics.median(values),
                                minimum=min(values), maximum=max(values))
        timing[name] = dict(source=str(source), metrics=numeric)
    target = out/folder
    target.mkdir(exist_ok=True)
    for path in source.parent.iterdir():
        if path.is_file():
            (target/path.name).write_bytes(path.read_bytes())

paired = {}
for label, control, candidate in [('regression', 'plan-mse16-full', 'head-g16-full'),
                                  ('development', 'plan-heldout-mse16', 'head-g16-heldout'),
                                  ('audit', 'head-audit-mse16', 'head-audit-g16-head')]:
    roots = [Path('benchmarks')/name for name in (control, candidate)]
    if not all((root/'summary.json').exists() for root in roots):
        continue
    manifests = [json.loads((root/'commands.json').read_text()) for root in roots]
    windows = [{str(w['window']):w for w in m['windows']} for m in manifests]
    if windows[0].keys() != windows[1].keys():
        raise ValueError('Paired quality windows differ')
    documents = {}
    for key, window in windows[0].items():
        for field in ('prompt_sha256', 'targets_sha256', 'scored_tokens'):
            if window[field] != windows[1][key][field]:
                raise ValueError('Paired quality input mismatch')
        rows = [list(csv.DictReader((root/key/'BF16-vs-native.csv').open())) for root in roots]
        if len(rows[0]) != len(rows[1]):
            raise ValueError('Paired quality row counts differ')
        doc = documents.setdefault(window.get('article_sha256', key), [0, 0.0, 0.0])
        for baseline, experiment in zip(*rows):
            if any(baseline[f] != experiment[f] for f in ('position', 'target_token')):
                raise ValueError('Paired quality token mismatch')
            doc[0] += 1
            doc[1] += float(experiment['target_nll_candidate'])-float(baseline['target_nll_candidate'])
            doc[2] += float(experiment['kld_teacher_candidate'])-float(baseline['kld_teacher_candidate'])
    sums = np.asarray(list(documents.values()))
    total = sums.sum(axis=0)
    rng = np.random.default_rng(1234)
    samples = sums[rng.integers(len(sums), size=(2000, len(sums)))].sum(axis=1)
    paired[label] = dict(positions=int(total[0]), documents=len(sums),
        ppl_ratio=math.exp(total[1]/total[0]), teacher_kl_delta=total[2]/total[0],
        nll_delta_95=np.percentile(samples[:,1]/samples[:,0], [2.5,97.5]).tolist(),
        teacher_kl_delta_95=np.percentile(samples[:,2]/samples[:,0], [2.5,97.5]).tolist(),
        interpretation='Paired quantization quality differences, not same-checkpoint implementation error',
        bootstrap_seed=1234, bootstrap_repetitions=2000)

(out/'results.json').write_text(json.dumps(dict(quality=quality, timing=timing, paired_g16_vs_unweighted=paired,
    scope='Measured compact head track. Historical controls retain their original measurement provenance; speed controls are paired in the new series.'), indent=2)+'\n')
print(json.dumps({k:dict(positions=v['positions'], ppl=v['candidate_ppl'], kl=v['candidate_kl'])
                  for k,v in quality.items()}, indent=2))
