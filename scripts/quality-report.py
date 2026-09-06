#!/usr/bin/env python3
"""Aggregate token metrics and paired document-bootstrap quality gates."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np


def report(root, ppl_tolerance=0.01, kl_tolerance=0.002, seed=1234):
    manifest = json.loads((root/'commands.json').read_text())
    records = []
    for window in manifest['windows']:
        folder = root/str(window['window'])
        with (folder/'BF16-vs-native.csv').open() as f:
            candidate = list(csv.DictReader(f))
        with (folder/'BF16-vs-llama.csv').open() as f:
            reference = list(csv.DictReader(f))
        compatibility = list(folder.glob('llama-*-vs-native.csv'))
        if len(compatibility) != 1:
            raise ValueError('Missing/ambiguous same-checkpoint reference comparison')
        with compatibility[0].open() as f:
            compat = list(csv.DictReader(f))
        if not (len(candidate) == len(reference) == len(compat) == window['scored_tokens']):
            raise ValueError('Incomplete window')
        for c, r, k in zip(candidate, reference, compat):
            if c['position'] != r['position'] or c['target_token'] != r['target_token']:
                raise ValueError('Paired token mismatch')
            records.append(dict(document=window.get('article_sha256', str(window['window'])),
                domain=window.get('domain', 'english-prose-regression'),
                nll=float(c['target_nll_candidate']), reference_nll=float(r['target_nll_candidate']),
                kl=float(c['kld_teacher_candidate']), reference_kl=float(r['kld_teacher_candidate']),
                compatibility_kl=float(k['kld_teacher_candidate']), top1=c['top1_equal'] == 'True'))
    if not records:
        raise ValueError('Empty evaluation')

    def aggregate(rows):
        mean = lambda name: float(np.mean([r[name] for r in rows]))
        delta = mean('nll') - mean('reference_nll')
        kl_delta = mean('kl') - mean('reference_kl')
        return dict(positions=len(rows), candidate_ppl=math.exp(mean('nll')),
            reference_ppl=math.exp(mean('reference_nll')), ppl_ratio=math.exp(delta),
            candidate_kl=mean('kl'), reference_kl=mean('reference_kl'),
            compatibility_kl=mean('compatibility_kl'), top1=mean('top1'),
            kl_p95=float(np.percentile([r['kl'] for r in rows], 95)),
            kl_p99=float(np.percentile([r['kl'] for r in rows], 99)),
            kl_max=max(r['kl'] for r in rows),
            strict_pass=delta <= 0 and kl_delta <= 0,
            engineering_pass=delta <= math.log1p(ppl_tolerance) and kl_delta <= kl_tolerance)

    result = aggregate(records)
    result['gates'] = dict(ppl_relative_tolerance=ppl_tolerance, kl_absolute_tolerance=kl_tolerance,
                           strict_ppl_relative_tolerance=0, strict_kl_absolute_tolerance=0)
    result['domains'] = {d: aggregate([r for r in records if r['domain'] == d])
                         for d in sorted({r['domain'] for r in records})}
    # Resample complete documents, retaining paired candidate/reference tokens.
    documents = sorted({r['document'] for r in records})
    sums = np.array([[len(rows), sum(r['nll']-r['reference_nll'] for r in rows),
                     sum(r['kl']-r['reference_kl'] for r in rows)]
                    for d in documents for rows in [[r for r in records if r['document'] == d]]])
    rng = np.random.default_rng(seed)
    samples = sums[rng.integers(len(documents), size=(2000, len(documents)))].sum(axis=1)
    result['paired_document_bootstrap'] = dict(seed=seed, repetitions=2000, documents=len(documents),
        nll_delta_95=np.percentile(samples[:, 1]/samples[:, 0], [2.5, 97.5]).tolist(),
        kl_delta_95=np.percentile(samples[:, 2]/samples[:, 0], [2.5, 97.5]).tolist())
    result['scope'] = 'Declared windows only; regression prose does not establish general equivalence'
    result['compatibility_same_checkpoint'] = manifest['checkpoint_sha256'] == manifest['reference_checkpoint_sha256']
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--ppl-tolerance', type=float, default=.01)
    parser.add_argument('--kl-tolerance', type=float, default=.002)
    args = parser.parse_args()
    if args.ppl_tolerance < 0 or args.kl_tolerance < 0:
        parser.error('Tolerances must be nonnegative')
    result = report(args.root, args.ppl_tolerance, args.kl_tolerance)
    (args.root/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))
