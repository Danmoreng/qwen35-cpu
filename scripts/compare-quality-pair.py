#!/usr/bin/env python3
"""Compare two native quantizations on matched teacher/token evaluations."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def compare(baseline, candidate):
    roots = [baseline, candidate]
    manifests = [json.loads((root / 'commands.json').read_text()) for root in roots]
    for field in ('teacher_sha256', 'llama_sha256', 'threads', 'max_context'):
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f'Mismatched evaluation setting: {field}')
    windows = [{str(w['window']): w for w in m['windows']} for m in manifests]
    if windows[0].keys() != windows[1].keys():
        raise ValueError('Mismatched windows')
    documents = {}
    domains = {}
    for key, window in windows[0].items():
        for field in ('prompt_sha256', 'targets_sha256', 'scored_tokens', 'scoring_mask'):
            if window[field] != windows[1][key][field]:
                raise ValueError(f'Mismatched window {key}: {field}')
        rows = []
        for root in roots:
            with (root / key / 'BF16-vs-native.csv').open() as stream:
                rows.append(list(csv.DictReader(stream)))
        if any(len(r) != window['scored_tokens'] for r in rows):
            raise ValueError('Incomplete evaluation')
        doc = documents.setdefault(window.get('article_sha256', key), np.zeros(5))
        domain = domains.setdefault(window.get('domain', 'english-prose-regression'), np.zeros(5))
        for left, right in zip(*rows):
            for field in ('position', 'target_token'):
                if left[field] != right[field]:
                    raise ValueError('Mismatched token alignment')
            values = np.array([1, float(left['target_nll_candidate']),
                               float(right['target_nll_candidate']),
                               float(left['kld_teacher_candidate']),
                               float(right['kld_teacher_candidate'])])
            if not np.all(np.isfinite(values)):
                raise ValueError('Nonfinite metric')
            doc += values
            domain += values

    def metrics(total):
        n, left_nll, right_nll, left_kl, right_kl = total
        return dict(positions=int(n), baseline_ppl=math.exp(left_nll / n),
                    candidate_ppl=math.exp(right_nll / n),
                    ppl_change_percent=100 * math.expm1((right_nll - left_nll) / n),
                    baseline_kl=left_kl / n, candidate_kl=right_kl / n,
                    kl_change_percent=100 * (right_kl / left_kl - 1) if left_kl else None)

    sums = np.array(list(documents.values()))
    rng = np.random.default_rng(1234)
    samples = sums[rng.integers(len(sums), size=(2000, len(sums)))].sum(axis=1)
    ppl_changes = 100 * np.expm1((samples[:, 2] - samples[:, 1]) / samples[:, 0])
    kl_deltas = (samples[:, 4] - samples[:, 3]) / samples[:, 0]
    return dict(baseline=str(baseline), candidate=str(candidate),
                overall=metrics(sums.sum(axis=0)),
                domains={k: metrics(v) for k, v in domains.items()},
                paired_document_bootstrap=dict(documents=len(sums), seed=1234,
                    repetitions=2000,
                    ppl_change_percent_95=np.percentile(ppl_changes, [2.5, 97.5]).tolist(),
                    kl_delta_95=np.percentile(kl_deltas, [2.5, 97.5]).tolist()),
                interpretation='Negative changes favor candidate; quantization quality, not implementation equivalence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.baseline, args.candidate)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['overall'], indent=2))
