"""Compute paired quality metrics from the fixed ticket panel and native readouts."""
import argparse
import gzip
from collections import Counter
import json
import math
from pathlib import Path
import statistics


def summarize(panel, rows):
    labels = list(panel['criteria'])
    cases = {t['id']: t for t in panel['tickets']}
    if len(cases) != len(panel['tickets']) or len({t['text'] for t in cases.values()}) != len(cases):
        raise ValueError('Duplicate ticket ID or text')
    if len(rows) != len(cases) or {r['id'] for r in rows} != set(cases):
        raise ValueError('Missing, duplicate or unexpected predictions')
    sizes = Counter(t['label'] for t in cases.values())
    if set(sizes) != set(labels):
        raise ValueError('Invalid or missing gold labels')

    methods = ('context_first', 'question_first', 'question_first_cached')
    output = {}
    for method in methods:
        confusion = {k: dict.fromkeys(labels, 0) for k in labels}
        by_subset = {}
        gold_probs = []
        for r in rows:
            case, pred = cases[r['id']], r[method]
            probs = pred['probabilities']
            if (set(probs) != set(labels) or pred['choice'] not in labels or
                    any(not 0 <= x <= 1 for x in probs.values()) or abs(sum(probs.values())-1) > .0005):
                raise ValueError(f"Invalid returned distribution for {r['id']}")
            valid_cache = (0 < pred['cached_tokens'] < r['tokens'] if method == 'question_first_cached'
                           else pred['cached_tokens'] == 0)
            if not valid_cache:
                raise ValueError(f"Unexpected cache reuse for {r['id']}")
            confusion[case['label']][pred['choice']] += 1
            by_subset.setdefault(case['subset'], []).append(case['label'] == pred['choice'])
            gold_probs.append(probs[case['label']])
        correct = sum(confusion[k][k] for k in labels)
        f1 = [2*confusion[k][k] / (sizes[k] + sum(row[k] for row in confusion.values())) for k in labels]
        output[method] = dict(correct=correct, accuracy=correct/len(rows), macro_f1=statistics.mean(f1),
                              mean_gold_probability=statistics.mean(gold_probs), confusion=confusion,
                              by_subset={k: dict(correct=sum(v), n=len(v)) for k, v in by_subset.items()})

    changes = []
    gained = lost = both_wrong = both_right = 0
    cache_flip = 0
    max_cache_logit = max_cache_probability = 0
    for r in rows:
        old, new, cached = (r[m] for m in methods)
        gold = cases[r['id']]['label']
        a, b = old['choice'], new['choice']
        gained += a != gold and b == gold
        lost += a == gold and b != gold
        both_wrong += a != gold and b != gold
        both_right += a == gold and b == gold
        if a != b:
            changes.append(dict(id=r['id'], gold=gold, context_first=a, question_first=b))
        cache_flip += cached['choice'] != b
        max_cache_probability = max(max_cache_probability,
            max(abs(new['probabilities'][k]-cached['probabilities'][k]) for k in labels))
        max_cache_logit = max(max_cache_logit,
            max(abs(x-y) for x,y in zip(new['logits'], cached['logits'], strict=True)))
    n = gained + lost
    p = min(1, 2 * sum(math.comb(n, i) for i in range(min(gained, lost)+1)) / 2**n)
    return dict(n=len(rows), methods=output, paired=dict(gained=gained, lost=lost, both_wrong=both_wrong,
                both_right=both_right, changed_choices=changes, mcnemar_exact_two_sided_p=p),
                cache=dict(choice_changes=cache_flip, max_probability_difference=max_cache_probability,
                           max_absolute_logit_difference=max_cache_logit,
                           prefix_lengths=sorted({r['question_first_cached']['cached_tokens'] for r in rows})),
                input_tokens=[min(r['tokens'] for r in rows), max(r['tokens'] for r in rows)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('panel', type=Path)
    parser.add_argument('rows', type=Path)
    args = parser.parse_args()
    panel = json.loads(args.panel.read_text(encoding='utf-8'))
    opener = gzip.open if args.rows.suffix == '.gz' else open
    with opener(args.rows, 'rt', encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream]
    print(json.dumps(summarize(panel, rows), indent=2, ensure_ascii=False, allow_nan=False))
