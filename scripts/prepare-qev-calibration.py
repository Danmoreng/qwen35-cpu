#!/usr/bin/env python3
"""Prepare a fixed, independent decision calibration mix from pinned train sources.

No Jev Skills examples or generators are used to construct calibration requests.
State 16-word shingles are screened against Jev Skills to remove source overlap.
"""
import argparse
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import struct
import pyarrow.parquet as pq

spec = importlib.util.spec_from_file_location('qev_adapter', Path(__file__).with_name('qev-cpu.py'))
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def shingles(state):
    words = re.findall(r'\w+', state.lower())
    return {hashlib.blake2b(' '.join(words[i:i+16]).encode(), digest_size=8).digest()
            for i in range(max(0, len(words)-15))}


def choice(instructions, criteria):
    return dict(type='choice', instructions=instructions, criteria=criteria)


def noul(instructions):
    return dict(type='noul', instructions=instructions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', type=Path, required=True)
    p.add_argument('--eval-items', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261021)
    args = p.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=False)
    eval_items = json.loads(args.eval_items.read_text())
    blocked = set()
    eval_states = {x['state'].strip().lower() for x in eval_items}
    for x in eval_items:
        blocked.update(shingles(x['state']))
    tok = adapter.TokenizerAdapter(args.model_dir)
    groups = collections.defaultdict(list)
    rejected = collections.Counter()
    seen_states = set()

    def add(domain, source_id, state, question, label, license):
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        key = state.strip().lower()
        if key in eval_states or shingles(state) & blocked:
            rejected[domain + '/evaluation_overlap'] += 1
            return
        if key in seen_states:
            rejected[domain + '/duplicate'] += 1
            return
        ids, rendered = adapter.encode(tok, state, question)
        if len(ids) > 2048:
            rejected[domain + '/over_2048_tokens'] += 1
            return
        seen_states.add(key)
        groups[domain].append(dict(source_id=source_id, domain=domain, state=state,
                                   question=question, label=label, license=license,
                                   tokens=ids, rendered=rendered))

    bank = list(enumerate(csv.DictReader((args.sources / 'banking-train.csv').open())))
    categories = json.loads((args.sources / 'banking-categories.json').read_text())
    rng.shuffle(bank)
    for i, row in bank:
        label = row['category']
        option_count = [4, 8, 16, 32, 77][i % 5]
        options = rng.sample([c for c in categories if c != label], option_count-1) + [label]
        rng.shuffle(options)
        add('banking', 'banking/train/' + str(i), {'customer_message': row['text']},
            choice('Which customer support intent best matches this message?',
                   {c: c.replace('_', ' ') for c in options}), label, 'CC-BY-4.0')
        if len(groups['banking']) >= 128:
            break
    for domain, filename, count in [('boolq', 'google--boolq.parquet', 256),
                                     ('paws', 'google-research-datasets--paws.parquet', 128),
                                     ('injection', 'deepset--prompt-injections.parquet', 128)]:
        rows = list(enumerate(pq.read_table(args.sources / filename).to_pylist()))
        rng.shuffle(rows)
        for i, row in rows:
            if domain == 'boolq':
                state = {'passage': row['passage']}
                question = noul('Based on the passage, is the answer to this question yes? ' + row['question'])
                label, license = bool(row['answer']), 'CC-BY-SA-3.0'
            elif domain == 'paws':
                state = {'sentence A': row['sentence1'], 'sentence B': row['sentence2']}
                question = noul('Do sentence A and sentence B express the same meaning?')
                label, license = bool(row['label']), 'Apache-2.0'
            else:
                state = row['text']
                question = noul('Does this text try to override or manipulate an AI assistant\'s instructions?')
                label, license = bool(row['label']), 'Apache-2.0'
            add(domain, domain + '/train/' + str(i), state, question, label, license)
            if len(groups[domain]) >= count:
                break
    preferences = list(enumerate(pq.read_table(args.sources / 'ultrafeedback-train.parquet').to_pylist()))
    rng.shuffle(preferences)
    for i, row in preferences:
        responses = [row['chosen'][-1]['content'], row['rejected'][-1]['content']]
        flip = rng.randrange(2)
        if flip:
            responses.reverse()
        state = {'user': row['prompt'], 'response A': responses[0], 'response B': responses[1]}
        add('preference', 'ultrafeedback/train_prefs/'+row['prompt_id'], state,
            choice('Which response is more helpful, correct, and appropriate for the user request?',
                   {'A':'Response A', 'B':'Response B'}), 'B' if flip else 'A', 'MIT')
        if len(groups['preference']) >= 128:
            break
    for i in range(512):
        if len(groups['policy']) < 128:
            values = [rng.randint(0, 30) for _ in range(4)]
            limits = [rng.randint(5, 25) for _ in range(4)]
            violations = sum(v > lim for v, lim in zip(values, limits))
            german = i % 2 == 0
            state = {'batch': f'CAL-{i:04d}', 'readings': dict(zip('ABCD', values)),
                     'maximum_allowed': dict(zip('ABCD', limits)),
                     'rule': 'Count readings strictly greater than their maximum; equality is allowed.'}
            instructions = ('Wie viele Messwerte überschreiten ihren jeweils erlaubten Höchstwert?' if german
                            else 'How many readings exceed their respective allowed maximum?')
            q = dict(type='score', instructions=instructions,
                     criteria=['no violations', 'one violation', 'two violations', 'three violations', 'four violations'])
            add('policy', f'local-policy/{i}', state, q, violations, 'CC0-1.0 (locally generated)')
        if len(groups['spatial']) < 128:
            origin = [rng.randint(-20, 20), rng.randint(-20, 20)]
            points = {f'point_{j}': [rng.randint(-20, 20), rng.randint(-20, 20)] for j in range(6)}
            distances = {k: abs(v[0]-origin[0])+abs(v[1]-origin[1]) for k,v in points.items()}
            order = sorted(distances, key=distances.get)
            if distances[order[0]] == distances[order[1]]:
                continue
            state = {'origin': origin, 'candidate_points': points, 'metric': 'Manhattan distance: abs(dx) + abs(dy)'}
            q = choice('Which point is nearest to the origin using the stated distance metric?', {k:None for k in points})
            add('spatial', f'local-spatial/{i}', state, q, order[0], 'CC0-1.0 (locally generated)')
        if len(groups['policy']) == len(groups['spatial']) == 128:
            break
    expected = dict(banking=128, boolq=256, paws=128, injection=128, policy=128, spatial=128, preference=128)
    if {k:len(v) for k,v in groups.items()} != expected:
        raise ValueError('Incomplete calibration groups: ' + str({k:len(v) for k,v in groups.items()}))
    # Eight-request blocks keep the 256-prefix representative of the 1,024 mix.
    order = []
    for i in range(128):
        order.extend([groups['banking'][i], groups['boolq'][2*i], groups['paws'][i],
                      groups['injection'][i], groups['policy'][i], groups['spatial'][i],
                      groups['preference'][i], groups['boolq'][2*i+1]])
    tokens_dir = args.out / 'tokens'
    tokens_dir.mkdir()
    jobs, records = [], []
    for i, row in enumerate(order):
        ident = f'cal-{i:04d}'
        path = tokens_dir / (ident + '.txt')
        path.write_text(','.join(map(str, row['tokens'])))
        jobs.append(ident + '\t' + str(path.resolve()) + '\n')
        records.append(dict(row, id=ident, token_count=len(row['tokens'])))
    (args.out / 'requests.json').write_text(json.dumps(records, ensure_ascii=False))
    (args.out / 'jobs.tsv').write_text(''.join(jobs))
    cfg = json.loads((args.model_dir / 'qev_config.json').read_text())
    (args.out / 'letters.txt').write_text(','.join(map(str, cfg['letter_token_ids'])))
    info = dict(seed=args.seed, independent_train_inputs=True, evaluation_labels_used=False,
                groups=expected, requests=1024, tokens=sum(len(x['tokens']) for x in order),
                first_256_tokens=sum(len(x['tokens']) for x in order[:256]),
                overlap_screen='normalized exact state and shared 16-word state shingles',
                rejected=dict(rejected), source_files={p.name:adapter.sha(p) for p in args.sources.iterdir() if p.is_file()})
    (args.out / 'manifest.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
