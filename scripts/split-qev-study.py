#!/usr/bin/env python3
"""Freeze grouped, stratified development/final panels before quantization fitting."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import shutil


def group_id(item):
    value = item['id'].replace('game_q-', 'game-').replace('game_s-', 'game-')
    return re.sub(r'-(complete|missing)$', '', value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--per-skill', type=int, default=45)
    p.add_argument('--seed', default='qev-20261021-development-v1')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    items = json.loads((args.prepared/'items.json').read_text())
    rows = json.loads((args.prepared/'rows.json').read_text())
    groups, skills = collections.defaultdict(list), collections.defaultdict(set)
    for i, item in enumerate(items):
        key = group_id(item)
        groups[key].append(i)
        skills[item['skill']].add(key)
    selected = set()
    for skill, keys in sorted(skills.items()):
        ordered = sorted(keys, key=lambda k:hashlib.sha256((args.seed+':'+k).encode()).hexdigest())
        selected.update(ordered[:args.per_skill])
    dev = {i for key in selected for i in groups[key]}
    jobs = {line.split('\t')[0]:line for line in (args.prepared/'jobs.tsv').read_text().splitlines()}
    summary = dict(seed=args.seed, per_skill=args.per_skill,
                   group_policy='same request, game question/state variants, missing/complete pairs stay together',
                   selection='stable SHA256 ordering, independent of labels/model outputs')
    for name, indices in [('development', dev), ('final', set(range(len(items)))-dev)]:
        folder = args.out/name
        folder.mkdir()
        ordered = sorted(indices)
        mapping = {old:new for new,old in enumerate(ordered)}
        panel_items = [items[i] for i in ordered]
        panel_rows = [dict(r, item=mapping[r['item']]) for r in rows if r['item'] in indices]
        (folder/'items.json').write_text(json.dumps(panel_items, ensure_ascii=False))
        (folder/'rows.json').write_text(json.dumps(panel_rows, ensure_ascii=False))
        (folder/'jobs.tsv').write_text(''.join(jobs[r['id']]+'\n' for r in panel_rows))
        shutil.copyfile(args.prepared/'letters.txt',folder/'letters.txt')
        manifest = json.loads((args.prepared/'manifest.json').read_text())
        manifest.update(items=len(panel_items),rows=len(panel_rows),tokens=sum(r['tokens'] for r in panel_rows),excluded=[])
        (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        summary[name] = dict(items=len(panel_items),rows=len(panel_rows),tokens=manifest['tokens'],
                             ids_sha256=hashlib.sha256('\n'.join(x['id'] for x in panel_items).encode()).hexdigest())
    (args.out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
