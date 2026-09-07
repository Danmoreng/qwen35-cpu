#!/usr/bin/env python3
"""Check pinned calibration selections against held-out source documents."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def spans(text):
    words = re.findall(r'\w+', text.lower())
    return {hashlib.sha256(' '.join(words[i:i + 32]).encode()).digest()
            for i in range(len(words) - 31)}


def main(root):
    documents = json.loads((root / 'selection.json').read_text())['documents']
    held = set()
    final_groups = {d['group'] for d in documents if d['split'] == 'final' and d['group']}
    for document in documents:
        if document['split'] == 'final':
            held |= spans((Path(document['folder']) / 'source.txt').read_text(encoding='utf-8'))
    for folder in ['comparison-2026-09-06', 'plan-heldout-v1', 'head-audit-v1']:
        for path in (Path('benchmarks') / folder).rglob('source.txt'):
            held |= spans(path.read_text(encoding='utf-8'))
    collisions, arithmetic, groups = [], [], []
    totals = {}
    for document in documents:
        if document['split'] != 'calibration':
            continue
        text = (Path(document['folder']) / 'source.txt').read_text(encoding='utf-8')
        if spans(text) & held:
            collisions.append(document['id'])
        if re.search(r'(?<![\w^])2\s*\+\s*2(?!\w)', text):
            arithmetic.append(document['id'])
        if document['group'] and document['group'] in final_groups:
            groups.append(document['id'])
        totals[document['domain']] = totals.get(document['domain'], 0) + document['tokens']
    report = dict(calibration_documents=sum(d['split'] == 'calibration' for d in documents),
                  tokens=sum(totals.values()), domain_tokens=totals,
                  overlapping_documents=collisions, arithmetic_documents=arithmetic,
                  cross_split_groups=groups,
                  limitation='Checks selected source text; does not establish pretraining independence.')
    (root / 'overlap-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if collisions or arithmetic or groups:
        raise ValueError('Calibration isolation audit failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
