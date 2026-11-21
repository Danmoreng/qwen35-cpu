"""Recreate the panels and sequential HTTP benchmark matrix from the repository root."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--server', type=Path, required=True)
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
root = Path(__file__).resolve().parents[3]
fixtures = root / 'docs/results/qev-api-2026-10-21'
tickets = json.loads((root / 'tests/data/finetunes/qev.json').read_text())
items = [c['fields']['state'] for c in tickets['cases']]
question = tickets['qev_question']
rubric = (
    'Support triage policy. Assign the ticket to the team responsible for the next action. '
    'Account support restores access and handles sign-in problems. Billing investigates posted charges, '
    'invoices and refunds. Sales prepares quotations and advises on upgrades before purchase. '
    'Technical support investigates reproducible product faults, crashes and export failures. '
    'Where several teams are mentioned, use the action the customer needs today. '
    'For example, a missing invoice for an existing subscription goes to billing even if the customer '
    'mentions changing plans next year. Select one team using the evidence in the ticket.'
)


def row(name, state, questions):
    return {'id': name, 'request': {'encoding': 'per_question', 'state': state, 'questions': questions}}


def item(i):
    return {**question, 'instructions': 'Which team should handle the following ticket?\n' + items[i]}


panels = {}
for length in ('short', 'long'):
    request = json.loads((fixtures / f'{length}-request.json').read_text())
    panels[length] = [{'id': name, 'request': request} for name in
                      ('first', 'repeat-1', 'repeat-2', 'repeat-3', 'repeat-4')]
panels['ticket-first'] = [row(f'ticket-{i}', items[i], {'dept': question}) for i in range(8)]
# Only an entire registered prefix matches. Two different single-question
# calls demonstrate the absence of an automatic snapshot at their common stem.
panels['rubric-first'] = [row(f'unprimed-{i}', rubric, {'dept': item(i)}) for i in (0, 1)]
panels['rubric-first'].append(row('prime', rubric, {'a': item(2), 'b': item(3)}))
panels['rubric-first'].extend(row(f'new-{i}', rubric, {'dept': item(i)}) for i in range(4, 12))

matrix = []
for name, rows in panels.items():
    panel = (args.out / f'{name}.jsonl').resolve()
    panel.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    for mode in ('off', 'auto'):
        matrix.append({'name': f'{name}-{mode}', 'executable': sys.executable,
                       'args': [str(root / 'scripts/benchmark-systemone-http.py'), '--backend', 'native',
                                '--server', str(args.server.resolve()), '--model-dir', str(args.model_dir.resolve()),
                                '--weights', str(args.model_dir.resolve() / 'model.q35h'), '--threads', '4',
                                '--jsonl-panel', '--request', str(panel), '--decision-head', 'auto',
                                '--prefix-cache', mode]})
(args.out / 'matrix.json').write_text(json.dumps(matrix, indent=2) + '\n')
