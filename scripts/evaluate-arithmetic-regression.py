#!/usr/bin/env python3
"""Revisit a previously selected arithmetic regression; do not mix into held-out scores."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

root = Path('benchmarks/comparison-2026-09-06/arithmetic-regression')
root.mkdir(parents=True, exist_ok=True)
case = json.loads(Path('configs/arithmetic-regression.json').read_text())['cases'][0]
prompt = case['prompt_tokens']
forced = case['expected_output_prefix'] + [case['expected_next_token']]
(root / 'prompt.csv').write_text(','.join(map(str, prompt)))
(root / 'targets.csv').write_text(','.join(map(str, forced)))
models = ['BF16', 'H128-Q4', 'Q4_0-pure', 'Q4_0', 'Q4_K_M', 'IQ4_XS']
rows = []
for model in models:
    dump = root / (model + '.logits')
    if model == 'H128-Q4':
        command = ['build/qwen35_cpu.exe', '--model-dir', 'models/hf-download-test',
                   '--weights', 'models/hf-download-test/model.q35h', '--tokens-file', str(root / 'prompt.csv'),
                   '--forced-tokens-file', str(root / 'targets.csv'), '--threads', '12',
                   '--max-context', str(case['max_context'])]
    else:
        command = ['build-llama/bin/llama-fixed-cpu-bench.exe', '--cpu-gguf',
                   f'models/llama-comparison/Qwen3.5-0.8B-{model}.gguf',
                   '--prompt-tokens-file', str(root / 'prompt.csv'), '--forced-output-tokens',
                   ','.join(map(str, forced)), '--cpu-threads', '12', '--max-context', str(case['max_context']),
                   '--max-new-tokens', str(len(forced)), '--profile-json', str(root / (model + '-profile.json'))]
    command += ['--logits-out', str(dump)]
    with (root / (model + '.log')).open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    dtype = np.dtype([('target', '<i4'), ('logits', '<f4', (248320,))])
    records = np.fromfile(dump, dtype=dtype, offset=24)
    assert len(records) == len(forced) and list(records['target']) == forced
    choices = []
    for index, record in enumerate(records):
        logits = record['logits'].copy()
        seen = list(set(prompt + forced[:index]))
        penalty = np.float32(case['repetition_penalty'])
        logits[seen] = np.where(logits[seen] > 0, logits[seen] / penalty, logits[seen] * penalty)
        choices.append(int(np.argmax(logits)))
    margin = float(logits[case['expected_next_token']] - logits[case['incorrect_q4_0_next_token']])
    row = dict(model=model, post_penalty_argmax=choices, margin_4_minus_2=margin)
    if model != 'BF16':
        subprocess.run([sys.executable, 'scripts/compare-logit-dumps.py', '--teacher', str(root / 'BF16.logits'),
                        '--candidate', str(dump), '--json', str(root / (model + '-quality.json')),
                        '--csv', str(root / (model + '-positions.csv'))], check=True, stdout=subprocess.DEVNULL)
        quality = json.loads((root / (model + '-quality.json')).read_text())
        row['mean_kl'] = quality['kld']['mean']
        dump.unlink()
    rows.append(row)
(root / 'BF16.logits').unlink()
(root / 'summary.json').write_text(json.dumps(dict(case=case, results=rows), indent=2) + '\n')
print(json.dumps(rows, indent=2))
