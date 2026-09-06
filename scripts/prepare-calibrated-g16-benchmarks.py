#!/usr/bin/env python3
"""Calibrated G16 acceptance: compare the same batches with no slowdown budget."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--executable', required=True)
p.add_argument('--extended', action='store_true')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
tokens = [int(x) for x in Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv').read_text().strip().split(',')]
models = [('calibrated-g32', 'model-calibrated-mse16.q35h'),
          ('calibrated-g16', 'model-calibrated-g16-head.q35h')]
workloads = [(b, 512, 128) for b in (1, 2, 4, 8, 16)]
if a.extended:
    workloads = [(1, 4096, 2), (1, 8192, 128)]
matrix = []
for batch, prompt, output in workloads:
    path = a.out/f'prompt-{prompt}.csv'
    path.write_text(','.join(str(tokens[i % len(tokens)]) for i in range(prompt)))
    for label, checkpoint in models:
        matrix.append(dict(name=f'{label}-b{batch}-t8-p{prompt}-n{output}', executable=a.executable,
            args=['--hf-model-dir', 'models/hf-download-test', '--cpu-q4-h128',
                'models/qwen3.5-0.8b/'+checkpoint, '--cpu-batch-full-logits',
                '--prompt-tokens-file', str(path), '--forced-output-tokens',
                ','.join(str(tokens[(prompt+i) % len(tokens)]) for i in range(output)),
                '--cpu-threads', '8', '--cpu-batch', str(batch), '--max-context', '16384',
                '--max-new-tokens', str(output)]))
(a.out/'matrix.json').write_text(json.dumps(matrix, indent=2)+'\n')
(a.out/'contract.json').write_text(json.dumps(dict(batch_gate='Candidate median must not be lower than matched G32; collect balanced repetitions if noise prevents a conclusion.',
    slowdown_budget=0, single_request_decode_minimum=100,
    quality='Assess directed teacher KL and token-weighted PPL separately; keep the published default until quality and speed support promotion.'), indent=2)+'\n')
