#!/usr/bin/env python3
"""Matched uncalibrated head candidates; timing uses the sequential harness."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--extended', action='store_true')
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
tokens = [int(x) for x in Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv').read_text().strip().split(',')]
models = [('mse16', 'model-mse16.q35h'), ('g16-head', 'model-mse16-g16-head.q35h'),
          ('h128-g16-head', 'model-mse16-h128-g16-head.q35h'),
          ('calibrated', 'model-calibrated-mse16.q35h')]
workloads = [(1, 512, 128)]
if a.extended:
    models = models[:2]
    workloads = [(1, 4096, 2), (1, 8192, 128), (16, 512, 128)]
cases = []
for batch, prompt, output in workloads:
    path = a.out/f'prompt-{prompt}.csv'
    path.write_text(','.join(str(tokens[i % len(tokens)]) for i in range(prompt)))
    for label, checkpoint in models:
        cases.append(dict(name=f'{label}-b{batch}-t8-p{prompt}-n{output}',
            executable='benchmarks/head-tools/qwen35_cpu_bench.exe', args=[
                '--hf-model-dir', 'models/hf-download-test', '--cpu-q4-h128',
                'models/qwen3.5-0.8b/'+checkpoint, '--cpu-batch-full-logits',
                '--prompt-tokens-file', str(path), '--forced-output-tokens',
                ','.join(str(tokens[(prompt+i) % len(tokens)]) for i in range(output)),
                '--cpu-threads', '8', '--cpu-batch', str(batch), '--max-context', '16384',
                '--max-new-tokens', str(output)]))
(a.out/'matrix.json').write_text(json.dumps(cases, indent=2)+'\n')
