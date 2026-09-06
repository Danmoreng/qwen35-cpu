#!/usr/bin/env python3
"""Check independent sequence state against serial full-vocabulary logits (not a speed run)."""
import json
from pathlib import Path
import subprocess
import numpy as np

root = Path('benchmarks/comparison-2026-09-06/batch-validation')
root.mkdir(parents=True, exist_ok=True)
tokens = list(map(int, Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv').read_text().strip().split(',')))
prompt, forced = tokens[:64], tokens[64:80]
common = ['build-llama/bin/llama-fixed-cpu-bench.exe', '--cpu-gguf',
          'models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf', '--cpu-threads', '8',
          '--forced-output-tokens', ','.join(map(str, forced)), '--max-new-tokens', str(len(forced)),
          '--max-context', '1024']

def run(name, prompt, batch):
    logits = root / (name + '.f32')
    command = common + ['--cpu-batch', str(batch), '--prompt-tokens', ','.join(map(str, prompt)),
                        '--profile-json', str(root / (name + '.json')), '--final-logits-out', str(logits)]
    with (root / (name + '.log')).open('wb') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = np.fromfile(logits, dtype='<f4').reshape(batch, 248320).astype(np.float64)
    assert np.isfinite(result).all()
    return result

serial = []
for row in range(16):
    row_prompt = prompt.copy()
    row_prompt[-1] = (row_prompt[-1] + row) % 248320
    serial.append(run(f'serial-{row}', row_prompt, 1)[0])
serial = np.array(serial)
results = []
for batch in (4, 16):
    actual = run(f'batch-{batch}', prompt, batch)
    delta = actual - serial[:batch]
    # Batched quantized matrix kernels need not be bit-identical. Detect wrong
    # sequence routing/state with a tight full-vocabulary error bound.
    maximum = float(np.max(np.abs(delta)))
    rmse = float(np.sqrt(np.mean(delta * delta)))
    agreement = float(np.mean(np.argmax(actual, axis=1) == np.argmax(serial[:batch], axis=1)))
    results.append(dict(batch=batch, max_abs=maximum, rmse=rmse, argmax_agreement=agreement))
    if maximum > 0.02 or rmse > 0.002 or agreement != 1:
        raise AssertionError(results[-1])
(root / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
