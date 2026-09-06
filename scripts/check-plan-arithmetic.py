#!/usr/bin/env python3
"""Check the fixed arithmetic regression and independently generate its answer."""
import argparse
import json
from pathlib import Path
import re
import struct
import subprocess
import numpy as np
from evaluation_common import sha

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=Path('benchmarks/plan-arithmetic'))
parser.add_argument('--native', type=Path, default=Path('benchmarks/plan-final-tools/qwen35_cpu.exe'))
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--label', default='candidate')
parser.add_argument('--model-dir', default='models/hf-download-test')
args = parser.parse_args()
if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in args.label):
    parser.error('Unsafe candidate label')
out = args.out
out.mkdir(parents=True, exist_ok=True)
case = json.loads(Path('configs/arithmetic-regression.json').read_text())['cases'][0]
prompt = case['prompt_tokens']
expected = case['expected_output_prefix'] + [case['expected_next_token']]
for name, tokens in [('prompt', prompt), ('targets', expected)]:
    (out/f'{name}.csv').write_text(','.join(map(str, tokens)))
binary = args.native
models = {'legacy': 'models/hf-download-test/model.q35h',
          'mse16': 'models/qwen3.5-0.8b/model-mse16.q35h',
          'calibrated': 'models/qwen3.5-0.8b/model-calibrated-mse16.q35h'}
if args.checkpoint:
    models = {args.label: str(args.checkpoint)}
rows = []
for name, checkpoint in models.items():
    command = [str(binary), '--model-dir', args.model_dir, '--weights', checkpoint,
               '--tokens-file', str(out/'prompt.csv'), '--threads', '8',
               '--max-context', str(case['max_context']), '--max-new-tokens', '32']
    for key in ('temperature', 'top_k', 'top_p', 'repetition_penalty', 'seed'):
        command += ['--'+key.replace('_', '-'), str(case[key])]
    free = subprocess.run(command, capture_output=True, encoding='utf-8', check=True)
    (out/f'{name}-free.txt').write_text(free.stdout, encoding='utf-8')
    (out/f'{name}-free.stderr').write_text(free.stderr, encoding='utf-8')
    answer = re.sub(r'^\s*<think>.*?</think>\s*', '', free.stdout, flags=re.S)
    answer = answer.replace('<|im_end|>', '').replace('<|endoftext|>', '').strip()
    dump = out/f'{name}.logits'
    forced = command + ['--forced-tokens-file', str(out/'targets.csv'), '--logits-out', str(dump)]
    result = subprocess.run(forced, capture_output=True, encoding='utf-8', check=True)
    (out/f'{name}-forced.stderr').write_text(result.stderr, encoding='utf-8')
    with dump.open('rb') as f:
        magic, version, vocab, positions = struct.unpack('<8sIIQ', f.read(24))
    assert magic == b'Q35LGT1\0' and version == 1 and positions == len(expected)
    records = np.fromfile(dump, dtype=np.dtype([('target','<i4'),('logits','<f4',(vocab,))]), offset=24)
    assert list(records['target']) == expected
    choices = []
    for index, record in enumerate(records):
        logits = record['logits'].copy()
        assert np.isfinite(logits).all()
        seen = list(set(prompt+expected[:index]))
        penalty = np.float32(case['repetition_penalty'])
        logits[seen] = np.where(logits[seen] > 0, logits[seen]/penalty, logits[seen]*penalty)
        choices.append(int(np.argmax(logits)))
    rows.append(dict(candidate=name, checkpoint_sha256=sha(Path(checkpoint)),
        commands=[command, forced], free_output=free.stdout, answer=answer,
        free_pass=answer == '4', prefix_and_answer_pass=choices == expected,
        post_penalty_argmax=choices,
        margin_4_minus_2=float(logits[19]-logits[17])))
    dump.unlink()
summary = dict(case=case, binary_sha256=sha(binary), results=rows)
(out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(rows, indent=2))
if not all(row['free_pass'] and row['prefix_and_answer_pass'] for row in rows):
    raise SystemExit('Arithmetic regression failed; see the saved summary and free outputs')
