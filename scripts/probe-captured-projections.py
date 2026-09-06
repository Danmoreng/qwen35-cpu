#!/usr/bin/env python3
"""Bounded weight/activation/kernel attribution; captured inputs are never speed results."""
import argparse
import json
from pathlib import Path
import subprocess
from evaluation_common import sha

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--capture', type=Path, default=Path('benchmarks/plan-calibration-pilot/0'))
parser.add_argument('--executable', type=Path, default=Path('build/q4_projection_probe.exe'))
parser.add_argument('--model-dir', type=Path, default=Path('models/qwen3.5-0.8b'))
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
prefix = 'model.language_model.'
projections = [('head', 'embed_tokens.weight', 'embed_tokens.weight', 'token_embd.weight',123456,123456),
    ('mlp-down', 'layers.0.mlp.down_proj.weight', 'layers.0.mlp.down_proj.weight','blk.0.ffn_down.weight',32,32),
    ('mlp-gate', 'layers.0.mlp.gate_proj.weight','layers.0.mlp.gate_up_proj.weight','blk.0.ffn_gate.weight',8,8),
    ('recurrent-a','layers.0.linear_attn.in_proj_a.weight','layers.0.linear_attn.in_proj_all.weight','blk.0.ssm_alpha.weight',0,8208)]
models = [('legacy', Path('models/hf-download-test/model.q35h')),
          ('mse16', args.model_dir/'model-mse16.q35h'),
          ('calibrated', args.model_dir/'model-calibrated-mse16.q35h')]
manifest = dict(binary_sha256=sha(args.executable), commands=[],
    scope='Eight output rows and first sixteen captured inputs per projection; diagnostic screening, not global KL attribution',
    checkpoints={name:sha(path) for name,path in models})
for name, checkpoint in models:
    for label, source, packed, captured, source_row, packed_row in projections:
        output = args.out/f'{name}-{label}.json'
        inputs = args.capture/(captured+'.f32')
        command = [str(args.executable.resolve()),str(args.model_dir),str(checkpoint),prefix+source,prefix+packed,
                   str(inputs),str(source_row),str(packed_row),str(output)]
        subprocess.run(command,check=True)
        manifest['commands'].append(dict(command=command,input_sha256=sha(inputs)))
(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
