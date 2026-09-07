#!/usr/bin/env python3
"""Verify that an offline fitter preserves G32 runtime layout and F32 tensors."""
import argparse
import importlib.util
import json
from pathlib import Path
from evaluation_common import sha

spec = importlib.util.spec_from_file_location('artifact', Path(__file__).with_name('audit-head-artifact.py'))
artifact = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifact)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('candidate', type=Path)
p.add_argument('--baseline', type=Path, default=Path('models/qwen3.5-0.8b/model-calibrated-mse16.q35h'))
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
base, candidate = artifact.inventory(a.baseline), artifact.inventory(a.candidate)
if base.keys() != candidate.keys() or a.baseline.stat().st_size != a.candidate.stat().st_size:
    raise ValueError('Artifact layout changed')
for name, tensor in candidate.items():
    for key in ('encoding', 'transform', 'group', 'seed', 'elements', 'offset', 'bytes', 'shape'):
        if tensor[key] != base[name][key]:
            raise ValueError(f'Layout changed: {name} / {key}')
    if tensor['encoding'] == 0 and tensor['sha256'] != base[name]['sha256']:
        raise ValueError('F32 tensor changed: ' + name)
changed = [name for name in candidate if candidate[name]['sha256'] != base[name]['sha256']]
a.out.write_text(json.dumps(dict(layout_identical=True, f32_unchanged=True,
    changed_tensors=changed, unchanged_tensor_count=len(candidate)-len(changed),
    baseline_sha256=sha(a.baseline), candidate_sha256=sha(a.candidate),
    candidate_bytes=a.candidate.stat().st_size), indent=2) + '\n')
print(f'Identical layout; {len(changed)} changed quantized tensors.')
