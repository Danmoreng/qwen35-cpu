#!/usr/bin/env python3
"""Prove that head experiments change exactly one tied tensor."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from evaluation_common import sha


def inventory(path):
    with path.open('rb') as f:
        if f.read(8) != b'Q35H128\0':
            raise ValueError('Invalid artifact magic')
        if struct.unpack('<4I', f.read(16)) != (1, 0x01020304, 96, 64):
            raise ValueError('Unsupported artifact header')
        count, directory, _, _, _ = struct.unpack('<5Q', f.read(40))
        f.seek(directory)
        result = {}
        for _ in range(count):
            length, rank, encoding, transform, group, _, seed, elements, offset, size, checksum = struct.unpack('<6I5Q', f.read(64))
            name = f.read(length).decode()
            shape = list(struct.unpack('<'+'Q'*rank, f.read(rank*8)))
            result[name] = dict(encoding=encoding, transform=transform, group=group, seed=seed,
                elements=elements, offset=offset, bytes=size, checksum=checksum, shape=shape)
        for tensor in result.values():
            f.seek(tensor['offset'])
            remaining = tensor['bytes']
            h = hashlib.sha256()
            while remaining:
                block = f.read(min(remaining, 1024*1024))
                if not block:
                    raise ValueError('Truncated tensor')
                h.update(block)
                remaining -= len(block)
            tensor['sha256'] = h.hexdigest()
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('candidate', type=Path)
    p.add_argument('--baseline', type=Path, default=Path('models/qwen3.5-0.8b/model-mse16.q35h'))
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    base, candidate = inventory(a.baseline), inventory(a.candidate)
    head = 'model.language_model.embed_tokens.weight'
    if base.keys() != candidate.keys():
        raise ValueError('Tensor inventory changed')
    for name, tensor in candidate.items():
        if name != head and any(tensor[k] != base[name][k]
                               for k in tensor if k != 'offset'):
            raise ValueError('Non-head tensor changed: '+name)
    recipe_path = Path(str(a.candidate)+'.quantization.json')
    calibration_path = Path(str(a.candidate)+'.calibration.json')
    result = dict(baseline_sha256=sha(a.baseline), candidate_sha256=sha(a.candidate),
        candidate_file_bytes=a.candidate.stat().st_size,
        candidate_tensor_bytes=sum(t['bytes'] for t in candidate.values()),
        head_only_change=True, non_head_payloads='byte-identical to the specified baseline',
        quantization_recipe=json.loads(recipe_path.read_text()) if recipe_path.exists() else None,
        calibration_manifest_sha256=sha(calibration_path) if calibration_path.exists() else None,
        tensors=candidate)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
