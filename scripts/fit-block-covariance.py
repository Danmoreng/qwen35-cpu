#!/usr/bin/env python3
"""Prepare 128-channel second moments for a bounded G32 error-compensation fit."""
import argparse
import importlib.util
import json
from pathlib import Path
import struct
import numpy as np
from threadpoolctl import threadpool_limits
from evaluation_common import sha

spec = importlib.util.spec_from_file_location('diagonal', Path(__file__).with_name('fit-dot4-quant.py'))
diagonal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagonal)


def fit(root, out, damping):
    compact = root / 'block-statistics.json'
    manifest = json.loads((compact if compact.exists() else root / 'manifest.json').read_text())
    statistics = {}
    inputs = []
    if compact.exists():
        archive_path = root / 'block-statistics.npz'
        if manifest['sign_seed'] != diagonal.SEED or sha(archive_path) != manifest['statistics_sha256']:
            raise ValueError('Invalid compact block statistics')
        with np.load(archive_path, allow_pickle=False) as archive:
            for tensor in manifest['tensors']:
                h = archive[tensor['name']].copy()
                if not np.isfinite(h).all() or h.shape[1:] != (128, 128) or tensor['samples'] <= 0:
                    raise ValueError('Invalid second moments')
                statistics[tensor['name']] = [h, tensor['samples'], tensor['basis']]
        inputs.append(dict(path=str(archive_path), sha256=sha(archive_path)))
    for document in ([] if compact.exists() else manifest['documents']):
        folder = Path(document['directory'])
        capture = json.loads((folder / 'capture.json').read_text())
        if capture['version'] != 1 or capture['basis'] != 'identity':
            raise ValueError('Expected identity-basis teacher captures')
        for tensor in capture['tensors']:
            name, basis = diagonal.hf_name(tensor['name'])
            path = folder / (tensor['name'] + '.f32')
            x = np.fromfile(path, dtype='<f4').reshape(tensor['samples'], tensor['columns'])
            if not len(x) or not np.isfinite(x).all() or x.shape[1] % 128:
                raise ValueError('Invalid capture')
            x = diagonal.rotate(x) if basis else x.astype(np.float64)
            blocks = x.reshape(len(x), -1, 128).transpose(1, 0, 2)
            h = blocks.transpose(0, 2, 1) @ blocks
            if name not in statistics:
                statistics[name] = [h, len(x), basis]
            else:
                statistics[name][0] += h
                statistics[name][1] += len(x)
            inputs.append(dict(path=str(path), sha256=sha(path)))
    if len(statistics) != 187:
        raise ValueError('Expected 187 projections')
    out.mkdir(parents=True, exist_ok=False)
    outputs = []
    for name, (h, count, basis) in sorted(statistics.items()):
        h /= count
        # Normalize each block; this preserves its reconstruction objective.
        energy = np.diagonal(h, axis1=1, axis2=2).mean(axis=1)
        h /= np.where(energy > 0, energy, 1)[:, None, None]
        regularized = h + damping * np.eye(128)[None, :, :]
        inverse = np.linalg.inv(regularized)
        upper = np.linalg.cholesky((inverse + inverse.transpose(0, 2, 1)) * 0.5).transpose(0, 2, 1)
        values = np.stack([h, upper], axis=1).astype('<f4')
        if not np.isfinite(values).all():
            raise ValueError('Invalid covariance factor')
        path = out / (name + '.cov')
        path.write_bytes(struct.pack('<8sQQIQ', b'Q35COV1\0', len(h) * 128,
                         diagonal.SEED if basis else 0, basis, count) + values.tobytes())
        outputs.append(dict(name=name, samples=count, blocks=len(h), sha256=sha(path)))
    (out / 'manifest.json').write_text(json.dumps(dict(version=1,
        recipe='g32-error-compensation-block128', damping=damping,
        limitation='Cross-block covariance omitted; teacher inputs, not sequential layer recalibration.',
        source_manifest_sha256=sha(compact if compact.exists() else root / 'manifest.json'), inputs=inputs, tensors=outputs), indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--damping', type=float, default=0.01)
    a = p.parse_args()
    if not 0 < a.damping <= 1:
        p.error('damping must be in (0,1]')
    with threadpool_limits(limits=2):
        fit(a.root, a.out, a.damping)
