#!/usr/bin/env python3
"""Preserve exact diagonal fitting statistics and list verified disposable raw files."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from evaluation_common import sha

spec = importlib.util.spec_from_file_location('diagonal', Path(__file__).with_name('fit-dot4-quant.py'))
diagonal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagonal)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('root', type=Path)
p.add_argument('--cleanup-manifest', type=Path, required=True)
p.add_argument('--document-id', help='Compact only this completed document')
a = p.parse_args()
manifest = json.loads((a.root / 'manifest.json').read_text())
disposable = []
for document in manifest['documents']:
    if a.document_id and document.get('id') != a.document_id:
        continue
    folder = Path(document['directory'])
    target = folder / 'diagonal-statistics.npz'
    metadata_path = folder / 'diagonal-statistics.json'
    if target.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if sha(target) != metadata['statistics_sha256']:
            raise ValueError('Statistics checksum mismatch')
    else:
        capture = json.loads((folder / 'capture.json').read_text())
        if capture['version'] != 1 or capture['basis'] != 'identity':
            raise ValueError('Expected raw identity inputs')
        values, tensors = {}, []
        for tensor in capture['tensors']:
            name, basis = diagonal.hf_name(tensor['name'])
            path = folder / (tensor['name'] + '.f32')
            x = np.fromfile(path, dtype='<f4').reshape(tensor['samples'], tensor['columns'])
            if not len(x) or not np.isfinite(x).all():
                raise ValueError('Invalid capture')
            x = diagonal.rotate(x) if basis else x.astype(np.float64)
            values[name] = np.sum(x*x, axis=0)
            tensors.append(dict(name=name, basis=basis, samples=len(x), path=str(path.resolve()),
                                sha256=sha(path), bytes=path.stat().st_size))
        if len(values) != 187:
            raise ValueError('Incomplete projection inventory')
        with target.open('wb') as stream:
            np.savez(stream, **values)
        with np.load(target, allow_pickle=False) as verified:
            if set(verified.files) != set(values) or any(not np.array_equal(verified[k], v) for k, v in values.items()):
                raise ValueError('Statistics readback failed')
        metadata = dict(version=1, sign_seed=diagonal.SEED, tensors=tensors,
            statistics_sha256=sha(target), capture_sha256=sha(folder / 'capture.json'),
            limitation='Sufficient for existing diagonal MSE16; raw covariance cannot be reconstructed.')
        metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    for tensor in metadata['tensors']:
        path = Path(tensor['path'])
        if path.exists():
            disposable.append(dict(path=str(path.resolve()), bytes=tensor['bytes']))
    print('Compacted', document.get('id', str(folder)), flush=True)
a.cleanup_manifest.write_text(json.dumps(dict(root=str(a.root.resolve()), files=disposable,
    bytes=sum(f['bytes'] for f in disposable)), indent=2) + '\n')
