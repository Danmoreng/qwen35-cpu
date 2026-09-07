"""Accumulate bounded block moments without retaining per-document raw captures."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from evaluation_common import sha

spec = importlib.util.spec_from_file_location('diagonal', Path(__file__).with_name('fit-dot4-quant.py'))
diagonal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagonal)


class BlockAccumulator:
    def __init__(self, root):
        self.root = Path(root)
        self.values = {}
        self.tensors = {}
        self.documents = []

    def add(self, document):
        folder = Path(document['directory'])
        capture = json.loads((folder / 'capture.json').read_text())
        if capture['version'] != 1 or capture['basis'] != 'identity':
            raise ValueError('Expected identity captures')
        names = set()
        with threadpool_limits(limits=2):
            for tensor in capture['tensors']:
                name, basis = diagonal.hf_name(tensor['name'])
                names.add(name)
                path = folder / (tensor['name'] + '.f32')
                x = np.fromfile(path, dtype='<f4').reshape(tensor['samples'], tensor['columns'])
                if not len(x) or x.shape[1] % 128 or not np.isfinite(x).all():
                    raise ValueError('Invalid covariance input')
                x = diagonal.rotate(x) if basis else x.astype(np.float64)
                blocks = x.reshape(len(x), -1, 128).transpose(1, 0, 2)
                h = blocks.transpose(0, 2, 1) @ blocks
                if name not in self.values:
                    self.values[name] = h
                    self.tensors[name] = dict(name=name, basis=basis, samples=0)
                self.tensors[name]['samples'] += len(x)
                if self.values[name] is not h:
                    self.values[name] += h
        if len(names) != 187 or names != set(self.values):
            raise ValueError('Incomplete covariance inventory')
        self.documents.append(dict(id=document['id'], capture_sha256=sha(folder / 'capture.json')))
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.root / 'block-statistics.tmp'
        with temporary.open('wb') as stream:
            np.savez(stream, **self.values)
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != names or any(not np.array_equal(archive[k], v) for k, v in self.values.items()):
                raise ValueError('Block statistics readback failed')
        target = self.root / 'block-statistics.npz'
        temporary.replace(target)
        metadata = dict(version=1, sign_seed=diagonal.SEED, statistics_sha256=sha(target),
                        documents=self.documents, tensors=list(self.tensors.values()))
        temporary = self.root / 'block-statistics.json.tmp'
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(self.root / 'block-statistics.json')
