#!/usr/bin/env python3
"""Record artifact hashes and actual tensor bit budgets for the pinned comparison."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import sys
from evaluation_common import source_identity

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, default=Path('benchmarks/comparison-2026-09-06/inputs.json'))
parser.add_argument('--gguf-dir', type=Path, default=Path('models/llama-comparison'))
parser.add_argument('--h128', type=Path, default=Path('models/hf-download-test/model.q35h'))
parser.add_argument('--hf-dir', type=Path, default=Path('models/qwen3.5-0.8b'))
parser.add_argument('--llama-source', type=Path, default=Path('.cache/llama.cpp'))
args = parser.parse_args()
sys.path.insert(0, str((args.llama_source/'gguf-py').resolve()))
from gguf import GGUFReader

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def record(path):
    return dict(path=path.as_posix(), file_bytes=path.stat().st_size, sha256=digest(path))

models = {}
for name in ['BF16', 'Q4_0-pure', 'Q4_0', 'Q4_K_M', 'IQ4_XS']:
    path = args.gguf_dir/f'Qwen3.5-0.8B-{name}.gguf'
    if not path.exists():
        continue
    reader = GGUFReader(str(path))
    info = record(path)
    info.update(tensor_bytes=sum(int(t.n_bytes) for t in reader.tensors),
                parameters=sum(int(t.n_elements) for t in reader.tensors),
                tensor_types=dict(collections.Counter(t.tensor_type.name for t in reader.tensors)),
                imatrix={k: v.contents() for k, v in reader.fields.items() if k.startswith('quantize.imatrix')})
    info['tensors'] = [dict(name=t.name, shape=[int(n) for n in reversed(t.shape)],
        parameters=int(t.n_elements), source_encoding=t.tensor_type.name, source_bytes=int(t.n_bytes),
        execution_encoding=('DOT4' if t.tensor_type.name == 'Q4_0' else t.tensor_type.name),
        packed_bytes=int(t.n_bytes), projection_group=t.name.rsplit('.', 1)[0],
        activation_scheme=('Q8-G32' if t.tensor_type.name in ('Q4_0', 'Q8_0') else
                           'signed-Q8-G256' if t.tensor_type.name in ('Q4_K','Q5_K','Q6_K') else 'float'),
        observed_kernel=None) for t in reader.tensors]
    models[name] = info
path = args.h128
info = record(path)
types = collections.Counter()
payload = parameters = 0
inventory = []
with path.open('rb') as f:
    assert f.read(8) == b'Q35H128\0'
    version, endian, header, alignment = struct.unpack('<4I', f.read(16))
    assert version == 1 and endian == 0x01020304
    count, directory = struct.unpack('<2Q', f.read(16))
    f.seek(directory)
    for _ in range(count):
        name_size, rank, encoding, transform, scale, reserved = struct.unpack('<6I', f.read(24))
        seed, elements, offset, size, checksum = struct.unpack('<5Q', f.read(40))
        tensor_name = f.read(name_size).decode()
        shape = list(struct.unpack('<'+'Q'*rank, f.read(8*rank)))
        inventory.append(dict(name=tensor_name, shape=shape, parameters=elements,
            source_encoding=encoding, source_bytes=size, execution_encoding=encoding, packed_bytes=size,
            activation_scheme='H128-Q8-G32' if transform else 'Q8-G32' if scale else 'float',
            sign_seed=seed, projection_group=tensor_name.rsplit('.', 1)[0], observed_kernel=None))
        parameters += elements
        payload += size
        types[str(encoding)] += 1
info.update(tensor_bytes=payload, parameters=parameters, tensor_types=dict(types), tensors=inventory)
models['H128-Q4'] = info
for info in models.values():
    info['bits_per_parameter'] = info['tensor_bytes'] * 8 / info['parameters']
binary_paths = list(Path('build-llama/bin').glob('*.dll')) + [
    Path('build-llama/bin/llama-fixed-cpu-bench.exe'), Path('build-llama/bin/llama-quantize.exe'),
    Path('build/qwen35_cpu.exe'), Path('build/qwen35_cpu_bench.exe')]
metadata = dict(models=models, binaries=[record(path) for path in binary_paths],
                llama_revision=source_identity(args.llama_source)['revision'],
                gguf_repo='unsloth/Qwen3.5-0.8B-GGUF', gguf_revision='6ab461498e2023f6e3c1baea90a8f0fe38ab64d0',
                h128_repo='danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4', h128_revision='cc7df08da7ef7ac15db62e80b4eda85e19a143da',
                fixture=record(Path('configs/qwen3_5_0_8b_text_1024x512_tokens.csv')),
                tokenizer=record(Path('models/hf-download-test/tokenizer.json')))
metadata['h128_quantization'] = json.loads(Path('models/hf-download-test/quantization.json').read_text())
metadata['quality_tokenizer_files'] = [record(Path('models/hf-download-test') / name)
                                      for name in ['tokenizer.json', 'tokenizer_config.json']]
metadata['python'] = sys.version
sys.path.insert(0, str(Path('.cache/python-deps').resolve()))
metadata['python_packages'] = {name: importlib.metadata.version(name)
                              for name in ['numpy', 'transformers', 'tokenizers', 'huggingface_hub', 'pyarrow']}
metadata['build_flags'] = {}
for build in ['build', 'build-llama']:
    ninja = Path(build) / 'build.ninja'
    metadata['build_flags'][build] = sorted({line.strip() for line in ninja.read_text().splitlines()
                                            if line.strip().startswith(('FLAGS =', 'DEFINES ='))})
# Optional audit against the BF16 conversion used in the parent project.
reference = Path('../vibe-inference/models/gguf/qwen3.5-0.8b-review-target-bf16.gguf')
if reference.exists():
    import numpy as np
    old = GGUFReader(str(reference))
    new = GGUFReader('models/llama-comparison/Qwen3.5-0.8B-BF16.gguf')
    old_tensors = {t.name: t for t in old.tensors}
    differences = []
    exact = collections.Counter()
    assert len(old.tensors) == len(new.tensors)
    for tensor in new.tensors:
        previous = old_tensors[tensor.name]
        assert previous.tensor_type == tensor.tensor_type
        if np.array_equal(previous.data, tensor.data):
            exact[tensor.tensor_type.name] += 1
        else:
            assert tensor.tensor_type.name == 'F32' and tensor.name.endswith('ssm_a'), tensor.name
            differences.append(dict(name=tensor.name, max_abs=float(np.max(np.abs(previous.data - tensor.data)))))
    metadata['source_bf16_verification'] = dict(reference=record(reference),
        exact_tensor_types=dict(exact), derived_f32_differences=differences)
metadata['source'] = source_identity(Path('.'))
metadata['base_shards'] = [record(p) for p in sorted(args.hf_dir.glob('*.safetensors'))]
output = args.out
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(metadata, indent=2) + '\n')
for name, info in models.items():
    print(f"{name}: {info['tensor_bytes']} payload bytes, {info['bits_per_parameter']:.6f} bits/parameter")
