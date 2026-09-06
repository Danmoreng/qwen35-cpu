#!/usr/bin/env python3
"""Record artifact hashes and actual tensor bit budgets for the pinned comparison."""
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import struct
import subprocess
import sys

sys.path.insert(0, str(Path('.cache/llama.cpp/gguf-py').resolve()))
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
    path = Path(f'models/llama-comparison/Qwen3.5-0.8B-{name}.gguf')
    reader = GGUFReader(str(path))
    info = record(path)
    info.update(tensor_bytes=sum(int(t.n_bytes) for t in reader.tensors),
                parameters=sum(int(t.n_elements) for t in reader.tensors),
                tensor_types=dict(collections.Counter(t.tensor_type.name for t in reader.tensors)),
                imatrix={k: v.contents() for k, v in reader.fields.items() if k.startswith('quantize.imatrix')})
    models[name] = info
path = Path('models/hf-download-test/model.q35h')
info = record(path)
types = collections.Counter()
payload = parameters = 0
with path.open('rb') as f:
    assert f.read(8) == b'Q35H128\0'
    version, endian, header, alignment = struct.unpack('<4I', f.read(16))
    assert version == 1 and endian == 0x01020304
    count, directory = struct.unpack('<2Q', f.read(16))
    f.seek(directory)
    for _ in range(count):
        name_size, rank, encoding, transform, scale, reserved = struct.unpack('<6I', f.read(24))
        seed, elements, offset, size, checksum = struct.unpack('<5Q', f.read(40))
        f.read(name_size + 8 * rank)
        parameters += elements
        payload += size
        types[str(encoding)] += 1
info.update(tensor_bytes=payload, parameters=parameters, tensor_types=dict(types))
models['H128-Q4'] = info
for info in models.values():
    info['bits_per_parameter'] = info['tensor_bytes'] * 8 / info['parameters']
binary_paths = list(Path('build-llama/bin').glob('*.dll')) + [
    Path('build-llama/bin/llama-fixed-cpu-bench.exe'), Path('build-llama/bin/llama-quantize.exe'),
    Path('build/qwen35_cpu.exe'), Path('build/qwen35_cpu_bench.exe')]
metadata = dict(models=models, binaries=[record(path) for path in binary_paths],
                llama_revision=subprocess.check_output(['git', '-C', '.cache/llama.cpp', 'rev-parse', 'HEAD'], text=True).strip(),
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
output = Path('benchmarks/comparison-2026-09-06/inputs.json')
output.write_text(json.dumps(metadata, indent=2) + '\n')
for name, info in models.items():
    print(f"{name}: {info['tensor_bytes']} payload bytes, {info['bits_per_parameter']:.6f} bits/parameter")
