#!/usr/bin/env python3
"""Prepare an allowlisted, independently usable model repository. No upload."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for data in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engine-url', default='https://github.com/Danmoreng/qwen35-cpu')
    args = p.parse_args()
    required = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                'vocab.json', 'merges.txt', 'chat_template.jinja', 'LICENSE']
    for name in required:
        if not (args.source / name).is_file():
            raise ValueError(f'Missing source file: {name}')
    if not args.artifact.is_file():
        raise ValueError('Missing packed artifact')
    # Only publish the already validated artifact until a new recipe is evaluated.
    artifact_hash = digest(args.artifact)
    if artifact_hash != 'e73de30bf646dee502dd5e519939221f7d2b60068ccbbe16dc1701597919c42f':
        raise ValueError('Artifact differs from the validated release candidate')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(args.artifact, args.output / 'model.q35h')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_hashes = {file.name: digest(file) for file in sorted(args.source.glob('*.safetensors'))}
    recipe = dict(format='q35h', recipe='H128/Q4-G32-DOT4', transform_size=128,
                  scale_group=32, artifact_sha256=artifact_hash,
                  base_model='Qwen/Qwen3.5-0.8B', source_shard_sha256=source_hashes,
                  engine_revision=revision, source_revision='not recorded; use source shard hashes',
                  text_only=True, kv_cache='fp16', recurrent_state='fp32')
    (args.output / 'quantization.json').write_text(json.dumps(recipe, indent=2) + '\n', encoding='utf-8')
    notice = ('Derived from Qwen/Qwen3.5-0.8B by Qwen, licensed under Apache-2.0.\n'
              'Modifications: text-only extraction, H128/Q4-G32 quantization and CPU DOT4 packing.\n'
              'Vision and MTP/draft weights are omitted. Original tokenizer/configuration retained.\n')
    if (args.source / 'NOTICE').is_file():
        notice += '\nOriginal NOTICE:\n' + (args.source / 'NOTICE').read_text(encoding='utf-8')
    (args.output / 'NOTICE').write_text(notice, encoding='utf-8')
    card = f'''---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- cpu
- quantized
- qwen3.5
- custom-format
---
# Qwen3.5-0.8B H128/Q4-G32-DOT4

CPU-ready **text-only** quantization of Qwen/Qwen3.5-0.8B for the
[Qwen3.5 CPU engine]({args.engine_url}). Download a matching Windows or Linux
server from its Releases page, then download this repository's files into one
model directory. No BF16 checkpoint or conversion step is required at runtime.

```sh
./qwen35_cpu_server --model-dir /path/to/downloaded-model --threads 8
```

Use `qwen35_cpu_server.exe` on Windows. The server listens on localhost:8080.
See the engine README for `/v1/completions`, downloads with revision pinning,
checksums and thread guidance. Raw completion is supported; render chat prompts
before sending them. Streaming and a chat-completions endpoint are not included.

## Format and scope

This is a custom `.q35h` artifact, **not GGUF or a Transformers-loadable checkpoint**.
Large projections use a 128-value signed Hadamard transform and Q4 groups of 32.
Embeddings use the recipe's Q4 encoding; prescribed small tensors retain FP32.
The nibble representation is prepacked for CPU DOT4 kernels. No image input,
vision encoder, MTP/draft execution or GPU backend is included.

Runtime uses FP16 KV caches and FP32 recurrent state. File size is
{args.artifact.stat().st_size:,} bytes. SHA-256: `{artifact_hash}`.
See `quantization.json` for source shard hashes and engine provenance;
`SHA256SUMS` covers all files needed by the runtime and model documentation.

## Quality and performance

Perplexity and KL comparisons against BF16 and llama.cpp Q4_0/Q4_K_M/IQ4_XS
are pending. No quality advantage is claimed from the format's name or bit width.
Historical speed measurements in the engine README identify their CPU, upstream
revision and workload; they are not universal speed guarantees.

## License

The original model's Apache-2.0 license is included as `LICENSE`, with modification
attribution in `NOTICE`. The inference engine is separately MIT-licensed.
'''
    (args.output / 'README.md').write_text(card, encoding='utf-8')
    manifest = ''.join(f'{digest(file)}  {file.name}\n' for file in sorted(args.output.iterdir()) if file.is_file())
    (args.output / 'SHA256SUMS').write_bytes(manifest.encode('ascii'))
    print(f'Prepared {args.output}; no files uploaded.')


if __name__ == '__main__':
    main()
