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
    # Publishing remains restricted to the evaluated standard checkpoint.
    artifact_hash = digest(args.artifact)
    if artifact_hash != '8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d':
        raise ValueError('Artifact differs from the validated release candidate')
    quantization_path = Path(str(args.artifact)+'.quantization.json')
    calibration_path = Path(str(args.artifact)+'.calibration.json')
    quantization = json.loads(quantization_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if quantization['quantizer'] != 'mse16' or quantization['importance'] != 'Q35CAL1':
        raise ValueError('Expected calibrated MSE16 provenance')
    args.output.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copyfile(args.source / name, args.output / name)
    shutil.copyfile(args.artifact, args.output / 'model.q35h')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_hashes = {file.name: digest(file) for file in sorted(args.source.glob('*.safetensors'))}
    recipe = dict(format='q35h', recipe='H128/Q4-G32-DOT4', quantizer='activation-weighted-mse16',
                  format_changed=False, transform_size=128,
                  scale_group=32, artifact_sha256=artifact_hash,
                  base_model='Qwen/Qwen3.5-0.8B', source_shard_sha256=source_hashes,
                  engine_revision=revision, source_revision='not recorded; use source shard hashes',
                  text_only=True, kv_cache='fp16', recurrent_state='fp32',
                  quantizer_parameters=quantization, calibration_provenance=calibration,
                  calibration_sidecar_sha256=digest(calibration_path),
                  validation_report=args.engine_url+'/blob/'+revision+'/docs/implementation-plan-results-2026-09-06.md')
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

The standard artifact now uses **calibrated MSE16** quantization. The `.q35h`
format, H128 transform, Q4 groups and DOT4 runtime layout are unchanged.
Legacy weights remain available at repository revision
`cc7df08da7ef7ac15db62e80b4eda85e19a143da`.

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

Same 8,192 scored tokens in 16 WikiText-2 article windows, with a common BF16
teacher; this is an English-prose subset, not full-corpus WikiText perplexity.

| Artifact | Perplexity | Mean KL to BF16 |
|---|---:|---:|
| Legacy H128 | 18.58329 | 0.174841 |
| Unweighted MSE16 H128 | 16.86128 | 0.105864 |
| **This calibrated MSE16 H128 artifact** | **16.04087** | **0.096456** |
| Native Q4_K_M (larger tensor payload) | 14.78432 | 0.034708 |

Compared with Legacy, perplexity decreases **13.68%** at the same
424,934,656-byte tensor payload. On a separate six-document, 1,024-token
screening suite, PPL decreases from 5.16899 to 4.85143. Unweighted MSE16 is
slightly better on that small suite (4.81958). Calibration covers four disjoint
English prose documents; neither suite establishes universal quality superiority.

Ryzen 9 9955HX3D, eight threads, affinity `0x5555`, FP16 KV, fixed tokens and
full logits: median **2,150.69 tok/s prefill** and **118.35 tok/s decode** for
one request with 512 input / 128 output tokens (one warmup, three measured runs).
Legacy measures 2,147.90 / 117.79 tok/s in the same series. A separate six-run
follow-up finds calibrated decode within -0.33% to +0.52% of Legacy for batch 16
and long context. Batched throughput is compared only at the same batch size.

The existing German `2+2` regression independently generates **4**, without
forced output tokens. This is one regression, not general arithmetic validation.

MSE16 searches both scale signs and the full `[-8,7]` code range, evaluates
stored FP16 scales, and minimizes squared reconstruction error. Calibration
weights the fitting objective using actual teacher activations in the correct
H128 basis (identity for the tied head). These changes affect offline conversion;
inference uses the same weight blocks and kernels. Q4_K_M-level quality at Legacy
speed remains an open target. See the
[full report and raw measurements]({args.engine_url}/blob/{revision}/docs/implementation-plan-results-2026-09-06.md).

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
