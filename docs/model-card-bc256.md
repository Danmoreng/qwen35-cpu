---
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

CPU-ready **text-only** quantization for the
[Qwen3.5 CPU engine](@ENGINE_URL@). The current standard uses **256-document
calibration with weighted MSE16 and block-128 error compensation (B+C)**.
The custom `.q35h` storage format and CPU kernels are unchanged.

Download the matching engine from GitHub Releases and this repository's files
into one model directory. No BF16 weights, calibration, Python or CUDA are needed
at runtime. This is **not GGUF or a Transformers-loadable checkpoint**.

```sh
./qwen35_cpu_server --model-dir /path/to/downloaded-model --threads 8
```

Use `qwen35_cpu_server.exe` on Windows. The server supports raw completions at
localhost:8080; render chat prompts before sending them. Streaming and a chat
completions endpoint are not implemented. Follow the engine README for immutable
revision downloads and checksum verification.

## Format and calibration

Large projections use signed H128 transforms and groups of 32 signed four-bit
codes with FP16 scales, prepacked for CPU DOT4. The tied embedding/output matrix
uses its identity-basis Q4 encoding; prescribed small tensors remain FP32.
Runtime uses FP16 KV and FP32 recurrent state. Vision and MTP/draft weights are
omitted. The CPU runtime has no GPU backend; a separate CUDA teacher was used
only for offline calibration.

Calibration uses 256 documents / 262,144 tokens: approximately 25% German,
25% English, 20% Python code, 20% dialog and 10% mathematics. Sources are
FineWeb2, FineWeb-Edu, CodeParrot Clean, SmolTalk and OpenWebMath, with pinned
revisions and excluded evaluation documents. A capacity-256 reservoir samples
projection inputs over each 1,024-token document. The fitting method adds
within-block covariance compensation with damping 0.01 and retains the MSE16
candidate when compensation does not improve local reconstruction error.
This is not full-matrix GPTQ or sequential layer recalibration.

Tensor payload: **424,934,656 bytes**; complete artifact: **424,964,864 bytes**.
SHA256: `@SHA256@`. See `quantization.json` for source hashes, calibration and
covariance provenance. `SHA256SUMS` covers the complete package.

## Quality

Common BF16 teacher and identical 8,192 scored tokens from 16 WikiText-2 test
article windows. These are English-prose subset scores, not full WikiText PPL.
Lower is better; different quantization labels do not imply equal payloads.

| Engine / checkpoint | Tensor MB | PPL | Mean KL to BF16, nats |
|---|---:|---:|---:|
| BF16 teacher | 1,505.8 | 14.38556 | 0 |
| **This engine, H128 B+C 256** | **424.9** | **15.80467** | **0.060190** |
| llama.cpp Q4_0 pure | 424.9 | 18.02588 | 0.144522 |
| llama.cpp Unsloth Q4_0 | 496.2 | 15.58088 | 0.068388 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.75290 | 0.034693 |
| llama.cpp Unsloth IQ4_XS | 481.6 | 15.16145 | 0.050539 |

On the independently reserved ten-document, 1,280-token mixed final set, this
artifact measures **PPL 12.71898 / KL 0.064178**, versus **13.10860 / 0.106256**
for the previous calibrated MSE16 standard: **2.97% lower PPL and 39.60% lower
KL**. No refitting followed that test. Math PPL increases 0.58% despite lower KL.
The six-document development set measures **PPL 4.65283 / KL 0.046494**.

The larger calibration set does not beat the 40-document B+C control on every
suite. More data with diagonal MSE16 alone did not improve quality. These bounded
tests do not establish universal superiority; the larger Unsloth Q4_K_M and
IQ4_XS artifacts retain better English PPL and KL.

## Performance

Ryzen 9 9955HX3D, eight threads, affinity `0x5555`, FP16 KV and full logits.
Fresh sequential comparison, one warmup and three measured runs per case;
median tokens/second, excluding model load, tokenization and HTTP.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| **This engine, H128 B+C 256** | 2,147.22 | 1,877.68 | 120.05 | 619.16 |
| llama.cpp Q4_0 `--pure` | 1,073.72 | 970.00 | 101.15 | 326.13 |
| llama.cpp Unsloth Q4_0 | 892.21 | 860.68 | 88.16 | 290.07 |
| llama.cpp Unsloth Q4_K_M | 673.80 | 661.63 | 84.29 | 259.64 |
| llama.cpp Unsloth IQ4_XS | 853.14 | 827.82 | 88.76 | 259.74 |

Prefill is single-request. Decode uses P512/N128, counting 127 actual forwards
per request; batch 16 is aggregate throughput across private requests. Batch
speedups compare identical batch sizes. Three runs are not a universal guarantee.

A separate matched G32 series finds decode medians within -0.13% to +0.76% of
the previous standard across B1/2/4/8/16, with prefill changes within 0.88%.

The artifact also passes prefix, scheduler, paged scheduler and arithmetic
regressions. One arithmetic fixture is not general mathematical validation.
See the [quality study](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/g32-large-calibration-2026-09-07.md)
and [public comparison](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/readme-comparison-2026-09-07.md)
for raw results, commands, hashes and limitations.

## Revisions and license

The previous four-document MSE16 standard remains available at
`59f422b2d410fdaf4a9efc71ff12f278abc2a5d1`; the legacy artifact remains at
`cc7df08da7ef7ac15db62e80b4eda85e19a143da`. Pin a revision rather than mutable
`main`. Prefix state must not be shared across different model identities.

Derived from Qwen/Qwen3.5-0.8B under Apache-2.0. The original license is included
as `LICENSE`, with modification attribution in `NOTICE`. The CPU engine is
separately MIT-licensed.
