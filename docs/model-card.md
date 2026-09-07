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
[Qwen3.5 CPU engine](@ENGINE_URL@), stored in the custom **H128/Q4-G32-DOT4**
`.q35h` format. Weights are fitted with activation-weighted MSE16 and error
compensation within 128-channel blocks, using 256 calibration documents.

## Download and run

Download the matching engine from [GitHub Releases](@ENGINE_URL@/releases/latest)
and this repository's files into one model directory. No BF16 weights,
calibration, Python or CUDA are needed at runtime. This is **not GGUF or a
Transformers-loadable checkpoint**.

```sh
./qwen35_cpu_server --model-dir /path/to/downloaded-model --threads 8
```

Use `qwen35_cpu_server.exe` on Windows. The server supports raw completions at
`http://localhost:8080/v1/completions`; render chat prompts before sending them.
Streaming and a chat completions endpoint are not implemented. Follow the
[engine README](@ENGINE_URL@/blob/@ENGINE_REVISION@/README.md) for pinned revision
downloads and checksum verification. Pin an immutable model revision for
reproducibility; prefix state must not be shared across different checkpoints.

## Format and calibration

Large projections use signed Hadamard transforms over 128 channels and groups
of 32 signed four-bit codes with FP16 scales, prepacked for CPU DOT4. The tied
embedding/output matrix uses its identity-basis Q4 encoding; prescribed small
tensors use FP32. Runtime uses FP16 KV and FP32 recurrent state. The artifact
contains text inference weights, without vision or MTP/draft weights.

Calibration uses 256 documents / 262,144 tokens: approximately 25% German,
25% English, 20% Python code, 20% dialog and 10% mathematics. Sources are
FineWeb2, FineWeb-Edu, CodeParrot Clean, SmolTalk and OpenWebMath, with pinned
revisions and excluded evaluation documents. A capacity-256 reservoir samples
projection inputs over each 1,024-token document. BF16 teacher activations supply
the fitting statistics; teacher execution uses CUDA only during offline calibration.

MSE16 searches the sixteen signed four-bit levels and stored FP16 scales using
activation-weighted reconstruction error. Error compensation uses second moments
within 128-channel blocks with damping 0.01 and retains the MSE16 candidate when
compensation does not improve the undamped reconstruction objective. This is
not full-matrix GPTQ or sequential layer recalibration.

Tensor payload: **424,934,656 bytes**; complete artifact: **424,964,864 bytes**.
Model SHA256: `@SHA256@`. See [quantization.json](quantization.json) for source
hashes and calibration provenance. [SHA256SUMS](SHA256SUMS) covers the package.

## CPU benchmarks: speed and quality

The engine uses the **H128/Q4-G32-DOT4** checkpoint with a **424.9 MB
tensor payload**. It is generated using activation-weighted MSE16 fitting and
error compensation within 128-channel blocks, calibrated on 256 mixed documents
/ 262,144 tokens. See the [quantization recipe](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/quantization.md).
Speed and quantization quality are measured separately.

### Performance

**AMD Ryzen 9 9955HX3D · Arch Linux · eight physical V-Cache cores · GCC 16.2.1
Release · AVX-512/VNNI.** Values are median **tokens/s** from three measured
runs after one warmup. All candidates use identical fixed tokens, FP16 KV and
full-vocabulary logits. Codex is minimized, XFCE compositing disabled, and the
display runs at 2560×1600 / 240 Hz. IK uses runtime tensor repacking.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| **This engine, H128/Q4-G32-DOT4** | 2,758.49 | 2,464.28 | 122.94 | 642.08 |
| llama.cpp Q4_0 | 1,105.29 | 994.72 | 108.64 | 488.26 |
| llama.cpp Unsloth Q4_0 | 953.59 | 843.96 | 93.73 | 372.75 |
| llama.cpp Unsloth Q4_K_M | 663.21 | 695.11 | 88.41 | 325.93 |
| llama.cpp Unsloth IQ4_XS | 940.31 | 887.65 | 94.12 | 336.28 |
| ik_llama.cpp Q4_0 | 2,860.45 | 2,517.56 | 122.22 | 423.79 |
| ik_llama.cpp Unsloth Q4_0 | 2,730.89 | 2,409.68 | 104.00 | 420.00 |
| ik_llama.cpp Unsloth Q4_K_M | 1,984.34 | 1,794.75 | 98.63 | 395.87 |
| ik_llama.cpp Unsloth IQ4_XS | 1,957.50 | 1,774.74 | 107.68 | 408.20 |
| ik_llama.cpp IQ4_K_R4 | 1,992.59 | 1,802.06 | 117.83 | 433.81 |
| ik_llama.cpp IQ4_KS_R4 | 2,060.38 | 1,841.85 | 129.21 | 455.14 |

Prefill uses one request. Decode uses 512 input / 128 output tokens and counts
127 actual decode forwards per request. B16 is aggregate throughput across
sixteen private requests. Loading, tokenization, HTTP and prefix-cache credit
are excluded. IK B16 uses a [local graph-capacity correction](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/ik-batch16.md);
IK B1, prefill and quality use the stock fork. These measurements describe this
CPU and workload.

### Perplexity and KL divergence

**8,192 scored tokens from 16 WikiText-2 test article windows**, with identical
prompts and scoring masks and a common BF16 teacher. This is an English-prose
subset, not full-corpus WikiText perplexity. **Lower PPL and KL are better.**
Each row identifies the backend used for scoring.

| Engine / checkpoint | Tensor payload, MB | Perplexity | Mean KL to BF16, nats |
| --- | ---: | ---: | ---: |
| BF16 teacher | 1,505.8 | 14.38556 | 0 |
| **This engine, H128/Q4-G32-DOT4** | **424.9** | **15.80467** | **0.060190** |
| llama.cpp Q4_0 | 424.9 | 18.02588 | 0.144522 |
| llama.cpp Unsloth Q4_0 | 496.2 | 15.58088 | 0.068388 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.75290 | 0.034693 |
| llama.cpp Unsloth IQ4_XS | 481.6 | 15.16145 | 0.050539 |
| ik_llama.cpp Q4_0 | 424.9 | 18.08595 | 0.145799 |
| ik_llama.cpp IQ4_K_R4 | 424.9 | 15.78482 | 0.073289 |
| ik_llama.cpp IQ4_KS_R4 | 401.4 | 15.90097 | 0.090599 |

H128 and pure Q4_0 / IQ4_K_R4 share a **424,934,656-byte tensor budget**.
The smaller IQ4_KS_R4 trades some quality for size and decode speed. Tensor
payload is not RAM usage: IK's loaded model tensors include additional storage;
KV and compute buffers also require memory. Quantization labels alone do not
establish equal quality.

The H128 checkpoint is calibrated on 256 mixed documents / 262,144 tokens.
An independent mixed-language, code and math test scores 12.71898 PPL and
0.064178 KL over 1,280 tokens; it is a separate corpus from the table above.

[Measurement details and reproduction](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/current-cpu-comparison.md) ·
[Performance CSV](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/current-cpu-2026-09-07/performance.csv) ·
[B16 CSV](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/ik-batch16-2026-09-07/performance.csv) ·
[Provenance](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/current-cpu-2026-09-07/manifest.json) ·
[Quantization recipe](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/quantization.md) ·
[Independent quality validation](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/g32-large-calibration-2026-09-07.md).

## License

Derived from Qwen/Qwen3.5-0.8B under Apache-2.0. The model license is included
as [LICENSE](LICENSE), with modification attribution in [NOTICE](NOTICE).
The CPU engine is separately MIT-licensed.
