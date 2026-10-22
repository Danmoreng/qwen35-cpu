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

The standard **H128/Q4-G32-DOT4** checkpoint has a **424.9 MB tensor payload**.
It uses activation-weighted MSE16 fitting and error compensation within
128-channel blocks, calibrated on 256 mixed documents / 262,144 tokens.
The checkpoint and comparator GGUF files are unchanged from the earlier comparison.

Measured **October 22, 2026**, using these exact engine revisions:

- Native engine: [`f928eec37ab3`](https://github.com/Danmoreng/qwen35-cpu/commit/f928eec37ab34739e03d4fa3d37fe34d44ff7229).
- llama.cpp **b11815**: [`7170cad4d02f`](https://github.com/ggml-org/llama.cpp/commit/7170cad4d02f7907517613931840cbb2367399a3).
- ik_llama.cpp: [`98aca8692ae3`](https://github.com/ikawrakow/ik_llama.cpp/commit/98aca8692ae334e6e23e4d39ac43ce0d21029c35).

### Speed, size and quality

**AMD Ryzen 9 9955HX3D · Arch Linux / kernel 7.2.9 · eight physical V-Cache cores
(0–7, affinity `0xff`) · GCC 16.2.1 Release · AVX-512/VNNI.** The laptop platform
profile is `max-power`; the separate Linux CPU settings are `amd-pstate-epp`
active mode, governor `powersave`, EPP `balance_performance`. These are the
recorded settings, not a claim of fixed CPU clocks or a low-power laptop profile.
Codex is minimized, XFCE compositing disabled, and the display runs at
2560×1600 / 240 Hz.

Median **tokens/s** from three measured runs after one warmup, executed
sequentially with alternating case order. All candidates use identical fixed
tokens, FP16 KV and full-vocabulary logits. IK uses runtime tensor repacking.
The native row uses the later [paired prefill measurement](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/native-prefill.md)
on the same day; comparator rows retain the full-engine comparison. The paired
native baseline confirms **+4.59% P512 / +3.62% P4096 prefill**, with unchanged
B1/B16 decode. Absolute differences between the two series can include clock
and thermal variation; small cross-engine gaps are not a definitive ranking.

Sorted by **single-request decode speed**, fastest first. Bold speeds mark the
best result in each column. MB is tensor payload (decimal), not RAM usage.
**↑ higher is better · ↓ lower is better.**

| Engine / checkpoint | Decode B=1 ↑ | Decode B=16 ↑ | Prefill 512 ↑ | Prefill 4,096 ↑ | MB | PPL ↓ | KL ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ik_llama.cpp IQ4_KS_R4 | **129.59** | 433.68 | **3,689.22** | 3,177.90 | 401.4 | 15.86784 | 0.084944 |
| **This engine, H128/Q4-G32-DOT4** | 123.03 | **639.63** | 3,052.57 | 2,683.16 | 424.9 | 15.79649 | 0.060100 |
| ik_llama.cpp Q4_0 | 122.34 | 419.26 | 3,142.75 | 2,756.00 | 424.9 | 18.06313 | 0.145489 |
| ik_llama.cpp IQ4_K_R4 | 122.04 | 415.26 | 3,633.49 | 3,159.75 | 424.9 | 15.77716 | 0.068812 |
| llama.cpp Q4_0 | 109.20 | 494.79 | 1,124.87 | 977.58 | 424.9 | 18.03403 | 0.144830 |
| ik_llama.cpp Unsloth IQ4_XS | 107.64 | 400.87 | 3,631.78 | **3,179.94** | 481.6 | 15.21385 | 0.050634 |
| ik_llama.cpp Unsloth Q4_0 | 104.33 | 413.16 | 3,126.61 | 2,751.60 | 496.2 | 15.58460 | 0.068085 |
| ik_llama.cpp Unsloth Q4_K_M | 99.03 | 387.90 | 3,403.18 | 3,047.68 | 521.6 | 14.77376 | 0.034637 |
| llama.cpp Unsloth IQ4_XS | 95.02 | 418.35 | 1,265.10 | 1,130.56 | 481.6 | 15.14159 | 0.050441 |
| llama.cpp Unsloth Q4_0 | 94.54 | 442.39 | 1,006.55 | 1,052.35 | 496.2 | 15.58211 | 0.068394 |
| llama.cpp Unsloth Q4_K_M | 88.66 | 419.22 | 1,146.78 | 1,019.74 | 521.6 | 14.77209 | 0.034894 |

**IK B16 uses the documented graph-capacity patch on the revision above.**
Stock pure-Q4_0 B16 still aborts; IK B1, prefill and quality use unmodified upstream.
See [IK batch validation](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/ik-batch16.md).

Prefill uses one request. Decode uses 512 input / 128 output tokens and counts
127 actual decode forwards per request. B16 is aggregate throughput across
sixteen private requests. Loading, tokenization, HTTP and prefix-cache credit
are excluded. Native prefill reports model-forward time; comparator prefill
reports synchronized `llama_decode` time, both including final full-vocabulary
logits. Quality scores are measured separately at B1 with the named engine.

### Perplexity and KL divergence

**8,192 scored tokens from 16 WikiText-2 test article windows**, with identical
prompts and scoring masks and the original frozen BF16 teacher logits. Comparator
rows were rescored with the revisions above. The optimized native revision
produces bit-identical logits to the native scoring run at all 8,192 positions,
so its PPL/KL are unchanged. The BF16 reference anchors both measurements.
This is an English-prose subset, not full-corpus WikiText perplexity.
**Lower PPL and KL are better.** KL is in nats. Sorted by tensor size, largest
first, then lower PPL and lower KL. Each row names its actual scoring backend;
IK scores are measured directly, including the three Unsloth checkpoints.

| Engine / checkpoint | Tensor MB ↓ | PPL ↓ | KL to BF16 ↓ |
| --- | ---: | ---: | ---: |
| BF16 teacher (frozen reference) | 1,505.8 | 14.38556 | 0 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.77209 | 0.034894 |
| ik_llama.cpp Unsloth Q4_K_M | 521.6 | 14.77376 | 0.034637 |
| llama.cpp Unsloth Q4_0 | 496.2 | 15.58211 | 0.068394 |
| ik_llama.cpp Unsloth Q4_0 | 496.2 | 15.58460 | 0.068085 |
| llama.cpp Unsloth IQ4_XS | 481.6 | 15.14159 | 0.050441 |
| ik_llama.cpp Unsloth IQ4_XS | 481.6 | 15.21385 | 0.050634 |
| ik_llama.cpp IQ4_K_R4 | 424.9 | 15.77716 | 0.068812 |
| **This engine, H128/Q4-G32-DOT4** | 424.9 | 15.79649 | 0.060100 |
| llama.cpp Q4_0 | 424.9 | 18.03403 | 0.144830 |
| ik_llama.cpp Q4_0 | 424.9 | 18.06313 | 0.145489 |
| ik_llama.cpp IQ4_KS_R4 | 401.4 | 15.86784 | 0.084944 |

H128 and pure Q4_0 / IQ4_K_R4 share a **424,934,656-byte tensor budget**.
Storage is not total resident memory: runtime repacking, KV and compute buffers
require additional memory. Quantization labels alone do not establish equal quality.

The independent mixed-language, code and math calibration validation remains a
separate corpus: 12.71898 PPL and 0.064178 KL over 1,280 tokens. It was not rerun
as part of this English benchmark refresh.

[Measurement details and reproduction](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/cpu.md) ·
[Comparator series CSV](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/cpu-refresh-2026-10-22/performance.csv) ·
[Native paired series CSV](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/native-prefill-reuse-2026-10-22/performance.csv) ·
[Quality CSV](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/cpu-refresh-2026-10-22/quality.csv) ·
[Revisions and provenance](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/cpu-refresh-2026-10-22/manifest.json) ·
[Quantization recipe](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/quantization.md).

The earlier [Intel i7-8750H comparison](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/cpu.md#earlier-second-cpu-comparison)
uses older engine revisions and was not rerun on this machine.

## License

Derived from Qwen/Qwen3.5-0.8B under Apache-2.0. The model license is included
as [LICENSE](LICENSE), with modification attribution in [NOTICE](NOTICE).
The CPU engine is separately MIT-licensed.
