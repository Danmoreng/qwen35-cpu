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
`http://localhost:8080/v1/completions` and native Qwen chat formatting at
`/v1/chat/completions`; both offer JSON responses or SSE streaming. Text Messages
at `/v1/messages` also support the Anthropic SDK, streaming and token counting.
Both chat APIs provide constrained JSON output and one structured tool call per
turn; the engine guide documents the supported schema subset and limits.
Follow the
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

<a id="qwen35-08b-performance"></a>

## CPU performance

Measured November 20, 2026 on the **AMD Ryzen 9 9955HX3D** laptop with eight threads pinned to
physical cores 0–7. Rates are median **tokens/s** from three runs after one
warmup. Prefill measures one prompt. Decode uses 512 input and 128 output
tokens per request. B=16 is total throughput across 16 private requests.

| Model | Engine / checkpoint | Tensor MB | Decode B=1 | Decode B=16 | Prefill 512 | Prefill 4,096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | ik_llama.cpp IQ4_KS_R4 | 401.4 | 134.2 | 440.1 | 3,715.4 | 3,214.2 |
| 0.8B | ik_llama.cpp IQ4_K_R4 | 424.9 | 126.2 | 425.1 | 3,703.7 | 3,195.1 |
| 0.8B | **Native H128/Q4-G32-DOT4** | 424.9 | 121.5 | 647.2 | 3,348.4 | 2,912.2 |
| 0.8B | llama.cpp Q4_0 | 424.9 | 112.5 | 493.1 | 1,119.6 | 1,089.0 |
| 0.8B | llama.cpp Unsloth Q4_K_M | 521.6 | 90.8 | 429.3 | 1,025.0 | 1,103.0 |
| 2B | ik_llama.cpp IQ4_K_R4 | 1,060.4 | 48.9 | 265.9 | 1,771.2 | 1,656.1 |
| 2B | **Native H128/Q4-G32-DOT4** | 1,060.4 | 47.5 | 435.5 | 1,639.0 | 1,526.9 |
| 2B | llama.cpp Q4_0 | 1,060.4 | 47.0 | 345.9 | 644.9 | 633.1 |
| 4B | ik_llama.cpp IQ4_K_R4 | 2,369.0 | 21.2 | 115.9 | 717.1 | 645.8 |
| 4B | **Native H128/Q4-G32-DOT4** | 2,369.0 | 20.8 | 186.2 | 674.4 | 608.2 |
| 4B | llama.cpp Q4_0 | 2,369.0 | 20.4 | 112.9 | 182.5 | 178.1 |

Native had the highest batch-16 decode throughput at all three model sizes. The three charted formats have the same tensor size within each model size. [Throughput measurements](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/laptop-refresh-9955hx3d.md#base-model-throughput)

<a id="quantization-quality"></a>

### 0.8B quantization quality

Perplexity (PPL) measures how well the model predicts the reference text.
KL measures how closely its predictions match the BF16 teacher, in nats.
Lower is better for both. Scores cover 8,192 tokens from 16 English WikiText-2
test windows. MB measures the on-disk tensor payload.

| Scoring engine / checkpoint | Tensor MB | PPL ↓ | KL to BF16 ↓ |
| --- | ---: | ---: | ---: |
| llama.cpp BF16 reference | 1,505.8 | 14.38556 | 0 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.77209 | 0.034894 |
| ik_llama.cpp IQ4_K_R4 | 424.9 | 15.77716 | 0.068812 |
| **Native H128/Q4-G32-DOT4** | 424.9 | 15.79649 | 0.060100 |
| llama.cpp Q4_0 | 424.9 | 18.03403 | 0.144830 |
| ik_llama.cpp IQ4_KS_R4 | 401.4 | 15.86784 | 0.084944 |

The November 19 native build reproduced every vocabulary logit from the October
quality run. The comparator scores come from that run with the same binaries
and weights. [Native verification](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/readme-2026-11-19/native-quality.json) · [Quality measurements](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/cpu.md#quality-and-size)

## License

Derived from Qwen/Qwen3.5-0.8B under Apache-2.0. The model license is included
as [LICENSE](LICENSE), with modification attribution in [NOTICE](NOTICE).
The CPU engine is separately MIT-licensed.
