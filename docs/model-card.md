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

Measured on the **AMD Ryzen 9 9955HX3D** laptop with eight threads pinned to
physical cores 0–7. Rates are median **tokens/s** from three runs after one
warmup. Prefill measures one prompt. Decode uses 512 input and 128 output
tokens per request. B=16 is total throughput across 16 private requests.

| Model | Engine / checkpoint | Tensor MB | Decode B=1 | Decode B=16 | Prefill 512 | Prefill 4,096 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | ik_llama.cpp IQ4_KS_R4 | 401.4 | 121.5 | 400.7 | 3,630.8 | 3,144.8 |
| 0.8B | ik_llama.cpp IQ4_K_R4 | 424.9 | 113.4 | 388.9 | 3,629.5 | 3,141.2 |
| 0.8B | **Native H128/Q4-G32-DOT4** | 424.9 | 111.0 | 575.6 | 3,052.4 | 2,670.7 |
| 0.8B | llama.cpp Q4_0 | 424.9 | 101.8 | 456.5 | 1,099.5 | 1,072.2 |
| 0.8B | llama.cpp Unsloth Q4_K_M | 521.6 | 83.3 | 389.8 | 1,106.7 | 959.4 |
| 2B | ik_llama.cpp IQ4_K_R4 | 1,060.4 | 43.6 | 245.7 | 1,737.7 | 1,636.2 |
| 2B | llama.cpp Q4_0 | 1,060.4 | 42.3 | 306.4 | 701.4 | 626.8 |
| 2B | **Native H128/Q4-G32-DOT4** | 1,060.4 | 42.2 | 382.6 | 1,504.9 | 1,414.4 |
| 4B | ik_llama.cpp IQ4_K_R4 | 2,369.0 | 18.9 | 106.5 | 707.8 | 635.6 |
| 4B | **Native H128/Q4-G32-DOT4** | 2,369.0 | 18.4 | 163.4 | 614.4 | 557.5 |
| 4B | llama.cpp Q4_0 | 2,369.0 | 18.2 | 98.3 | 195.9 | 175.7 |

At 2B and 4B, IK led single-request decode and prefill. Native delivered the
highest decode throughput across 16 requests. All three formats have the same
on-disk tensor size within each of these model sizes. Timings use fixed tokens,
FP16 KV and full-vocabulary logits and exclude loading, tokenization and HTTP.
[0.8B runs, November 19](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/ryzen-9955hx3d.md) · [2B and 4B runs, November 20](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/ryzen-9955hx3d-large.md)

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

The November native build reproduced every vocabulary logit from the October
quality run. The comparator scores come from that run with the same binaries
and weights. [Native verification](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/results/readme-2026-11-19/native-quality.json) · [Quality measurements](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/cpu.md#quality-and-size)

## License

Derived from Qwen/Qwen3.5-0.8B under Apache-2.0. The model license is included
as [LICENSE](LICENSE), with modification attribution in [NOTICE](NOTICE).
The CPU engine is separately MIT-licensed.
