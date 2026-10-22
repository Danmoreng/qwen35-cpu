# Qwen3.5 CPU

A small C++20 inference engine specialized for **Qwen3.5-0.8B text inference on
CPUs**, using **calibrated H128/Q4-G32-DOT4** weights. No CUDA toolchain, GPU runtime or
general-purpose model framework is required.

The project contains a native HTTP server, text-completion CLI, checkpoint packer,
embeddable multi-request engine, CPU tests and comparison tools. It is an independent
extraction of the validated CPU implementation in qwen35x; see [provenance](PROVENANCE.md).
The HTTP server supports **OpenAI-compatible text Chat Completions** with native
Qwen prompt formatting, plus raw `/v1/completions`. Both support greedy or sampled
decoding and complete JSON responses. SSE streaming and tool calls are not yet
implemented; see the [supported API fields](docs/server.md#chat-completions).

The native server also supports **qev-0.8b decisions via `POST /v1/systemone`**
with a separate calibrated [qev model](https://huggingface.co/danmoreng/qev-0.8b-H128-Q4-G32-DOT4).
See the [decision API](docs/server.md#qev-decision-api),
[decision quality study](docs/benchmarks/decision-quality.md) and
[current decision HTTP/concurrency measurements](docs/benchmarks/decision-http.md).
The current native qev server projects only the needed option logits. Fresh
HTTP B1 medians are **44.27 / 202.76 ms** for 74 / 579 input tokens with the
adaptive chunk default, or **40.05 / 203.03 ms** with `--prefill-chunk 128`
on the Ryzen CPU described below. These are decision latencies, not decode
tokens/s; weights and calibration are unchanged.
The Qwen chat/base-model benchmarks below remain separate from that fine-tune.

Try the [live CPU decision playground](https://huggingface.co/spaces/danmoreng/qev-0.8b-cpu-playground):
seven interactive demos use this custom engine compiled to WASM. The initial
model/config/tokenizer download is about 448 MB; inference runs on your browser's
CPU. See [Space packaging and validation](docs/browser/hosting.md).

[Documentation](docs/README.md) · [Validation](docs/validation.md) · [Tooling](scripts/README.md)

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
The native row uses the later [paired prefill measurement](docs/benchmarks/native-prefill.md)
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
See [IK batch validation](docs/benchmarks/ik-batch16.md).

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

[Measurement details and reproduction](docs/benchmarks/cpu.md) ·
[Comparator series CSV](docs/results/cpu-refresh-2026-10-22/performance.csv) ·
[Native paired series CSV](docs/results/native-prefill-reuse-2026-10-22/performance.csv) ·
[Quality CSV](docs/results/cpu-refresh-2026-10-22/quality.csv) ·
[Revisions and provenance](docs/results/cpu-refresh-2026-10-22/manifest.json) ·
[Quantization recipe](docs/quantization.md).

The earlier [Intel i7-8750H comparison](docs/benchmarks/cpu.md#earlier-second-cpu-comparison)
uses older engine revisions and was not rerun on this machine.

## Download and run

The release workflow builds Windows x64 ZIP and Linux x64 tar.gz archives with
the server, CLI, model download scripts and licenses. Runtime needs no compiler,
Python or CUDA. Windows uses a static MSVC runtime; Linux releases target
Ubuntu 22.04 or newer (glibc 2.35+), not every Linux distribution.

Model: [danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4](https://huggingface.co/danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4).
Download binaries from [GitHub Releases](https://github.com/Danmoreng/qwen35-cpu/releases).
The commands below pin the validated model revision, so later uploads cannot
silently change the weights or tokenizer.

After extracting a release archive, run from its directory:

```powershell
# Windows PowerShell 5.1+; no Python required
./download-model.ps1 -Repo danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 -Revision f6cb5cf04a9094670f9041578a3f4e44b94a1395
./qwen35_cpu_server.exe --model-dir models/qwen3.5-0.8b --threads 8
```

```sh
# Linux: bash, curl and sha256sum
bash ./download-model.sh danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 f6cb5cf04a9094670f9041578a3f4e44b94a1395
./qwen35_cpu_server --model-dir models/qwen3.5-0.8b --threads 8
```

The downloader verifies every file against the pinned revision's SHA-256
manifest. Download into a new directory. The checkpoint is already CPU-packed.
The server defaults to localhost:8080, 16 resident requests, bounded queueing,
FP16 shared KV pages and an 8192-token context limit.

```sh
curl http://127.0.0.1:8080/v1/completions -H "Content-Type: application/json" -d '{"prompt":"Once upon a time","max_tokens":64,"temperature":0}'
```

Use `curl.exe` on Windows, or `Invoke-RestMethod` with the same JSON body.
For sampling (v0.1.1+), send e.g. `"temperature":0.7, "top_p":0.8,
"top_k":20, "repetition_penalty":1.05, "seed":42`. Omitting temperature keeps
greedy decoding for compatibility. Each request has its own random state.
See [server API and limits](docs/server.md) and [publishing](docs/publishing.md).

## What is specialized

- One architecture: Qwen3.5-0.8B, including its hybrid DeltaNet/full-attention layers.
- Main weight paths: H128 transforms with Q4 groups of 32, or plain Q4_0, both using CPU DOT4 packing.
  Large projection weights use H128/Q4; embeddings and small retained tensors use
  their prescribed encodings. This is not an all-tensors-four-bit claim.
- H128 packing and sign conversion happen during checkpoint conversion. Plain
  Q4_0 GGUF weights are losslessly repacked once at model load. Inference reads
  the prepared blocks; loading also allocates metadata, scratch and projection groups.
- Experimental Q4_K_M GGUF execution is retained but slower at prefill; further
  K-quant optimization is paused. It does not use the optimized DOT4 path.
- FP16 KV caches, FP32 recurrent state, CPU-dispatched scalar/AVX2/VNNI/AVX-512 kernels.
- Batched decode, bounded scheduling, cancellation, output backpressure, exact-prefix
  reuse including recurrent state, and optional shared immutable KV pages.

The public C++ namespace remains `qwen35x` to keep the extraction small and traceable.
See the [engine API](docs/cpu-engine-api.md).

## Build

Windows: Visual Studio 2022 C++ Build Tools, CMake 3.24+, Ninja and PowerShell 7.
The server build downloads hash-pinned cpp-httplib and nlohmann/json headers;
neither is required separately at runtime. Set `-DQWEN35_BUILD_SERVER=OFF` for
an engine/CLI-only build with no downloaded dependencies.

```powershell
./scripts/build.ps1
ctest --test-dir build --output-on-failure
```

Linux: a C++20 compiler, CMake and Ninja/pthreads.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Windows/MSVC and Linux/GCC builds and kernel tests pass in GitHub CI.
Real-model release tests are required on both platforms before a release tag
publishes binaries. The [earlier Intel i7-8750H measurements](docs/benchmarks/cpu.md#earlier-second-cpu-comparison)
cover that AVX2 system. ISA dispatch is designed for x86 portability, but
this is not a claim of testing every x86 CPU. AVX2 or newer is the practical
performance target; keep the portable core free of global `-march=native` flags.

## Model and packing

Place the original Qwen3.5-0.8B Hugging Face checkpoint and tokenizer files in
`models/qwen3.5-0.8b`. Models and build products are ignored by Git. The model
has its own license; the engine's MIT license does not replace it.

```powershell
./build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model.q35h --quantizer mse16 --importance-dir benchmarks/bc-large-study/full-importance --covariance-dir benchmarks/bc-large-study/full-covariance
./build/qwen35_cpu.exe --model-dir models/qwen3.5-0.8b --weights models/qwen3.5-0.8b/model.q35h --prompt "Once upon a time" --threads 8 --max-new-tokens 128
```

The command above reproduces the calibrated recipe after collecting and fitting
teacher inputs into the full importance and covariance directories above. The packer uses `cpu-dot4`
layout. Downloading the prepared standard model requires no calibration or
conversion at runtime.
See [the standard recipe and conversion workflow](docs/quantization.md). Runtime
loading requires the matching `.q35h`, `config.json` and tokenizer files, not
the original BF16 shards. The validated local artifact is about 425 MB
(decimal); it is not shipped in this repository.

The CLI performs **raw text completion**, defaulting to greedy decoding. Use
`--temperature 0.7 --top-p 0.8 --top-k 20 --repetition-penalty 1.05 --seed 42`
for sampling (v0.1.1+). It does not
automatically render chat templates. For chat, pass an already rendered prompt
using `--prompt-file`. Tokenization and teacher-forced logit export are also
available through `--help`. On Linux, omit `.exe` in executable names.

## Threads, batching and prefixes

Pass `--threads` explicitly for the CLI; use `CpuEngineConfig::threads` for the
library. Eight threads is a starting point, not universal autotuning. Measure
thread counts separately for single-request latency and batched throughput.
On the development Ryzen, larger batches benefited from sixteen physical-core
threads; using all SMT threads did not automatically improve performance.

The engine owns one executor per instance. Drive `step()` from one owner,
drain output and release completed request records. Prefixes must match exact
token IDs, model identity and trust namespace. Enable `shared_kv_pages` to
share immutable FP16 prefix pages; hybrid recurrent state remains request-private.

## Reproducibility and release status

Run all timed experiments through `scripts/benchmark-inference-seq.ps1`. It
serializes processes, records commands and binary hashes, alternates case order,
and writes profiles plus a completed CSV. Three measured runs and one warmup
are the default. Quality/logit-dump runs are separate from speed measurements.

- Native kernel, unit and evaluation checks: see the [validation guide](docs/validation.md).
- Real-model scheduler/prefix/shared-page regression: passed.
- AVX-512 prefill: bitwise full-model logit equivalence over 24 prompts, including
  chunk boundaries and a 4,096-token prompt.
- Native text CLI and full-vocabulary logit export: smoke-tested.
- Native HTTP adapter, concurrent request/prefix correctness and packaged Windows
  server with the prepared Hugging Face model: passed locally.
- GitHub CI covers Windows/Linux builds, kernel tests and packaging. The public
  Hugging Face model is pinned for real-model tests and tag-triggered releases.
- Mixed 256-document calibration, error-compensated fitting and independent quality validation: passed.
- Broader context-length and corpus validation remains ongoing.

See [validation details](docs/validation.md) and the [publication workflow](docs/publishing.md).

Calibrated MSE16 and block-128 error compensation run offline and preserve the
DOT4 runtime and payload size. The standard download passes the free `2+2 = 4`
regression. See the [calibration report](docs/benchmarks/calibration.md) for
quality scores and reproduction, and the [documentation index](docs/README.md)
for current guides.
