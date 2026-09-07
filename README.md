# Qwen3.5 CPU

A small C++20 inference engine specialized for **Qwen3.5-0.8B text inference on
CPUs**, using **B+C calibrated H128/Q4-G32-DOT4** weights. No CUDA toolchain, GPU runtime or
general-purpose model framework is required.

The project contains a native HTTP server, text-completion CLI, checkpoint packer,
embeddable multi-request engine, CPU tests and comparison tools. It is an independent
extraction of the validated CPU implementation in qwen35x; see [provenance](PROVENANCE.md).
The HTTP server supports a **limited OpenAI-style `/v1/completions` API**:
raw prompts, greedy or sampled decoding and non-streaming responses. Chat completions and
SSE streaming are not implemented.

## CPU benchmarks: speed and quality

The engine uses the **H128/Q4-G32-DOT4 B+C 256** checkpoint with a **424.9 MB
tensor payload**. Speed and quantization quality are measured separately.

### Performance

**AMD Ryzen 9 9955HX3D · Arch Linux · eight physical V-Cache cores · GCC 16.2.1
Release · AVX-512/VNNI.** Values are median **tokens/s** from three measured
runs after one warmup. All candidates use identical fixed tokens, FP16 KV and
full-vocabulary logits. Codex is minimized, XFCE compositing disabled, and the
display runs at 2560×1600 / 240 Hz. IK uses runtime tensor repacking.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| **This engine, H128 B+C 256** | 2,758.49 | 2,464.28 | 122.94 | 642.08 |
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
are excluded. IK B16 uses a [local graph-capacity correction](docs/ik-batch16.md);
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
| **This engine, H128 B+C 256** | **424.9** | **15.80467** | **0.060190** |
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

[Measurement details and reproduction](docs/current-cpu-comparison.md) ·
[Performance CSV](docs/results/current-cpu-2026-09-07/performance.csv) ·
[B16 CSV](docs/results/ik-batch16-2026-09-07/performance.csv) ·
[Provenance](docs/results/current-cpu-2026-09-07/manifest.json) ·
[Quantization recipe](docs/quantization.md) ·
[Independent quality validation](docs/g32-large-calibration-2026-09-07.md).

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
publishes binaries; Intel i7-8750H hardware validation remains pending. ISA dispatch is designed for x86 portability, but
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

- CPU-only build and sixteen kernel/unit/evaluation tests: passed locally.
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

See [validation details](docs/validation.md) and the [bounded release plan](docs/comparison-plan.md).

Calibrated MSE16 and block-128 error compensation run offline and preserves the DOT4 runtime and payload
size. The standard download passes the free `2+2 = 4` regression. See the
[calibration and benchmark report](docs/g32-large-calibration-2026-09-07.md)
for quality scores, sequential speed measurements, reproduction and remaining
plan stages.
