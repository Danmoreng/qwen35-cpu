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

The published **H128/Q4-G32-DOT4 B+C 256** checkpoint uses weighted MSE16 plus
block-128 error compensation, calibrated on 256 documents / 262,144 tokens.
It achieves **15.80467 perplexity** and **0.060190 mean KL to BF16** on the common
English test subset, with a **424.9 MB tensor payload**. Quality and speed are
measured separately; the unchanged custom format needs no GPU at runtime.

### Speed

Speed includes two complete platform-specific comparison series and a separate
native-engine Linux/Ryzen decode check. All use FP16 KV,
identical fixed tokens, full-vocabulary logits and **three measured runs after
one warmup**, with alternating case order. Values are median **tokens/s**.

#### Windows — Ryzen 9 9955HX3D

Eight threads on physical VCache cores (`0x5555`), MSVC Release, AVX-512/VNNI.
All five candidates were measured together in a fresh sequential series.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| **This engine, H128 B+C 256** | 2,147.22 | 1,877.68 | 120.05 | 619.16 |
| llama.cpp Q4_0 `--pure` | 1,073.72 | 970.00 | 101.15 | 326.13 |
| llama.cpp Unsloth Q4_0 | 892.21 | 860.68 | 88.16 | 290.07 |
| llama.cpp Unsloth Q4_K_M | 673.80 | 661.63 | 84.29 | 259.64 |
| llama.cpp Unsloth IQ4_XS | 853.14 | 827.82 | 88.76 | 259.74 |

Against equal-payload llama.cpp pure Q4_0, this checkpoint delivers **1.94–2.00×
prefill throughput**, **1.19× single-request decode** and **1.90× batch-16 decode**.

#### Linux — Ryzen 9 9955HX3D (native-engine decode check)

Arch Linux, kernel 7.2.2, GCC 16.2.1 Release, AVX-512/VNNI, eight threads on
physical VCache cores (`0xff`). The unchanged native engine was measured with
the Codex window minimized and XFCE compositing disabled; the internal display
remained at 2560×1600 / 240 Hz.

| Engine / checkpoint | Decode B=1 | Decode B=16 |
| --- | ---: | ---: |
| **This engine, H128 B+C 256** | **127.48** | **638.53** |

This is a native-engine P512/N128 check, not a new five-candidate comparison.
The three measured runs span 127.36–127.59 tok/s for B1 and 637.43–639.15 tok/s
for aggregate B16. Active desktop rendering substantially reduced throughput
in the initial Linux investigation; minimizing Codex alone, with compositing
still enabled at 240 Hz, also recovered B1 performance (126.42 tok/s).
Keep desktop conditions fixed when comparing builds. See the
[Linux/Ryzen measurement details](docs/linux-desktop-performance-root-cause-2026-09-07.md)
and [results](docs/results/linux-desktop-2026-09-07/summary.csv).

#### Linux — Intel Core i7-8750H

Ubuntu 26.04.1 LTS, GCC 15.2 Release, six threads pinned to the six physical
cores (`0x3f`), AVX2/FMA/F16C. All five candidates were rebuilt and measured
together in a separate fresh sequential series.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| **This engine, H128 B+C 256** | 404.17 | 319.01 | 62.28 | 179.36 |
| llama.cpp Q4_0 `--pure` | 230.63 | 185.92 | 47.42 | 91.06 |
| llama.cpp Unsloth Q4_0 | 213.16 | 172.86 | 42.34 | 79.66 |
| llama.cpp Unsloth Q4_K_M | 184.74 | 152.73 | 39.89 | 74.73 |
| llama.cpp Unsloth IQ4_XS | 230.81 | 184.95 | 40.82 | 76.78 |

Against equal-payload llama.cpp pure Q4_0 on this Linux system, the custom
checkpoint delivers **1.72–1.75× prefill throughput**, **1.31× single-request
decode** and **1.97× batch-16 decode**.

Prefill columns use one request. Decode uses 512 input / 128 output tokens and
counts 127 actual decode forwards per request. Batch 16 is aggregate throughput
across 16 private requests, compared only with batch 16. Load, tokenization,
HTTP and prefix-cache credit are excluded. Three runs describe this CPU/workload,
not a universal guarantee. In a separate matched G32 comparison, B+C decode
medians remained within -0.13% to +0.76% of the previous standard across B1/2/4/8/16.

### Perplexity and KL divergence

Identical prompts and scoring masks, **8,192 scored tokens from 16 WikiText-2
test article windows**, common BF16 teacher. This is an English-prose subset,
not full-corpus WikiText perplexity. Lower PPL and KL are better. These
checkpoint-level scores were not rerun per operating system; ISA-dependent
floating-point rounding may still produce negligible numerical differences.

| Engine / checkpoint | Tensor payload, MB | Perplexity | Mean KL to BF16, nats |
| --- | ---: | ---: | ---: |
| BF16 teacher | 1,505.8 | 14.38556 | 0 |
| **This engine, H128 B+C 256** | **424.9** | **15.80467** | **0.060190** |
| llama.cpp Q4_0 `--pure` | 424.9 | 18.02588 | 0.144522 |
| llama.cpp Unsloth Q4_0 | 496.2 | 15.58088 | 0.068388 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.75290 | 0.034693 |
| llama.cpp Unsloth IQ4_XS | 481.6 | 15.16145 | 0.050539 |

At the same **424,934,656-byte tensor budget** as pure Q4_0, B+C H128 has
**12.3% lower perplexity** and **58.4% lower KL**. Unsloth Q4_0 has lower PPL
but higher KL; the larger Q4_K_M and IQ4_XS achieve lower PPL and KL. The full
H128 file is **424,964,864 bytes**. The table makes this storage/quality tradeoff explicit.

### Independent mixed quality check

Calibration mixes German, English, Python code, rendered dialogs and mathematics.
Ten separate documents, reserved before fitting, provide **1,280 scored tokens**:

| Checkpoint | Perplexity | Mean KL to BF16 |
| --- | ---: | ---: |
| Previous four-document MSE16 standard | 13.10860 | 0.106256 |
| **Current B+C 256 standard** | **12.71898** | **0.064178** |

That is **2.97% lower PPL and 39.60% lower KL**, at effectively unchanged measured
CPU throughput. These scores belong to a different corpus from the table above;
the Unsloth candidates were not all evaluated on this mixed suite. Math PPL
increases 0.58% despite improved KL. More data alone did not help MSE16, and
256 documents do not outperform the 40-document B+C control on every suite.

The German **`2 + 2` regression returns `4`** in free greedy generation and
passes the separate expected-token/logit check. This is one regression, not
broad mathematical validation. No parameter tuning followed the final test.

**[Windows comparison details](docs/readme-comparison-2026-09-07.md)** ·
[Linux comparison details](docs/readme-comparison-linux-2026-09-07.md) ·
[Windows speed CSV](docs/results/readme-bc256-2026-09-07/performance.csv) ·
[Linux speed CSV](docs/results/readme-linux-i7-8750h-2026-09-07/performance.csv) ·
[Quality CSV](docs/results/readme-bc256-2026-09-07/quality.csv) ·
[Windows provenance](docs/results/readme-bc256-2026-09-07/manifest.json) ·
[Linux provenance](docs/results/readme-linux-i7-8750h-2026-09-07/manifest.json) ·
[Calibration study and independent test](docs/g32-large-calibration-2026-09-07.md).

See [the standard recipe](docs/quantization.md) for fitting and the unchanged
`.q35h` layout, and [GGUF setup](docs/q4km-native.md) for experimental execution
paths. The [selective Q8](docs/q8-experiments-2026-09-06.md) and
[Q8 head](docs/q8-head-experiment-2026-09-06.md) experiments remain separate.

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
- Extraction versus frozen source executable: exact output-token matches for
  B=1 and B=4 with shared-prefix pages, sixteen generated tokens per request.
- Native text CLI and full-vocabulary logit export: smoke-tested.
- Native HTTP adapter, concurrent request/prefix correctness and packaged Windows
  server with the prepared Hugging Face model: passed locally.
- Windows/Linux builds, kernel tests and packaging pass in GitHub CI. The public
  Hugging Face model is pinned for real-model tests and tag-triggered releases.
- Mixed 256-document calibration, error-compensated fitting and independent quality validation: passed.
- Further context-length/corpus studies, a new llama.cpp quantization sweep and
  standalone Intel i7-8750H model validation: pending.

See [validation details](docs/validation.md) and the [bounded release plan](docs/comparison-plan.md).

Calibrated MSE16 and block-128 error compensation run offline and preserves the DOT4 runtime and payload
size. The standard download passes the free `2+2 = 4` regression. See the
[calibration and benchmark report](docs/g32-large-calibration-2026-09-07.md)
for quality scores, sequential speed measurements, reproduction and remaining
plan stages.
