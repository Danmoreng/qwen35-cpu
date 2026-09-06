# Qwen3.5 CPU

A small C++20 inference engine specialized for **Qwen3.5-0.8B text inference on
CPUs**, using **calibrated MSE16 H128/Q4-G32-DOT4** weights. No CUDA toolchain, GPU runtime or
general-purpose model framework is required.

The project contains a native HTTP server, text-completion CLI, checkpoint packer,
embeddable multi-request engine, CPU tests and comparison tools. It is an independent
extraction of the validated CPU implementation in qwen35x; see [provenance](PROVENANCE.md).
The HTTP server supports a **limited OpenAI-style `/v1/completions` API**:
raw prompts, greedy or sampled decoding and non-streaming responses. Chat completions and
SSE streaming are not implemented.

## CPU benchmarks: speed and quality

**Calibrated MSE16 H128/Q4-G32-DOT4** combines fast CPU inference with
**16.04087 perplexity** and **0.096456 mean KL to BF16** on the evaluated corpus,
using a **424.9 MB tensor payload**. The following tables compare the current
published model with llama.cpp pure Q4_0 and Unsloth's mixed-precision quants.
Quality and speed are measured separately.

### Speed

Ryzen 9 9955HX3D, eight threads on physical VCache cores (`0x5555`), FP16 KV,
fixed identical tokens and full-vocabulary logits. All five candidates were
measured together in a fresh sequential series: **three measured runs after
one warmup**, alternating case order. Values are median **tokens/s**.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| **This engine, calibrated MSE16 H128** | **2,169.80** | **1,887.38** | **119.94** | **613.34** |
| llama.cpp Q4_0 `--pure` | 1,071.24 | 983.86 | 100.61 | 351.30 |
| llama.cpp Unsloth Q4_0 | 894.28 | 860.69 | 88.84 | 294.86 |
| llama.cpp Unsloth Q4_K_M | 679.80 | 663.53 | 83.91 | 264.86 |
| llama.cpp Unsloth IQ4_XS | 842.57 | 823.99 | 80.21 | 268.08 |

Against equal-payload llama.cpp pure Q4_0, calibrated H128 delivers **1.92–2.03×
prefill throughput**, **1.19× single-request decode** and **1.75× batch-16 decode**,
while also reducing perplexity and KL divergence as shown below.

Prefill columns use one request. Decode uses 512 input / 128 output tokens
and counts 127 actual decode forwards per request. Batch 16 is aggregate
throughput across 16 requests, compared only with batch 16. Load, tokenization
and HTTP are excluded; requests have private state and no prefix-cache credit.
Three runs describe this CPU and workload, not a universal speed guarantee.

### Perplexity and KL divergence

Identical prompts and scoring masks, **8,192 scored tokens from 16 WikiText-2
test article windows**, common BF16 teacher. This is an English-prose subset,
not full-corpus WikiText perplexity. Lower PPL and KL are better.

| Engine / checkpoint | Tensor payload, MB | Perplexity | Mean KL to BF16, nats |
| --- | ---: | ---: | ---: |
| BF16 teacher | 1,505.8 | 14.38556 | 0 |
| **This engine, calibrated MSE16 H128** | **424.9** | **16.04087** | **0.096456** |
| llama.cpp Q4_0 `--pure` | 424.9 | 18.02588 | 0.144522 |
| llama.cpp Unsloth Q4_0 | 496.2 | 15.58088 | 0.068388 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.75290 | 0.034693 |
| llama.cpp Unsloth IQ4_XS | 481.6 | 15.16145 | 0.050539 |

At the same **424,934,656-byte tensor budget** as pure Q4_0, calibrated H128
has **11.0% lower perplexity** and **33.3% lower KL**. The larger Unsloth
mixed-precision recipes achieve lower PPL and KL; the table makes that storage
and quality tradeoff explicit. The full H128 file is **424,964,864 bytes**.

Calibration uses four disjoint English prose documents. On a separate
six-document, 1,024-token screening suite, the current model measures PPL
**4.85143** and KL **0.071835**. These bounded evaluations do not establish
universal quality superiority.

The German **`2 + 2` regression returns `4`** in independent free greedy
generation, with no forced output tokens. The expected prefix and answer also
pass a separate logit-level check. This is one regression, not a broad
arithmetic benchmark.

**[Current comparison and measurement details](docs/readme-comparison-2026-09-06.md)** ·
[Speed CSV](docs/results/readme-calibrated-2026-09-06/performance.csv) ·
[Quality CSV](docs/results/readme-calibrated-2026-09-06/quality.csv) ·
[Raw speed profiles, commands and hashes](docs/results/readme-calibrated-2026-09-06/raw-results.zip) ·
[Quantization and validation report](docs/implementation-plan-results-2026-09-06.md).

See [the calibrated recipe](docs/quantization.md) for the unchanged `.q35h`
layout and offline fitting method. Additional GGUF execution paths and their
limitations are documented in [GGUF setup](docs/q4km-native.md).
The [selective Q8 gate experiment](docs/q8-experiments-2026-09-06.md) is reported
separately; its mixed quality result did not replace the calibrated Q4 standard.

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
./download-model.ps1 -Repo danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 -Revision 59f422b2d410fdaf4a9efc71ff12f278abc2a5d1
./qwen35_cpu_server.exe --model-dir models/qwen3.5-0.8b --threads 8
```

```sh
# Linux: bash, curl and sha256sum
bash ./download-model.sh danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 59f422b2d410fdaf4a9efc71ff12f278abc2a5d1
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
./build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model.q35h --quantizer mse16 --importance-dir benchmarks/plan-calibration-fit
./build/qwen35_cpu.exe --model-dir models/qwen3.5-0.8b --weights models/qwen3.5-0.8b/model.q35h --prompt "Once upon a time" --threads 8 --max-new-tokens 128
```

The command above reproduces the calibrated recipe after collecting and fitting
teacher inputs into `benchmarks/plan-calibration-fit`. The packer uses `cpu-dot4`
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

- CPU-only build and fourteen kernel/unit/evaluation tests: passed locally.
- Real-model scheduler/prefix/shared-page regression: passed.
- Extraction versus frozen source executable: exact output-token matches for
  B=1 and B=4 with shared-prefix pages, sixteen generated tokens per request.
- Native text CLI and full-vocabulary logit export: smoke-tested.
- Native HTTP adapter, concurrent request/prefix correctness and packaged Windows
  server with the prepared Hugging Face model: passed locally.
- Windows/Linux builds, kernel tests and packaging pass in GitHub CI. The public
  Hugging Face model is pinned for real-model tests and tag-triggered releases.
- Broader calibration, a new llama.cpp quantization sweep and standalone
  Intel i7-8750H model validation: pending.

See [validation details](docs/validation.md) and the [bounded release plan](docs/comparison-plan.md).

Calibrated MSE16 fitting runs offline and preserves the DOT4 runtime and payload
size. The standard download passes the free `2+2 = 4` regression. See the
[implementation and benchmark report](docs/implementation-plan-results-2026-09-06.md)
for quality scores, sequential speed measurements, reproduction and remaining
plan stages.
