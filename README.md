# Qwen3.5 CPU

A small C++20 inference engine specialized for **Qwen3.5-0.8B text inference on
CPUs**, using **H128/Q4-G32-DOT4** or **native Q4_0** weights. No CUDA toolchain, GPU runtime or
general-purpose model framework is required.

The project contains a native HTTP server, text-completion CLI, checkpoint packer,
embeddable multi-request engine, CPU tests and comparison tools. It is an independent
extraction of the validated CPU implementation in qwen35x; see [provenance](PROVENANCE.md).
The HTTP server supports a **limited OpenAI-style `/v1/completions` API**:
raw prompts, greedy or sampled decoding and non-streaming responses. Chat completions and
SSE streaming are not implemented.

## CPU benchmarks: speed and quality

**The standard published model is calibrated MSE16 H128/Q4-G32-DOT4.**
It improves perplexity at essentially unchanged speed and size. This is a new
quantization recipe in the **same `.q35h` format and DOT4 layout**; the runtime
still executes the same kernels. Source builds also support native pure Q4_0
and experimental Q4_K_M GGUF execution.

### Quality

Identical prompts and scoring masks, 8,192 scored tokens from 16 WikiText-2 test
article windows, common BF16 teacher. This is an English-prose subset, not
full-corpus WikiText perplexity. Lower PPL and KL are better.

| Native checkpoint | Tensor payload, MB | Perplexity | Mean KL to BF16, nats |
| --- | ---: | ---: | ---: |
| Legacy H128 | 424.9 | 18.58329 | 0.174841 |
| MSE16 H128, without calibration | 424.9 | 16.86128 | 0.105864 |
| **Calibrated MSE16 H128 (standard download)** | **424.9** | **16.04087** | **0.096456** |
| Pure Q4_0 | 424.9 | 18.03818 | 0.144650 |
| Q4_K_M | 521.6 | 14.78432 | 0.034708 |

Calibrated MSE16 reduces PPL by **13.68%** and KL by **44.83%** versus Legacy,
with exactly **424,934,656 tensor bytes** and a **424,964,864-byte file**.
On the separate six-document, 1,024-token screening suite, PPL improves from
5.16899 to **4.85143**. Unweighted MSE16 is slightly better on that small suite
(4.81958). Calibration uses four disjoint English prose documents; these results
do not establish universal quality superiority. Q4_K_M still has better measured
quality and a larger payload.

The existing German **`2 + 2` regression returns `4`** with both new variants
in independent free greedy generation, with no forced output tokens. The full
expected prefix and answer also pass a separate logit-level check. This is one
regression, not a broad arithmetic benchmark. Pure Q4_0 failed the historical
arithmetic case; the new free-generation check covers the three H128 variants.

### Speed

Ryzen 9 9955HX3D, eight threads pinned to physical VCache cores (`0x5555`),
FP16 KV, fixed identical tokens and full-vocabulary logits. Tokens/s are medians
of three measured runs after one warmup. All performance runs are sequential.

| Native checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16 |
| --- | ---: | ---: | ---: | ---: |
| Legacy H128 | 2,147.90 | 1,880.17 | 117.79 | 605.28 |
| MSE16 H128 | 2,150.71 | 1,882.79 | 117.94 | 605.64 |
| **Calibrated MSE16 H128** | **2,150.69** | **1,884.19** | **118.35** | **603.26** |
| Pure Q4_0 | 2,013.67 | 1,782.53 | 118.49 | 596.75 |
| Q4_K_M | 369.82 | 362.09 | 87.31 | 209.42 |

Prefill columns are single-request forward throughput; decode uses 512 input /
128 output tokens and counts 127 actual decode forwards per sequence. Batch 16
is aggregate throughput, compared only with batch 16. Load, tokenization and
HTTP are outside timing; requests have private state and no prefix-cache credit.

Both MSE16 variants stay within a 3% slowdown tolerance across all five tested
workloads. A separate six-measurement follow-up finds calibrated decode **-0.33%**
versus Legacy at batch 16 and **+0.52%** at an 8,192-token prompt. This supports
essentially unchanged speed; it is not a statistical proof of equivalence or a
speed guarantee for other CPUs. Quality gains come from offline conversion.

**[Full report, tests and scope](docs/implementation-plan-results-2026-09-06.md)** ·
[Quality and gates](docs/results/implementation-plan-2026-09-06/quality-and-gates.csv) ·
[Speed with min/median/max](docs/results/implementation-plan-2026-09-06/performance.csv) ·
[Six-run follow-up](docs/results/implementation-plan-2026-09-06/performance-followup.csv) ·
[Raw profiles, commands and hashes](docs/results/implementation-plan-2026-09-06/raw-results.zip).

Earlier llama.cpp speed comparisons used Legacy weights and separate timing
series; they are retained as [historical comparisons](docs/comparison-2026-09-06.md),
including [native Q4_0 versus the identical llama.cpp checkpoint](docs/native-q4_0-2026-09-06.md).
Their speed ratios are not reattributed to the new model. See
[GGUF setup](docs/q4km-native.md) and [quantization recipe details](docs/quantization.md).
Released v0.1.1 binaries do not include GGUF support; the new H128 artifact keeps
the existing `.q35h` layout.

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
./download-model.ps1 -Repo danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 -Revision cc7df08da7ef7ac15db62e80b4eda85e19a143da
./qwen35_cpu_server.exe --model-dir models/qwen3.5-0.8b --threads 8
```

```sh
# Linux: bash, curl and sha256sum
bash ./download-model.sh danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 cc7df08da7ef7ac15db62e80b4eda85e19a143da
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
./build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model.q35h
./build/qwen35_cpu.exe --model-dir models/qwen3.5-0.8b --weights models/qwen3.5-0.8b/model.q35h --prompt "Once upon a time" --threads 8 --max-new-tokens 128
```

The packer defaults to `cpu-dot4` and **unweighted `mse16`**. It rejects other
layout selections. The published standard artifact adds activation calibration;
reproducing that recipe requires `--importance-dir` with the fitted calibration
data. Use `--quantizer legacy-absmax15` only to reproduce the old artifact.
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
- Fresh llama.cpp quantization/quality matrix, parallel comparison and standalone
  Intel i7-8750H model validation: pending.

See [validation details](docs/validation.md) and the [bounded release plan](docs/comparison-plan.md).

Offline `--quantizer mse16` conversion improves H128 perplexity
while retaining the existing DOT4 runtime and payload size. Optional
`--importance-dir` enables calibration in the correct H128 basis. Both measured
variants preserve the free `2+2 = 4` regression. See the
[implementation and benchmark report](docs/implementation-plan-results-2026-09-06.md)
for quality scores, sequential speed measurements, reproduction and remaining
plan stages. MSE16 conversion is the default; the standard download is calibrated MSE16.
