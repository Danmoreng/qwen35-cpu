# Qwen3.5 CPU

A small C++20 inference engine specialized for **Qwen3.5-0.8B text inference on
CPUs**, using **H128/Q4-G32-DOT4** weights. No CUDA toolchain, GPU runtime or
general-purpose model framework is required.

The project contains a native HTTP server, text-completion CLI, checkpoint packer,
embeddable multi-request engine, CPU tests and comparison tools. It is an independent
extraction of the validated CPU implementation in qwen35x; see [provenance](PROVENANCE.md).
The HTTP server supports a **limited OpenAI-style `/v1/completions` API**:
raw prompts, greedy or sampled decoding and non-streaming responses. Chat completions and
SSE streaming are not implemented.

## CPU benchmarks: speed and quality

On the **Ryzen 9 9955HX3D**, this engine measured **1.90–1.98× prefill throughput**,
**1.19× single-request decode** and **1.72× decode at batch 16** against the pinned
llama.cpp Q4_0 `--pure`, with equal eight-thread settings and **identical tensor
storage**. These are September 6, 2026 measurements of this standalone engine.
Quality is evaluated separately below; the current H128 checkpoint does not show
a perplexity advantage on the tested corpus.

All values below are **tokens/s**, medians of three runs after one warmup.

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 | Decode B=16, 8 threads |
| --- | ---: | ---: | ---: | ---: |
| **H128/Q4 (this engine)** | 2,201.0 | 1,929.5 | 125.7 | 620.7 |
| llama.cpp Q4_0 `--pure` | 1,114.8 | 1,014.8 | 106.0 | 360.9 |
| llama.cpp Unsloth Q4_0 | 930.0 | 883.9 | 92.8 | 299.6 |
| llama.cpp Unsloth Q4_K_M | 698.5 | 682.2 | 86.7 | 269.3 |
| llama.cpp Unsloth IQ4_XS | 872.4 | 844.8 | 89.2 | 270.9 |

Single-request columns use eight physical cores on the V-Cache CCD (`0x5555`);
the batch column uses eight threads on that CCD (`0xffff`). Both engines use the
same mask within each comparison. Decode uses 512 input / 128 output tokens and
counts 127 actual decode forwards per sequence. Prefill includes the final full
vocabulary head. Both use fixed identical token IDs and FP16 K/V; load, tokenization
and HTTP are outside timing. Batched requests have private state and no prefix-cache credit.

At batch 16, this engine reaches **660.6 tokens/s with 16 physical-core threads**.
llama.cpp pure Q4_0's best tested batch result is **360.9 with eight threads**:
an **83% throughput advantage when each uses its better tested setting**.
The report includes equal-thread comparisons and all tested 8/12/16-thread configurations.

The initial SMT-eligible prefill runs varied substantially. Restricting the same
CCD to one logical processor per core reduced H128's prefill spread to below 0.5%
in the repeated series above. Both series are retained. Affinity and CPU placement
matter; these numbers are not a guarantee for an unpinned server or every x86 CPU.

**Quality: 8,192 scored tokens from 16 WikiText-2 test article windows**, using a
common BF16 teacher and identical externally tokenized inputs. This is an
English-prose subset, not full-corpus WikiText perplexity. Lower PPL and KL are better.

| Checkpoint | Tensor payload, MB | Perplexity | Mean KL to BF16, nats |
| --- | ---: | ---: | ---: |
| BF16 teacher | 1,505.8 | 14.3856 | 0 |
| **H128/Q4 (this engine)** | 424.9 | 18.5833 | 0.17484 |
| llama.cpp Q4_0 `--pure` | 424.9 | 18.0259 | 0.14452 |
| llama.cpp Unsloth Q4_0 | 496.2 | 15.5809 | 0.06839 |
| llama.cpp Unsloth Q4_K_M | 521.6 | 14.7529 | 0.03469 |
| llama.cpp Unsloth IQ4_XS | 481.6 | 15.1614 | 0.05054 |

H128 and pure Q4_0 each store exactly **424,934,656 tensor bytes (4.518 bits/parameter)**.
The downloaded Unsloth recipes use mixed precision and larger tensor budgets.
H128's perplexity is **3.1% higher** than equal-payload pure Q4_0 on this subset;
no general quality superiority is claimed. A separately repeated historical
arithmetic case still favors H128: it selects **4** for “2 + 2”, while pure Q4_0
selects **2**. One successful task example does not override the corpus result.

llama.cpp revision: `73a43d1f69345aee8bb186ef4b3172cef892f2e5`, CPU-only MSVC
AVX-512/VNNI build. Published quant revisions, hashes, calibration caveats, tokenizer
limitations and exact timing boundaries are recorded in the report.

**[Full comparison and all workload tables](docs/comparison-2026-09-06.md)** ·
[Speed CSV with min/median/max](docs/results/2026-09-06/performance-summary.csv) ·
[Quality CSV](docs/results/2026-09-06/quality-summary.csv) ·
[Raw profiles, commands and hashes (ZIP)](docs/results/2026-09-06/raw-results.zip).
The older predecessor comparison remains available as [historical data](docs/historical-results.md).

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
- One inference artifact recipe: H128 transforms, Q4 groups of 32 and CPU DOT4 packing.
  Large projection weights use H128/Q4; embeddings and small retained tensors use
  their prescribed encodings. This is not an all-tensors-four-bit claim.
- Weight packing and sign conversion happen during conversion. Inference reads
  CPU-ready quantized blocks; ordinary loading still allocates runtime metadata,
  scratch and packed projection groups.
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

The packer defaults to `cpu-dot4` and rejects other layout selections. Runtime
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

- CPU-only build and eight kernel/unit tests: passed.
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
