# Qwen3.5 CPU

A small C++20 inference engine specialized for **Qwen3.5-0.8B text inference on
CPUs**, using **H128/Q4-G32-DOT4** weights. No CUDA toolchain, GPU runtime or
general-purpose model framework is required.

The project contains a native text-completion CLI, checkpoint packer, embeddable
multi-request engine, CPU tests and comparison tools. It is an independent
extraction of the validated CPU implementation in qwen35x; see [provenance](PROVENANCE.md).
**An HTTP/OpenAI-compatible server is not included yet.**

## Performance against llama.cpp

The predecessor's controlled single-request comparison on a **Ryzen 9 9955HX3D**
measured **1.19–1.94× prefill throughput** and **1.17–1.19× decode throughput**
at eight threads against llama.cpp **Q4_0 `--pure`**. Both used the same source
checkpoint, fixed input/output token IDs and FP16 KV caches.

| Historical workload, 8 threads | This engine's predecessor | llama.cpp | Ratio |
| :--- | ---: | ---: | ---: |
| Prefill, 512 tokens | 2,133.47 tok/s | 1,101.73 tok/s | 1.94× |
| Prefill, 4,096 tokens | 1,190.83 tok/s | 1,000.60 tok/s | 1.19× |
| Decode, 512 input / 128 output | 122.42 tok/s | 102.63 tok/s | 1.19× |

These are **September 5, 2026 historical measurements**, not a fresh benchmark of
this repository or a claim about current upstream llama.cpp. Later prefill and
multi-request optimizations are included in this extraction. The historical
llama.cpp revision was `74a7c897f049c17e7080423aa2111776eff6ebbf`.
[Method and raw results](docs/historical-results.md).

A new **Q4_0 / Q4_K_M / IQ4_XS** speed-and-quality comparison remains pending.
No multi-request speedup over llama.cpp, perplexity advantage or KL advantage
is claimed yet. Similar bit widths do not imply equivalent quantization quality.
The [comparison protocol](docs/comparison-plan.md) specifies the next release gate.

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

Windows/MSVC is validated for this extraction. Linux/GCC and Intel AVX2 release
validation remain pending. ISA dispatch is designed for x86 portability, but
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

The CLI performs **raw text completion** with greedy decoding. It does not
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
- Fresh llama.cpp quantization/quality matrix, parallel comparison, HTTP adapter
  and standalone Linux/Intel validation: pending.

See [validation details](docs/validation.md) and the [bounded release plan](docs/comparison-plan.md).
