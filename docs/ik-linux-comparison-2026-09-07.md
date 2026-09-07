# ik_llama.cpp on Linux / Ryzen 9 9955HX3D

This experiment compares unmodified upstream ik_llama.cpp with the pinned
llama.cpp baseline and the unchanged native H128 B+C 256 engine. All eleven
candidates are measured in one fresh sequential single-request series. The
four existing GGUF files are identical between the two llama implementations;
two additional IK quantizations are generated directly from the archived BF16
model without an importance matrix or fitting on the evaluation set.

## Measurement contract

The machine, compiler and desktop conditions match the
[earlier Linux comparison](readme-comparison-linux-ryzen-2026-09-07.md): Arch
Linux, GCC 16.2.1 Release, eight threads on physical V-Cache cores 0–7
(`0xff`), maximum-performance power policy, Codex automatically minimized and
XFCE compositing disabled, with the display unchanged at 2560×1600 / 240 Hz.
The wrapper restores the desktop even if a benchmark fails.

Performance uses `scripts/benchmark-inference-seq.ps1`, one warmup and three
measured runs, reversing case order on alternate passes. No compilation,
quantization or quality scoring runs concurrently. All candidates use identical
fixed tokens, full-vocabulary logits, FP16 KV and a maximum context of 16,384.
The workloads are B1/P512/N128 (127 decode forwards) and B1/P4096/N2 (long
prefill). Loading, runtime repacking, tokenization and HTTP are excluded.
Neither quality-logit capture nor projection instrumentation is enabled in the
performance series. Medians and min/max are in the
[performance CSV](results/ik-linux-2026-09-07/performance.csv).

Both llama builds are CPU-only, native ISA, OpenMP, flash attention enabled,
with `n_batch=2048`, `n_ubatch=512`, eight decode/prefill threads and no KQV
offload. IK uses its default optimized IQK matrix multiplication and flash
attention, with runtime tensor repacking explicitly enabled (`--ik-repack`).
Repacking changes the in-memory layout, not the source GGUF file. A preliminary
three-run pure-Q4_0 screen measured 120.07 tok/s without repacking and 122.19
tok/s with it; the final table uses the subsequent complete series.

## Results

| Engine / checkpoint | Prefill 512 | Prefill 4,096 | Decode B=1 |
| --- | ---: | ---: | ---: |
| **This engine, H128 B+C 256** | 2,180.04 | 1,910.84 | 122.64 |
| llama.cpp Q4_0 `--pure` | 1,060.22 | 1,069.80 | 108.53 |
| ik_llama.cpp Q4_0 `--pure` | 2,894.20 | 2,520.52 | 122.18 |
| ik_llama.cpp IQ4_K_R4 `--pure` | 1,988.23 | 1,795.99 | 118.16 |
| ik_llama.cpp IQ4_KS_R4 `--pure` | 2,061.58 | 1,838.91 | 129.37 |
| llama.cpp Unsloth Q4_0 | 927.35 | 944.17 | 94.27 |
| ik_llama.cpp Unsloth Q4_0 | 2,733.27 | 2,432.20 | 104.31 |
| llama.cpp Unsloth Q4_K_M | 666.18 | 657.12 | 88.28 |
| ik_llama.cpp Unsloth Q4_K_M | 1,977.99 | 1,795.43 | 98.74 |
| llama.cpp Unsloth IQ4_XS | 926.87 | 835.48 | 94.37 |
| ik_llama.cpp Unsloth IQ4_XS | 1,946.21 | 1,773.20 | 107.50 |

For the same GGUF files, IK improves decode by 10.6–13.9% and prefill by
2.10–2.97× in this series. Native H128 reaches 122.64 tok/s; the smaller
IQ4_KS_R4 reaches 129.37 tok/s, while IQ4_K_R4 reaches 118.16 tok/s.
Mainline prefill has wider run-to-run variation; consult the min/max values
before treating median ratios as precise bounds.

## Quantization and quality

IQ4_K_R4 `--pure` has a **424,934,656-byte tensor payload**, exactly matching
native H128 and pure Q4_0. IQ4_KS_R4 `--pure` has **401,437,952 bytes**, 5.53%
less. Their complete GGUF files are 435,896,352 and 414,992,672 bytes,
respectively. File size includes metadata/tokenizer and is not tensor payload.
IK's `llama_model_size()` sums loaded tensors, which is not the GGUF payload:
it reports 567,966,976 bytes for pure Q4_0 and IQ4_K_R4, and 540,110,336 bytes
for IQ4_KS_R4. The speed CSV keeps on-disk `tensor_bytes` separate from
`loaded_model_tensor_bytes`; the latter is not total RSS and excludes KV and
compute buffers. Native loaded tensor bytes are left blank because its profile
does not expose this API metric. Thus the same storage budget is not a claim
of equal RAM use.

These are IK-specific formats, not claims of compatibility with the mainline
llama.cpp baseline or the native engine.

Quality is measured separately on the same 16 WikiText-2 test article windows
and 8,192 scored tokens as the README. Both new formats and an IK pure-Q4_0
control are scored against the original cached BF16 logits. Cache identities,
actual logit-file hashes, target-token alignment, finite values and scored-token
counts are checked. The reconstructed teacher perplexity is 14.38556024,
matching the original result. These quality runs use one pass without warmup;
their timings are excluded from speed results. Per-token NLL, KL and top-1
agreement are retained locally, along with candidate logit hashes. Disposable
candidate logit dumps are removed after scoring.

| IK checkpoint | Tensor payload, MB | Perplexity | Mean KL to BF16, nats |
| --- | ---: | ---: | ---: |
| Q4_0 `--pure`, control | 424.93 | 18.08595 | 0.145799 |
| IQ4_K_R4 `--pure` | 424.93 | 15.78482 | 0.073289 |
| IQ4_KS_R4 `--pure` | 401.44 | 15.90097 | 0.090599 |

Lower PPL and KL are better. The equal-payload IQ4_K_R4 improves both over
the IK pure-Q4_0 control. Against the existing native B+C score (PPL 15.80467,
KL 0.060190), its PPL is very slightly lower but its KL is higher; this does
not establish a universal quality winner. The smaller IQ4_KS_R4 loses some
quality relative to IQ4_K_R4. This is an English-prose subset, not a broad
chat, code or multilingual evaluation. Neither new quantization was calibrated
on it. Mainline pure Q4_0 previously scored PPL 18.02588 / KL 0.144522;
the IK control shows that backend numerics are not bitwise identical.
See the [quality CSV](results/ik-linux-2026-09-07/quality.csv).

## Batch-16 limitation

The initial stock IK pure-Q4_0 B16/P512/N128 warmup aborted with
`GGML_ASSERT(cgraph->n_nodes < cgraph->size)` in `ggml.c`. Its Qwen3.5 graph
capacity calculation is a likely cause, but no upstream source was patched to
obtain a score. No IK batch-16 throughput is published. This records a failure
of this pinned fork and harness configuration, not a claim that all IK batching
is unsupported. The previous README batch-16 comparison remains a separate
completed native/mainline experiment.

## Build and reproduction

Pinned sources:

- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp/tree/fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf):
  `fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf`, clean upstream worktree.
- llama.cpp: `73a43d1f69345aee8bb186ef4b3172cef892f2e5`.
- H128: `f6cb5cf04a9094670f9041578a3f4e44b94a1395`.
- Unsloth: `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.

The optional adapter builds the existing fixed-token comparison harness against
IK's API. Changes are conditional on `QWEN35_IK_LLAMA`; both mainline and IK
builds were checked. No production inference kernel changes are required.

```sh
cmake -S tools/ik-llama-comparison -B build-ik-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DIK_LLAMA_SOURCE_DIR="$PWD/build-ik-source" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ik-linux --target ik-fixed-cpu-bench llama-quantize -j 8

build-ik-linux/bin/llama-quantize --pure \
  models/ik-comparison/Qwen3.5-0.8B-BF16.gguf \
  models/ik-comparison/Qwen3.5-0.8B-IQ4_K_R4-pure.gguf IQ4_K_R4 8
build-ik-linux/bin/llama-quantize --pure \
  models/ik-comparison/Qwen3.5-0.8B-BF16.gguf \
  models/ik-comparison/Qwen3.5-0.8B-IQ4_KS_R4-pure.gguf IQ4_KS_R4 8
```

The [manifest](results/ik-linux-2026-09-07/manifest.json) preserves revisions,
binary/library/checkpoint hashes, commands and experiment identities. Raw
profiles, per-token quality metrics, command lines, matrices, build logs,
desktop snapshots and the minimization wrappers remain in ignored
`benchmarks/ik-linux-20260907/`. Weights, builds and raw archives are not
committed. The local wrapper has machine-specific X11 and PowerShell paths;
inspect those before reuse and choose a fresh output directory.
