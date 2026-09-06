# Release and comparison status

The bounded standalone comparison is complete. The
[September 6, 2026 report](comparison-2026-09-06.md) contains the actual protocol,
results, reproducible commands and limitations; the [README](../README.md)
highlights the measured speed and quality separately.

## Completed

1. **Independent CPU extraction and publication.** CPU-only CMake build,
   H128/Q4-G32-DOT4 loading and packing, native tokenizer/CLI, sampling, bounded
   HTTP completions, multi-request library, CPU tests and full-logit export.
   The GitHub repository, pinned Hugging Face model and Windows/Linux v0.1.1
   binaries are public. Release CI includes real-model and extracted-package tests.
2. **Frozen comparison inputs.** Fresh pinned llama.cpp build with CPU VNNI,
   downloaded BF16/Q4_0/Q4_K_M/IQ4_XS GGUFs, and a locally generated pure Q4_0
   control. Tensor payloads, actual bit budgets, artifact/binary/DLL hashes and
   source provenance are recorded. All 187 source BF16 tensors match; seven
   derived FP32 tensors have tiny conversion-rounding differences.
3. **Bounded quality evaluation.** 8,192 teacher-forced positions from 16
   deterministic WikiText-2 test article windows, with identical HF token IDs.
   Full-vocabulary PPL, mean/p95 KL and top-1 agreement are retained. A separate
   historical arithmetic regression is repeated without mixing it into the
   corpus aggregate. H128 does not show a perplexity advantage on this subset.
4. **Matched speed measurements.** P=512/1024/2048/4096, N=128/256/512/1024 at
   B=1 with 8/12 threads; static independent B=4/16 with 8/16 threads. A second
   eight-physical-core series checks prefill affinity sensitivity. There are
   125 configurations and 500 sequential runs including warmups, with original
   and controlled-affinity results preserved. Batched llama logits are checked
   against independent single-sequence references.
5. **Auditable README results.** Native-call speed, tensor storage, quality,
   affinity, thread counts, repetition ranges and recipe differences are stated
   explicitly. No HTTP throughput or universal x86 performance claim is inferred.

## Remaining work beyond this comparison

- **Broader task quality:** German/multilingual text, code, transcript cleanup,
  instruction following and an actual task-accuracy suite. The English-prose
  subset and one previously selected arithmetic case do not cover these.
- **Controlled calibration:** the downloaded Unsloth GGUFs identify their
  importance-matrix metadata, but the calibration contents and disjointness
  from WikiText were not verified. Independently reproduce calibrated recipes
  from a documented disjoint corpus before claiming a controlled calibration study.
- **Tokenizer parity:** investigate the native multiple-whitespace segmentation
  mismatch found in one article. The current comparison bypasses it with shared
  external token IDs; it does not establish end-to-end tokenizer equivalence.
- **Serving workloads:** continuous arrivals, HTTP latency/throughput, mixed
  prefill/decode scheduling, cold/warm shared prefixes, total latency and memory
  against an equivalently configured llama server. The current batch test uses
  private states, sequential prompt initialization and static joint decode.
- **Hardware coverage:** Intel i7-8750H validation and further x86 CPUs. Passing
  Windows/Linux CI and one Ryzen benchmark are not exhaustive hardware validation.

Grouped prefix attention, mixed prefill/decode projection batches and H256 remain
deferred. They are not prerequisites for publishing this bounded comparison.

## Measurement rules for follow-up work

Use `scripts/benchmark-inference-seq.ps1` for every timed performance series,
with three measured runs after one warmup, one process at a time and matched
inputs/settings. Freeze binaries and checkpoints; retain JSON profiles, CSV,
commands, hashes, CPU masks and upstream revisions. Exclude logit capture from
speed timing. Run quality separately, aggregate token NLL before exponentiating
and distinguish end-to-end engine arithmetic from isolated quantization error.

Use the [report's reproduction commands](comparison-2026-09-06.md#reproduction)
for the completed matrix. Adapt affinity masks to the target CPU instead of
assuming the development machine's logical-processor numbering is universal.
