# CPU performance audit: wider DOT4 decode and weight locality

The retained implementation improves native H128/Q4-G32-DOT4 execution on the
Ryzen 5 7600 without changing the checkpoint, activation quantization or
per-output accumulation order. A matched final series measures **3.7% faster
single-request decode, 22.2% faster batch-8 decode and 2.4% faster batch-16
decode**. Prefill improves by 2–4%. These are new measurements on this host;
the README's Ryzen 9955HX3D and Intel results remain separate.

## Measured result

AMD Ryzen 5 7600, six physical cores, affinity `0x3f` (CPUs 0–5), GCC 16.2.1,
Release, AVX-512/VNNI, Linux. The existing `powersave` governor and normal
desktop session were retained. No compilation, correctness testing or other
inference benchmark ran concurrently with timed processes.

Both builds use the same calibrated model, fixed prompt/output tokens, FP16 KV,
16,384-token context limit and default 128-token prefill chunks. All cases run
sequentially through `scripts/benchmark-inference-seq.ps1`, with one warmup and
three measured runs, alternating case order. Values below are medians in
tokens/s. Loading, tokenization and HTTP are excluded.

| Workload | Baseline | Optimized | Change |
| --- | ---: | ---: | ---: |
| Decode B1, full logits, P512/N128 | 112.88 | 117.03 | +3.7% |
| Decode B1, fused greedy, P512/N128 | 112.74 | 117.29 | +4.0% |
| Decode B8, full logits, P512/N128 | 369.76 | 451.68 | +22.2% |
| Decode B16, full logits, P512/N128 | 562.29 | 575.60 | +2.4% |
| Prefill B1, 512 tokens, full logits | 1,515.98 | 1,577.45 | +4.1% |
| Prefill B1, 4,096 tokens, full logits | 1,265.14 | 1,295.69 | +2.4% |

Decode counts 127 actual forwards per request. B8 and B16 are aggregate
throughput across private requests, compared only with the same batch size.
The greedy row also uses fixed output tokens for timing, so its selected tokens
are validated separately. P4096/N2 supplies the long-prefill measurement; its
single decode forward is excluded from decode speed claims. All raw values,
including that short decode sample, remain in the CSV and local profiles.

[Final medians and ranges](results/cpu-performance-audit-2026-09-26/decode-comparison.csv)
and [provenance](results/cpu-performance-audit-2026-09-26/manifest.json).
No confidence interval or cross-platform speed guarantee is inferred from three
samples. Windows, Intel and the V-Cache Ryzen were not rebenchmarked.

## What changed

1. **Sixteen-row AVX-512 matvec and fused argmax.** The old decode kernel uses
   eight 256-bit output lanes even on AVX-512/VNNI machines. The new path pairs
   adjacent existing eight-row weight blocks into sixteen lanes. It shares the
   existing wide kernel's unpacking and arithmetic, including signed-activation
   correction, scale multiplication and FMA order. An eight-row tail uses the
   established EVEX kernel. Greedy selection preserves penalties and lowest-index
   tie breaking.
2. **Wide matrix dispatch from eight tokens.** Previously the sixteen-row,
   eight-token kernel required at least 32 tokens. B8 consequently executed two
   four-token tiles with the 256-bit kernel. It now uses the wide eight-token
   tile. B16 already had a specialized sixteen-token 256-bit tile, which helps
   explain its smaller improvement.
3. **Row-pair-first prefill traversal.** Each sixteen-row weight pair now serves
   all token tiles before advancing to the next row pair. Previously the kernel
   traversed all rows again for every eight tokens. The new order improves
   weight locality while retaining the block order within every output sum.
   Token and row tails retain the established fallback arithmetic.

The ISA-specific implementation remains in `src/cpu/q4_dot4_prefill_avx512.cpp`
behind runtime AVX-512/VNNI dispatch. Scalar, AVX2 and AVX-VNNI selection is
unchanged. There is no new packed-weight copy, model format or memory-expanding
weight conversion. Native pure Q4_0 shares the DOT4 kernels, but this report's
full-model speed measurements use the calibrated H128 checkpoint only.

An initial three-variant series separated the cache traversal from the small-batch
dispatch change. With unchanged single-token kernels, B8 went from 371.95 to
371.73 tokens/s with traversal alone, and to 459.92 with wide dispatch. B1
decode stayed near 112 tokens/s. This supports attributing the large B8 gain
to the kernel choice. The subsequent final series above includes the new matvec
and greedy kernels; numbers from the two series are not combined.
[Initial ablation medians and ranges](results/cpu-performance-audit-2026-09-26/comparison.csv).

## Correctness

- All 14 CTests available in this environment pass. The two optional NumPy-based
  evaluation tests were not configured because NumPy is absent; evaluation code
  was not changed.
- The expanded DOT4 test passes 2,700 cases across available scalar/AVX2/VNNI
  backends. It covers eight- and twelve-token batches, larger batches, row/token
  tails, padded output strides, extreme signed activations, different scale
  signs and greedy penalties. Comparisons are bitwise on this host.
- The new optional real-model `cpu_batch_model_test` compares full-vocabulary
  logits bitwise between individual requests and batches of 4, 8, 12, 16, 20
  and 32 requests, using different prompts and lengths across the 128-token
  chunk boundary.
- Baseline and optimized CLI dumps match byte for byte for prompt lengths
  7, 8, 12, 31, 32, 127, 128, 129, 511, 512, 513 and 4,096. Each has eight
  forced positions: **23,838,720 finite full-vocabulary logits** in total.
  [Logit identities](results/cpu-performance-audit-2026-09-26/equivalence.json).
- Four separate unforced greedy completions, including English, German,
  arithmetic and code prompts with repetition penalty 1.05, match the baseline.

The model tests establish implementation equivalence on the tested cases.
They do not substitute for a new corpus quality evaluation; the checkpoint and
published PPL/KL scores were not changed.

## Further opportunities, not measured here

- **B2/B3 and remainder requests:** `matmul_2d_quantized_batch` executes the
  one-to-three-request remainder as separate matvec calls, rereading weights and
  dispatching separately. A dedicated two-request tile could improve these
  cases without changing the quantization. Measure matching batch sizes.
- **Greedy B8 vocabulary head:** `greedy_batch_rows` selects tiles of 16, 4 or
  1 requests, so an eight-request greedy head still uses two four-request calls.
  Adding an eight-request choice is a concrete follow-up to the wide kernel.
  The B8 speed row above deliberately measures full logits.
- **Prefill chunk and DeltaNet work:** chunk size is configurable through the
  engine API, but the benchmark CLI does not expose it. Measure chunk sizes
  separately before changing the default. DeltaNet's recurrent updates still
  use AVX2 and visit the state twice per token; a specialized 128-column kernel
  or a blocked prefill algorithm would need separate numerical validation.

A separate instrumented baseline run attributes 328.62 ms to 128 vocabulary
head calls, including the first prefill head: approximately 2.57 ms per call.
The 248,320-by-1,024 head accounts for 143,032,320 tensor bytes. It remains a
substantial B1 target, but the new result does not prove a memory-bandwidth
limit. The same profile records only 14.75 ms of preparation across single-vector
Q4 operations versus 980.01 ms in their kernels, so further H128 transform
optimization is a lower-priority B1 candidate on this workload.
[Diagnostic operation summary](results/cpu-performance-audit-2026-09-26/operation-summary.csv).
These instrumented timings are excluded from the speed table.

## Reproduction and local evidence

Build commands, baseline revision, binary/checkpoint hashes, case arguments,
affinity and hashes of the raw manifests are in the linked provenance file.
The baseline is commit `3e46a29`; `decode-wide` denotes the retained optimized
candidate. No external inference engine is part of this comparison.

Raw timing JSON, stdout/stderr, command manifests, per-variant source snapshots,
frozen binaries, diagnostic CSV, logit dumps and validation logs remain in the
ignored directory `benchmarks/perf-audit-20260926/`. Only compact summaries and
provenance are included with the report. The pre-existing untracked Bash
benchmark runner was not modified or used.

```sh
cmake -S . -B build-perf-audit -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWEN35_BUILD_SERVER=OFF
cmake --build build-perf-audit -j 6
ctest --test-dir build-perf-audit --output-on-failure
taskset 0x3f build-perf-audit/cpu_batch_model_test models/qwen3.5-0.8b models/qwen3.5-0.8b/model.q35h
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/perf-audit-20260926/decode-matrix.json -OutputDir benchmarks/perf-audit-new-run -Runs 3 -WarmupRuns 1 -Affinity 63
```

The timing command assumes the frozen binaries and matrix are available locally.
For another checkout, build both revisions separately and adapt the executable,
checkpoint and token-file paths before using the same sequential runner.
