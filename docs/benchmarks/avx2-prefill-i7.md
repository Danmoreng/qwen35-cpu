# Long-context AVX2 prefill on the Intel Core i7-8750H

With a contiguous FP16 KV cache, the AVX2 prefill path processes up to 16
queries per tile for chunks of at least 128 tokens starting at position 1,024
or later. More queries reuse the unpacked keys and values. The scheduler can
still shrink the tile to keep its workers busy. The first 512 prompt tokens use
the same attention geometry as the baseline.

## Single-request prefill

Each number is the median model-forward rate after one warmup and three
measured runs per executable. The runner alternated case order between runs.
The independent confirmation put the candidate first in the matrix.

| Series | Prompt tokens | Baseline tokens/s | Adaptive tokens/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Initial | 512 | 468.79 | 470.75 | +0.42% |
| Initial | 2,048 | 413.59 | 415.44 | +0.45% |
| Initial | 4,096 | 358.58 | 368.09 | +2.65% |
| Confirmation | 4,096 | 357.51 | 366.82 | +2.61% |

The 4,096-token rows measure the benefit on this six-core laptop. The small
differences at 512 and 2,048 tokens should be interpreted in light of the
individual samples. These runs used a single request, a cold prompt, full
vocabulary logits, six threads pinned to physical cores 0–5, AVX2/FMA/F16C,
and the same H128/Q4-G32-DOT4 checkpoint. Loading and tokenization are outside
the timing interval.

## Validation and reproduction

At prompt lengths 512, 1,151, 1,152, 1,153 and 4,096, four full-vocabulary FP32
logit vectors per prompt matched the baseline byte for byte. The middle three
lengths cover the first eligible 128-token chunk and its tails. The tiled
attention regression test also exercises a 16-query AVX2 tile against the
FP64 reference at position 4,032.

Build the baseline and this revision in Release mode with the same compiler.
The 512- and 4,096-token fixtures are in `docs/results/cpu-refresh-2026-10-22`.
The 2,048-token fixture consists of the first 2,048 IDs of the longer fixture.
The logit checks use the first 1,151, 1,152 or 1,153 IDs where applicable, and
the four forced output IDs `303,22461,4843,57900`. Adapt the executable and
fixture paths in the recorded matrices, then run each matrix from the
repository root.

```sh
taskset -c 0-5 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix MATRIX.json -OutputDir NEW_DIRECTORY -WarmupRuns 1 -Runs 3 -Affinity 63
```

[Individual samples](../results/avx2-prefill-i7-2026-11-18/samples.csv) ·
[Matrices, revisions, hashes and environment](../results/avx2-prefill-i7-2026-11-18/manifest.json)
