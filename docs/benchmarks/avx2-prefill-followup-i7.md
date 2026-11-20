# AVX2 prefill follow-up on the Intel Core i7-8750H

This study screened three ways to accelerate native H128/Q4-G32-DOT4
prefill. It used the same calibrated qev-0.8b and Qwen3.5-2B checkpoints
and fixed inputs as the earlier four-token DOT4 study. Stock was
`dd2ee600d22bdf4594f5402cc22c7d59f2ee4aa5`. [Earlier study](dot4-pair4-i7.md)

## Recurrent state and attention

DeltaNet already processes four value rows at a time. The first prototype
retained those rows across the entire token chunk. The second retained
blocks of 16 rows so it could still reuse each token's Q and K across four
row groups. Each state row kept its original causal update and reduction
order.

The PV prototypes tried four queries by three eight-lane value vectors and
two queries by four value vectors. The second variant halves probability
broadcasts across four query rows and visits the value panel separately
for each pair. Each output accumulates the same values in the same
key order.

Median changes in uninstrumented model-forward tokens/s relative to stock
in the same series are shown below. A dash marks a case outside that
matrix.

| Prototype | Model | Prompt | Initial | Follow-up | Confirmation |
| --- | --- | ---: | ---: | ---: | ---: |
| DeltaNet, 4 rows | qev-0.8b | 512 | +3.53% | −0.72% | — |
| DeltaNet, 4 rows | qev-0.8b | 4,096 | +1.22% | — | — |
| DeltaNet, 4 rows | Qwen3.5-2B | 512 | +1.01% | — | — |
| DeltaNet, 16 rows | qev-0.8b | 512 | — | +0.72% | −0.88% |
| DeltaNet, 16 rows | qev-0.8b | 4,096 | — | −1.16% | +9.41% |
| DeltaNet, 16 rows | Qwen3.5-2B | 512 | — | +1.11% | −1.02% |
| PV, 4 queries × 3 vectors | qev-0.8b | 4,096 | +0.84% | — | — |
| PV, 2 queries × 4 vectors | qev-0.8b | 512 | — | — | +0.32% |
| PV, 2 queries × 4 vectors | qev-0.8b | 4,096 | +7.09% | +4.29% | −1.83% |
| PV, 2 queries × 4 vectors | Qwen3.5-2B | 512 | — | — | −0.91% |
| PV, 2 queries × 4 vectors | Qwen3.5-2B | 4,096 | — | — | −1.06% |

Both the recurrent loop changes and wider PV tiles stayed experimental
after end-to-end results changed direction across series. The slower PV
confirmation used the same executable bytes as the faster screening series.

Separate profiling put the sum of the qev P512 DeltaNet kernel's wall
times at 113.80 ms for stock, 107.53 ms with four retained rows and
104.74 ms with 16. For qev P4096 the two-query PV variant increased the
sum of PV worker times from 4,884.13 to 5,988.11 ms and full-attention
wall time from 2,291.85 to 2,463.56 ms. Each figure is the median of three
instrumented passes. The worker times are summed across cores.

## Paired DOT4 panels

The original four-token candidate reuses activation broadcasts across
adjacent eight-row panels. Splitting the low and high nibbles into separate
passes cut static YMM stack-reference sites from 62 to 46. Qwen3.5-2B P512
throughput fell from 215.35 to 185.78 tokens/s. The counts describe instruction
sites in the matrix function. [Original patch](../results/decision-tuning-2026-11-20/avx2-pair4.patch)

Separating the FP sums from the live integer accumulators with compiler
constraints cut the count to 54 sites. At P512 on 2B, the new schedule
reached 214.84 and 214.14 tokens/s in two series. The corresponding stock
rates were 204.41 and 206.30 tokens/s, and the original pair4 rates were
216.61 and 215.90 tokens/s.

The selected schedule runs for at least 128 packed vectors on the AVX2
path and processes pairs of adjacent eight-row panels. Smaller batches
and an unpaired panel use the eight-row kernel. The selected build has
54 static YMM stack-reference sites. Each output keeps the same Q4 block
order, weight scale, activation scale and floating-point accumulation.

Rates are medians in tokens/s with measured minimum and maximum in
brackets. The two series started from opposite matrix orders.

| Checkpoint | Prompt | Forward order stock | Forward order paired | Change | Reverse order stock | Reverse order paired | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qev-0.8b | 512 | 469.98 [469.36–471.10] | 477.54 [475.75–489.80] | +1.61% | 469.87 [469.25–474.15] | 488.88 [486.06–489.25] | +4.05% |
| qev-0.8b | 4,096 | 366.18 [366.07–366.67] | 375.96 [373.50–376.62] | +2.67% | 367.24 [367.14–367.46] | 376.92 [374.48–377.48] | +2.63% |
| Qwen3.5-2B | 512 | 206.17 [204.71–206.99] | 215.50 [214.40–217.69] | +4.53% | 204.21 [203.42–205.49] | 216.09 [213.22–218.58] | +5.82% |
| Qwen3.5-2B | 4,096 | 175.49 [175.17–180.38] | 183.53 [182.83–183.59] | +4.58% | 180.80 [147.22–181.00] | 183.53 [183.31–183.91] | +1.51% |

Every measured candidate pass was faster than every stock pass within
each paired case and series. The peak package temperature across the
series was 68 °C and Linux reported zero thermal throttling events.

The candidate passed all 22 local CTest tests. Each stock and candidate
dump matched byte for byte for four forced output positions on both
checkpoints at both prompt lengths. Each record contains all 248,320
FP32 logits. The model-backed chunk test also checked cold and
prefix-restored logits with contiguous and paged FP16 KV storage.

## Validation and reproduction

Each case used one request, full logits, an empty prefix cache, six threads
on physical CPUs 0–5, and strict AVX2/FMA/F16C dispatch. The metric was
`prefill_forward_tokens_per_second`. The laptop was on AC with
`intel_pstate` active, the `powersave` governor, `balance_power`
and `max_perf_pct=80`. All builds were GCC 15.2.0 Release builds.
The runner used one warmup and three measured passes per case.
It reversed the starting matrix order on odd passes.

The manifest records executable and checkpoint hashes, compilation options,
the environment and the matrices. Resolve executable and input paths to
absolute paths before running from the matching source worktree.
[Manifest](../results/i7-avx2-followup-2026-11-20/manifest.json)

```bash
OMP_NUM_THREADS=6 taskset -c 0-5 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix "$matrix" -OutputDir "$output" -WarmupRuns 1 -Runs 3 -Affinity 63
```

[Individual timings](../results/i7-avx2-followup-2026-11-20/samples.csv) ·
[Instrumented stages](../results/i7-avx2-followup-2026-11-20/stage-profile.json) ·
[FP32 logit hashes](../results/i7-avx2-followup-2026-11-20/logit-parity.json)
