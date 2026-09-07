# Calibrated head quality and Q4_K_M recheck

## Acceptance criteria

The user's September 7 clarification supersedes the zero-throughput-loss rule
in the previous handoff. A 5-10% throughput cost can be acceptable for a clear,
substantial improvement in both PPL and teacher KL. The observed development
result of PPL +0.66% and KL -0.78% is explicitly insufficient. No exact minimum
quality percentage has been fixed; tiny or mixed improvements do not qualify.

## Evaluation scope

First compare calibrated G32 and calibrated G16 with the same frozen v3 CLI on
the 8,192-position regression and independent 384-position audit. Preserve
matched teacher, tokens, scoring masks and per-document uncertainty. These are
bounded evaluations, not full-corpus or general-capability claims.

Then remeasure the current Q4_K_M implementation against the original K binary
and current calibrated G32. Use identical P512/N128 workloads at B1/B2/B4/B8/B16,
eight threads and affinity 21845, sequentially with one warmup and three measured
runs. Keep full-logit behavior fixed. Report prefill and decode separately.

The frozen tools are in `benchmarks/g16-calibrated-batch/tools-v3/` and the
original K performance control is `benchmarks/q8-tools/qwen35_cpu_bench.exe`.
Local outputs go to `benchmarks/quality-k-recheck-2026-09-07/`. No inference
kernel or quantizer is changed during this measurement task.

## Calibrated G16 results

| Suite | Positions | G32 PPL | G16 PPL | PPL change | G32 KL | G16 KL | KL change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regression | 8,192 | 16.040874 | 16.061301 | +0.13% | 0.096456 | 0.094371 | -2.16% |
| Independent audit | 384 | 6.132137 | 6.021578 | -1.80% | 0.069196 | 0.064710 | -6.48% |
| Development (previous run) | 1,024 | 4.851430 | 4.883446 | +0.66% | 0.071835 | 0.071274 | -0.78% |

Lower is better. The fresh regression reproduces the archived G32 baseline.
G16 improves the small audit, but does not demonstrate the required substantial
improvement in both metrics across suites. Keep calibrated G32 as the standard;
do not spend further kernel work merely to rescue this G16 quality point.

The regression paired document bootstrap gives a 95% interval of -0.40% to
+0.64% for PPL change: this supports neither a clear PPL gain nor a clear loss.
The KL reduction is clearer (absolute delta interval -0.00369 to -0.00054),
but small. Audit uncertainty and per-domain results are retained in the raw report.

Scheduler, scheduler with shared KV pages, and prefix-cache model checks all
passed for the calibrated G16 artifact. No inference code changed in this task.

## Q4_K_M quality and speed

The current native K implementation reproduces all 8,192 archived per-token NLL
and teacher-KL values exactly in this run. Regression PPL is **14.784321** and
teacher KL **0.034708**: **7.83% lower PPL and 64.02% lower KL** than current
calibrated G32. This is a substantial quality benefit on this regression suite.
Native execution is not bit-identical to llama.cpp: direct native/reference KL
is 0.003703, and reference PPL is 14.752902. Do not infer equivalence by subtracting
teacher KL values.

All 60 speed runs completed (15 cases, one warmup plus three measured runs).
The table reports median aggregate decode tokens/s for the matched P512/N128
workload. B1 is single-request throughput; larger batches are aggregate throughput.

| Batch | Custom G32 | Current K | Original K | Current K vs G32 | Current K vs original K |
|---|---:|---:|---:|---:|---:|
| 1 | 124.57 | 96.89 | 93.14 | -22.22% | +4.02% |
| 2 | 167.61 | 167.24 | 156.17 | -0.22% | +7.09% |
| 4 | 328.62 | 228.91 | 211.58 | -30.34% | +8.19% |
| 8 | 415.87 | 239.92 | 219.61 | -42.31% | +9.24% |
| 16 | 643.89 | 234.81 | 211.36 | -63.53% | +11.09% |

The current K path improves over the original across this matrix, but fails a
5-10% loss budget against Custom G32 except at B2. In particular, B16 is not
near parity. These are same-series comparisons; yesterday's absolute rates are
not used as today's speed baseline.

Median aggregate prefill tokens/s (measured separately from decode):

| Batch | Custom G32 | Current K | Original K |
|---|---:|---:|---:|
| 1 | 2139.37 | 440.53 | 380.87 |
| 2 | 2144.96 | 440.62 | 381.10 |
| 4 | 2160.86 | 441.62 | 379.94 |
| 8 | 2142.54 | 436.12 | 378.56 |
| 16 | 2149.53 | 437.68 | 378.32 |

Current K prefill improves about 15-16% over original K, but remains roughly 79%
slower than Custom G32. This substantial prefill gap matters independently of
the decode target. Long-context and production prompt-distribution tests are
outside this recheck.

## Decision and next useful work

Keep calibrated Custom G32 as the performance default. Do not promote calibrated
G16: its quality benefit is too small and inconsistent across suites. Preserve
Q4_K_M as a quality reference and experimental execution option, not as a
performance-equivalent replacement.

For a 10% loss limit against this run's G32, current K would need roughly 16%
more B1 throughput and 2.47 times its current B16 throughput. That is a materially
larger batch-kernel task than recovering a few percent. The measurements do not
identify a specific bottleneck by themselves; renewed K optimization needs
projection/batch attribution and a bounded, measurably different kernel design.

The lower runtime-risk custom direction is a bounded better-fitting experiment
within the existing G32 representation, reusing the existing calibration captures.
It may improve quality without changing stored layout or runtime work, but the
quality improvement remains an untested hypothesis. Full-covariance/error-
compensated fitting has not been implemented. Do not promise Q4_K_M quality from
it or begin a large format redesign based on these measurements alone.

## Evidence and reproduction

[Archived raw results](results/quality-k-recheck-2026-09-07/) contain all five new
quality evaluations, per-token CSVs, paired document-bootstrap reports, model
checks, command lines, executable/checkpoint hashes and speed profiles. The
`speed/metadata.json` records affinity, source state and upstream identity;
`speed-summary.json` retains all three measured samples alongside medians.
`SHA256SUMS.json` identifies the archived files. Weights and executables remain
local and are excluded from Git.

`scripts/compare-quality-pair.py` checks teacher/configuration and token alignment
before computing PPL/KL changes and document-bootstrap intervals. A self-comparison
of the archived G32 evaluation returned exactly zero changes. Inference code was
unchanged, and the three calibrated G16 model-state checks passed. The previous
v3 build had already passed all 16 CTests.

The local `run-quality.py` and `run-k-recheck.py` drivers are archived with the
results. They intentionally require fresh evaluation output directories. Use a
fresh output directory for any repeat of the speed harness:

```powershell
pwsh -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/quality-k-recheck-2026-09-07/speed-matrix.json -OutputDir benchmarks/quality-k-recheck-2026-09-07/speed-repeat -Runs 3 -WarmupRuns 1 -Affinity 21845 -SpeedTolerance 0.05
```

No further benchmarks or optimization jobs remain running from this task.

