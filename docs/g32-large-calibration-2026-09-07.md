# Larger G32 calibration study

The selected 256-document B+C artifact improves independent final PPL by 2.97%
and teacher KL by 39.60% versus original calibrated G32. Matched CPU decode
medians change by -0.13% to +0.76% across batches 1–16; prefill changes stay within
0.88%. No material throughput loss is observed in this series. More data alone
does not help MSE16, and 256 documents do not beat the 40-document B+C control
on every existing quality suite.

## Frozen design

The study follows corpus expansion, resident collection, checkpoint fitting, then
quality and sequential CPU throughput evaluation. It compares two data sizes and
two fitting methods without changing the runtime representation or inference
kernels. All generated checkpoints and external source text remain outside Git.

| Domain | Documents | Tokens | Token share |
|---|---:|---:|---:|
| German | 64 | 65,536 | 25.00% |
| English prose | 64 | 65,536 | 25.00% |
| Code | 51 | 52,224 | 19.92% |
| Dialog | 51 | 52,224 | 19.92% |
| Mathematics | 26 | 26,624 | 10.16% |
| Total | 256 | 262,144 | 100% |

Dataset revisions and the first ten reserved final documents are inherited from
`bc-corpus-v2`. Source selection retains the earlier independent first-eligible
document rule, rather than claiming random sampling. Each document contributes
1,024 tokens. Python-only code and fixed context length remain limitations.
The source audit checks complete documents for 32-word spans shared with final
and prior evaluation sources, cross-split source groups, and the arithmetic
fixture's literal expression. All checks pass for this selection.

The 40-document control contains 10 German, 10 English, 8 code, 8 dialog and 4
mathematics documents from the larger collection. It uses exactly the same GPU
teacher execution and reservoir sampler as the full set, isolating data size
within this run. This does not isolate GPU execution from the previous CPU B run.

Four artifacts are fitted: small/full data with weighted MSE16 (B), and small/full
data with the same importance plus block-128 error compensation (B+C). Damping
remains 0.01. No tuning uses the reserved final set.

## Storage and collection

The resident CUDA teacher processes the documents in one process. CPU workers
accumulate float64 block second moments, storing only full/control aggregates,
and retain per-document diagonal moments for MSE16. Each aggregate is read back
and checked before raw capture deletion. At most two raw documents are pending;
the free-space reserve is 16 GiB. Aggregate moments are committed after each
document so covariance is retained before raw inputs are discarded.

Using the original four-document raw captures, all 187 covariance fit files from
the aggregate path are byte-identical to those from the original raw-data path.
The old raw captures are retained. Larger-run raw files are disposable only after
both covariance and diagonal statistics have been preserved.

## Evaluation protocol

Use the frozen v3 native CLI and CPU benchmark. Compare against the original
calibrated G32 artifact and the previous four-document C artifact. Development
and the existing 8,192-position regression suite select the candidate; the ten
reserved documents provide a final 1,280-position check across five domains.
Report PPL and teacher KL separately, including paired document intervals.

The selected candidate and original baseline then run the same forced-token
P512/N128 workload at batch sizes 1, 2, 4, 8 and 16, eight threads, affinity 21845,
through `benchmark-inference-seq.ps1`, with one warmup and three measured runs.
No quality or calibration jobs run concurrently with those performance tests.
Published results retain raw profiles, commands and binary/checkpoint hashes.

## Status

Corpus selection and isolation audit passed. Covariance compaction equivalence
passed. All 256 documents were collected with exactly one model load. The
collection phase including CPU statistics and persistence took 1,914.49 seconds
(31.91 minutes); this is operational elapsed time, not an isolated GPU benchmark.
Full and 40-document control statistics passed checksum and projection-count
checks. No raw tensor files remain. Final-logit vectors were also removed after
their hashes were recorded. Approximately 56.6 GiB was free after collection.

All four checkpoints were created and checksum-verified. Each is 424,964,864 bytes
and passes the identical-layout/F32-unchanged audit. The shared converter also
reproduces the original calibrated G32 artifact byte-for-byte with its old
importance inputs (SHA256 `8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d`).
The 16-test CTest suite passed.

## Development results

Six documents, 1,024 scored positions. Relative changes against the original
calibrated G32 checkpoint; negative is better.

| Candidate | PPL | PPL change | Teacher KL | KL change |
|---|---:|---:|---:|---:|
| Original G32 | 4.851430 | baseline | 0.07183477 | baseline |
| 40 documents, B | 4.837583 | -0.2854% | 0.06994570 | -2.6297% |
| 256 documents, B | 4.841936 | -0.1957% | 0.07220013 | +0.5086% |
| 40 documents, B+C | 4.705461 | -3.0088% | 0.04743694 | -33.9638% |
| 256 documents, B+C | 4.652825 | -4.0937% | 0.04649399 | -35.2765% |

Increasing data size alone does not help diagonal MSE16 on this suite. With
error compensation, the larger set improves PPL by 1.1186% and KL by 1.9878%
relative to the 40-document B+C control. The large B+C candidate's paired
document bootstrap interval versus original G32 is [-7.4209%, -1.0274%] for PPL
change and [-0.038082, -0.009465] for absolute KL change (2,000 draws, seed 1234).
These intervals describe this small document set, not population-wide accuracy.
The full-versus-small B+C intervals include zero: [-2.2418%, +0.1395%] for PPL
and [-0.005414, +0.003327] for absolute KL. The extra benefit from data size is
therefore less certain than the benefit of error compensation versus baseline.

Development gains vary by domain. Large B+C PPL/KL changes are -10.82%/-42.60%
for German prose, -6.66%/-44.51% for English prose, -0.76%/-28.64% for code,
+0.20%/+28.57% for mathematics, -7.12%/-13.69% for rendered chat, and
+0.50%/-53.69% for the longer recurrent-code window. These are very small
per-domain samples; the aggregate gain does not imply every domain improves.

## English regression results

Sixteen documents, 8,192 scored positions; the same frozen reference cache and
native CLI as the original calibrated G32 comparison.

| Candidate | PPL | PPL change | Teacher KL | KL change |
|---|---:|---:|---:|---:|
| Original G32 | 16.040874 | baseline | 0.09645640 | baseline |
| 40 documents, B | 15.975891 | -0.4051% | 0.09956839 | +3.2263% |
| 256 documents, B | 16.089494 | +0.3031% | 0.09944746 | +3.1009% |
| 40 documents, B+C | 15.699950 | -2.1253% | 0.06045391 | -37.3251% |
| 256 documents, B+C | 15.804668 | -1.4725% | 0.06019021 | -37.5985% |
| Previous C, four documents | 15.697961 | -2.1377% | 0.06015055 | -37.6396% |

Large B+C improves both metrics over original G32. Its paired document intervals
are [-2.8022%, -0.1183%] for PPL change and [-0.039607, -0.032796] for absolute
KL change. However, the smaller B+C control has better English PPL. Compared
directly with previous C, large B+C changes English PPL by +0.6797% and KL by
+0.0659%, while mixed development PPL improves 2.1414% and KL improves 34.7659%.
Thus the larger corpus is a distribution tradeoff, not a universal improvement.

## Candidate decision before final evaluation

Large B+C was selected for the independent final check because it has the best
mixed development point estimates for both metrics and improves both over
original G32 on the English regression suite. This decision was recorded in
`candidate-decision.json` before evaluating the reserved set. Original G32 and
previous C are the fixed references. No fitting or parameter tuning follows the
final evaluation. The final set is balanced across five domains (two documents
each), not weighted according to the calibration mixture's token shares.

## Independent final check

Ten reserved documents, 1,280 positions. No checkpoint was refitted after this
evaluation. Relative changes are against original calibrated G32.

| Candidate | PPL | PPL change | Teacher KL | KL change |
|---|---:|---:|---:|---:|
| Original G32 | 13.108597 | baseline | 0.10625601 | baseline |
| Previous C, four documents | 13.086784 | -0.1664% | 0.09059745 | -14.7366% |
| 256 documents, B+C | 12.718975 | -2.9723% | 0.06417807 | -39.6005% |

The new candidate improves PPL by 2.8105% and KL by 29.1613% relative to previous
C on this final set. Versus original G32, its paired document intervals are
[-5.3420%, -0.8529%] for PPL change and [-0.052815, -0.030880] for absolute KL
change. This remains a small final sample and does not establish universal gains.

| Final domain | PPL change | KL change |
|---|---:|---:|
| German | -7.3665% | -40.6349% |
| English | -1.5342% | -48.1513% |
| Code | -2.1851% | -29.3301% |
| Dialog | -4.1665% | -27.0406% |
| Mathematics | +0.5782% | -48.6922% |

Math PPL regresses slightly despite better teacher KL. Overall improvements must
not be described as every domain improving in both metrics. The smaller B+C
control was not evaluated on this final set; no claim of final-set superiority
of 256 over 40 documents can be made.

The chosen artifact passes native prefix, scheduler, paged scheduler and the
frozen arithmetic regression (including exact expected greedy token choices).

## Matched CPU performance

AMD Ryzen 9 9955HX3D, Windows balanced power scheme, eight threads, affinity
21845, P512/N128, full-logit decode. One warmup and three measured runs per case,
sequential and with alternating case order. No calibration or quality job ran
alongside these measurements. Values below are aggregate decode tokens/second
medians; each percentage compares the same batch size with its original G32
baseline in this new series, not yesterday's absolute throughput.

| Batch | Original G32 | Previous C | Large B+C | B+C decode change | B+C prefill change |
|---|---:|---:|---:|---:|---:|
| 1 | 120.81 | 120.82 | 120.96 | +0.13% | -0.43% |
| 2 | 161.79 | 162.02 | 161.58 | -0.13% | -0.52% |
| 4 | 319.70 | 321.39 | 322.12 | +0.76% | +0.17% |
| 8 | 401.86 | 402.82 | 401.93 | +0.02% | -0.41% |
| 16 | 616.10 | 615.72 | 619.10 | +0.49% | -0.88% |

These sub-percent median differences do not establish a speed optimization.
They support retaining the existing throughput while changing the offline fit.
Raw runs, including isolated slower observations, remain in the archived CSV
and profiles; no measured run was excluded. B-only candidates were not given a
separate speed series because their quality results did not justify selection.

## Artifact and evidence

Selected local artifact: `models/qwen3.5-0.8b/model-calibrated-large-full-bc.q35h`.
SHA256: `013fbfaa03760e759181301ddaf964bb5c200c50c50617afe72557fd65bcbf0a`.
The original baseline and default artifact paths were not overwritten.

A [compact summary](results/g32-large-calibration-2026-09-07/summary.json)
contains aggregate quality, intervals, speed runs, checks and the candidate
decision. Full per-token comparisons, activation manifests, source snapshots and
raw profiles are retained locally under `benchmarks/bc-large-study/` and
`benchmarks/bc-large-evidence/`, outside Git. The archive script preserves that
separation. Weights and statistics arrays are also excluded from Git.
Approximately 52 GiB remained free after evaluation and cleanup.

The result supports the new B+C artifact as a candidate for mixed German/code/chat
use at essentially unchanged measured CPU throughput. Further corpus expansion
is not justified by these results alone: the fitting method gives the strong
gain, while additional data introduces smaller, distribution-dependent tradeoffs.
