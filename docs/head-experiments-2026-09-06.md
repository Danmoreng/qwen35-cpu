# Data-free tied-head experiments

> Historical experiment: the later requirement of no batch throughput loss supersedes
> the optional B1 retention decision below. See [the current handoff](next-session-2026-09-07.md).

This experiment follows the compact-model track in `CODEX_NEXT_EXPERIMENTS_V2.md`,
prioritizing the requested uncalibrated MSE16 backbone and Q4-G16 vocabulary head.
The supplied plan is an experiment specification, not measured evidence. Existing
K-quant working-tree changes are retained; this experiment does not change K dots.

**Decision: retain uncalibrated MSE16 with an identity-basis G16 head as an
optional single-request quality/performance point. Do not replace the calibrated
standard. Reject head rotation as a quality upgrade in this screen.** G16 improves
teacher KL modestly, costs about 3% B1 decode and 15.89 MB, and remains comfortably
above 100 tok/s at P512/N128. The initial G16 batch kernel loses about 34% B16
throughput; it is not recommended for batch serving.

The compact track is complete as a bounded experiment. K-superblock arithmetic,
the retained-K rebaseline, full projection attribution, calibration resampling,
the real-group exhaustive scale oracle and long-context attention changes remain
separate work from the supplied plan. No claim is made that the entire plan has
been implemented or that this is a global optimum.

## Opt-in conversion

```powershell
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-mse16-g16-head.q35h --head-g16
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-mse16-h128-g16-head.q35h --head-g16 --head-basis h128
```

`--head-basis identity|h128` defaults to identity. `--head-g16` defaults off.
MSE16 is the default fitter. G16 requires MSE16 without calibration; rotated heads
also reject calibration inputs. No published artifact or runtime default changes.
The files above must not already exist. Weights and build products remain ignored.

G16 stores eight rows by 32 columns in 160 bytes: two sets of eight FP16 scales
followed by the existing 128-byte DOT4 nibble layout. Encoding IDs 8 and 9 identify
identity and H128 bases. Old readers reject these new IDs. Old runtimes also
reject a transformed G32 embedding role. One tied matrix supplies both lookup and
output; there is no second embedding copy or expanded G32 cache.

Each G16 half is fitted with the production MSE16 fitter by duplicating its sixteen
values, which multiplies the objective by two without changing its minimizer.
Each half also retains its parent's G32 codes and scale as an explicit fallback.
This guarantees nonincreasing local SSE relative to that G32 fit, not improved PPL.
Activation quantization remains G32. The isolated AVX2 kernel applies independent
half corrections and scales, with runtime dispatch and a portable scalar fallback.
The initial batch implementation invokes that kernel for each vector; it does not
yet reuse unpacked weights across vectors. Greedy G16 selection uses full logits.

For a rotated head, lookup applies `D H / sqrt(128)` to the dequantized stored row;
output preparation applies `H D / sqrt(128)` to its input. Transform indices restart
at zero for each row. Seeds and bases are recorded in tensor metadata. Prefixes are
bound to the loaded model object, preventing reuse across these artifacts.

## Evaluation contract

The weight screen freezes 4,096 vocabulary midpoint rows and four seeds (default,
1, 42, 2026), with identity preserved. It reports normalized SSE, row error tails
and sixteen vocabulary strata. Runtime experiments use the default seed; there is
no selection against development text. `tools/head_weight_screen.cpp` is a bounded
representation screen, not a full-matrix or exhaustive scale oracle.

The six existing development documents score 1,024 positions. The regression suite
scores 8,192 positions. A separate 384-position synthetic audit is frozen by
`scripts/prepare-head-audit.py`, including prose, German, mathematics, chat, code,
and code following 4,096 context tokens. Its two code windows share a source and
are one document for bootstrap purposes. No audit text is used to fit weights.
This small authored suite is not a general capability benchmark.

Speed uses `scripts/prepare-head-benchmarks.py` and exclusively the sequential
PowerShell harness, one warmup and three measured runs, affinity 21845, eight
threads, full-vocabulary logits and fixed P512/N128 tokens. P8192 and aggregate B16
must be reported separately. Raw logs and command/hash manifests live under
`benchmarks/head-*`; publish compact summaries with their provenance.

`scripts/audit-head-artifact.py` compares payload hashes and metadata and requires
all 229 non-head tensors to remain identical to the unweighted MSE16 artifact.
G16 files contain 440,857,344 bytes; G32 files contain 424,964,864 bytes. File size
includes metadata and padding and is distinct from tensor payload or process RSS.

## Validation

Sixteen CTests cover the existing engine plus head runtime reference arithmetic,
G16 local error fallback, scalar/AVX2 dispatch, inverse transform and tied lookup,
batch tails, greedy penalties, and artifact metadata validation. Quality-report
alignment now checks positions and targets in all three comparison CSVs; tests
reject shifted targets, positions and missing rows. Model scheduler and prefix
tests run separately on the new artifacts.

## Measured quality

All values below are directed BF16-teacher KL and token-weighted PPL. The controls
in the first two columns retain their archived evaluation provenance. Every
non-head tensor of the new models is byte-identical to unweighted MSE16. The audit
controls were evaluated again with the same frozen executable as the candidates.

| Model | Regression PPL / KL (8,192) | Development PPL / KL (1,024) | New audit PPL / KL (384) |
|---|---:|---:|---:|
| Unweighted MSE16, G32 head | 16.86128 / 0.105864 | 4.81958 / 0.075179 | 6.10787 / 0.072089 |
| Published calibrated MSE16 | 16.04087 / 0.096456 | 4.85143 / 0.071835 | Not measured |
| Unweighted, H128-G32 head | Stopped after development | 4.85603 / 0.077248 | Not measured |
| Unweighted, identity G16 head | 16.80496 / 0.102354 | 4.84565 / 0.072220 | 6.06544 / 0.068069 |
| Unweighted, H128-G16 head | Stopped after development | 4.86844 / 0.074996 | 6.07044 / 0.070108 |

Against unweighted G32, identity G16 changes regression PPL by -0.334% and teacher
KL by -3.315%. Development PPL is +0.541% worse, while KL is -3.936% better. Audit
PPL is -0.695% and KL -5.577%. The paired document-bootstrap NLL intervals cross
zero on all three suites, so the small PPL gains are not established statistically.
Regression and development KL-difference intervals exclude zero; the small audit
interval does not. These are differences in quantization quality, not estimates
of same-checkpoint implementation error. Domain and tail results, including
unfavorable changes, are retained in [the raw summaries](results/head-experiments-2026-09-06/results.json).

The H128-G16 development maximum KL is 3.6844 versus 1.1310 for unweighted G32 and
1.6253 for identity G16. Its small weight-SSE advantage does not justify promoting
it over identity G16. Neither new representation approaches the archived original
Q4_K_M regression KL of about 0.0347.

## Matched speed and memory

Windows, Ryzen 9 9955HX3D, affinity 21845, eight threads, fixed tokens, full logits.
Every series has one warmup and three sequential alternating-order measured runs.
The first P512 series measured G32 at 124.30 and identity G16 at 120.08 tok/s
(-3.397%). A second series after adding explicit head-format/kernel profile fields
is below. It is more variable; both series and all runs are preserved, without
mixing their controls. No near-100 result or borderline promotion is involved.

| P512/N128 B1 model | Decode median tok/s | Measured range | Prefill median tok/s | Peak RSS median MiB |
|---|---:|---:|---:|---:|
| Unweighted G32 | 120.89 | 120.41â€“123.22 | 2158.12 | 470.00 |
| Unweighted identity G16 | 117.29 | 115.06â€“121.40 | 2116.51 | 485.15 |
| Unweighted H128-G16 | 116.92 | 116.35â€“118.78 | 2131.81 | 485.08 |
| Published calibrated G32 | 121.56 | 120.12â€“124.76 | 2143.99 | 470.00 |

The second G16/control ratio is -2.975%. These P512 runs use 128 delivered tokens
and 127 decode forwards, consistently for all candidates. RSS is process memory,
not weights alone. The G16 tensor payload is 440,827,136 bytes; the file is
440,857,344 bytes. The first-series profiles predate the precise head labels;
their inventories identify G16, and the actual head kernel was AVX2. The second
series explicitly reports `identity-q4-g16-dot4` and `q4-g16-avx2`.

| Separate workload | Unweighted G32 | Identity G16 | Interpretation |
|---|---:|---:|---|
| P4096 prefill tok/s | 1925.63 | 1918.31 | Similar prefill |
| P8192/N128 B1 decode tok/s | 71.48 | 68.13 | Both below 100 at long context |
| P512/N128 B16 aggregate decode tok/s | 634.60 | 418.18 | Significant G16 batch regression |

The G16 kernel's missing cross-vector weight reuse is an implementation limitation,
not proof of an inherent G16 batching cost. Do not use these B16 numbers as B1
throughput or the short-context headline as a long-context result.

Raw timing profiles, all runs, command lines, source/binary/checkpoint hashes,
quality token CSVs, artifact inventories, the frozen audit and validation logs
are preserved under [results/head-experiments-2026-09-06](results/head-experiments-2026-09-06/results.json).
Frozen executables remain locally under `benchmarks/head-tools` and
`benchmarks/head-attribution-tools`; no binaries or weights enter source control.
The original conversion executable was not separately frozen before later builds;
conversion logs and exact resulting artifact hashes are retained, so the record
does not claim a captured hash of that original conversion binary.
