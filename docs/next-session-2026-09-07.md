# Next session: calibrated G16 head, 2026-09-07

## Decision at shutdown

Keep the calibrated G32 standard. G16 remains experimental and is **not promoted**:
the user requires no loss of batch throughput at matching batch sizes. No 3% loss
budget applies. The final v3 run fails at B4 and B16. Earlier optional B1 retention
in the data-free report is superseded by this stricter acceptance requirement.
Work stopped after the final benchmark at the user's request; no more jobs are running.

## Final speed result

Same frozen v3 binary, calibrated G32 versus calibrated G16, P512/N128, eight
threads, CPU affinity 21845, one warmup and three measured runs per case. Values
are median aggregate decode tokens/s; B1 is a separate single-request result.

| Batch | Calibrated G32 | Calibrated G16 | Change |
|---|---:|---:|---:|
| 1 | 121.16 | 116.09 | -4.18% |
| 2 | 161.45 | 187.89 | +16.38% |
| 4 | 317.77 | 308.98 | -2.77% |
| 8 | 400.68 | 448.32 | +11.89% |
| 16 | 615.87 | 603.30 | -2.04% |

Raw profiles, run commands, checkpoint/binary hashes, source state and upstream
metadata are in [the archived results](results/calibrated-g16-batch-2026-09-06/speed-v3/metadata.json).
All three series and per-run samples are in
[speed-summary.json](results/calibrated-g16-batch-2026-09-06/speed-summary.json).
v1 used unweighted artifacts; v2/v3 used calibrated artifacts. Do not mix them
into a quality or speed comparison. Earlier v2 also failed the batch gate.

## Quality is mixed, not a demonstrated upgrade

Development evaluation (1,024 positions): calibrated G32 PPL 4.85142955098286,
KL 0.07183477183997745; calibrated G16 PPL 4.883445529958501,
KL 0.07127441495124141. KL improves approximately 0.78%, but PPL worsens 0.66%.
The G32 numbers are archived baseline results. The G16 evaluation used the frozen
v2 CLI with B1 arithmetic; v3 changes batch preparation and execution.
A new 8,192-position regression and independent 384-position audit are pending.
No claim of improved full-suite quality is justified yet.

## Implementation and verification

- Identity-basis G16 head conversion now accepts calibrated MSE16 importance.
  The existing weighted G32 candidate is retained as a local fit fallback.
- The artifact audit proves all 229 non-head tensors are byte-identical to the
  calibrated G32 checkpoint. Only the tied embedding/output head changes.
- v3 uses transient block-major activation tiles, AVX-512 VNNI 16-row/8-vector
  execution and coalesced row jobs; AVX2 and scalar fallbacks remain available.
  ISA-specific code is isolated and runtime dispatched. No expanded weight copy.
- All 16 CTests passed on v3, including scalar/AVX2/auto head checks, tail batches,
  extreme codes, weighted fitting and embedding/head consistency.
- Full-model scheduler/prefix checks for the newly calibrated G16 artifact remain
  pending. Long-context performance is also not covered by this final matrix.
- Earlier K-quant edits are preserved in this checkpoint; that work remains paused
  and separately documented in `pro-review-brief-2026-09-06.md` and its results.

## Resume in this order

1. Address the B4/B16 regressions only if another bounded kernel experiment is
   worthwhile. Inspect the preserved v2/v3 assembly and final profiles. Otherwise
   close G16 as failing the user's throughput requirement and retain G32.
2. For a new kernel, rerun matched G32/G16 cases sequentially with the harness.
   If results are near noise, confirm with balanced repetitions before claiming
   parity. Never accept a slowdown because it is smaller than the old 34% loss.
3. Only after the speed gate passes, run full-model scheduler/prefix checks and
   the full 8,192-position quality regression plus independent 384-position audit,
   comparing calibrated G32/G16 against the same teacher and token positions.
4. Report quality and speed separately, then check long-context/prefill workloads.
   Do not enlarge calibration data or resume head rotation without a new reason.

## Local resume inputs

The following ignored artifacts stay on this machine; weights and build outputs
are intentionally absent from Git:

- Baseline: `models/qwen3.5-0.8b/model-calibrated-mse16.q35h`.
- Candidate: `models/qwen3.5-0.8b/model-calibrated-g16-head.q35h`.
- Frozen current tools: `benchmarks/g16-calibrated-batch/tools-v3/`.
- Calibration: `benchmarks/plan-calibration-fit/`.
- Development manifest: `benchmarks/plan-heldout-v1/quality-windows.json`.
- Independent audit fixture: `benchmarks/head-audit-v1/`; teacher cache:
  `benchmarks/head-audit-cache/`.

Baseline SHA-256: `8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d`.
Candidate SHA-256: `a2d7dcb6b8634d958a4572ebedecd6cccdd6610c3caeda38875d7085368b680e`.
Calibration manifest SHA-256: `5d691b849820816389190606df89aa599d02673aaf67aba96df775af714e6c08`.
Exact conversion inputs and command are archived in `conversion-inputs.json`;
`inventory.json` contains the artifact comparison.

Final benchmark command (use a fresh output directory when repeating):

```powershell
pwsh -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/g16-calibrated-batch/batch-v3.json -OutputDir benchmarks/g16-calibrated-batch/speed-v3-repeat -Runs 3 -WarmupRuns 1 -Affinity 21845 -SpeedTolerance 0
```

The archived quality `commands.json` records the six evaluated windows. Use
`scripts/evaluate-native-gguf.py` and `scripts/quality-report.py` for the remaining
quality work, preserving reference/cache alignment. The final source checkpoint
and archived evidence are ready for continuation; no promotion is pending tonight.
