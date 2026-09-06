# Selective Q8 experiment, 2026-09-06

The small recurrent gate experiment shows **a small KL benefit but no consistent
PPL improvement**, with roughly 2-3% lower prefill throughput.
The calibrated MSE16 H128/Q4-G32-DOT4 download remains the standard. The optional
Q8 embedding/output experiment was conditional on a successful gate experiment
and has not been run.

## Implementation and storage

`--q8-gates` converts `linear_attn.in_proj_b.weight` and
`linear_attn.in_proj_a.weight` in each of the 18 recurrent layers directly from
BF16 to Q8_0. These are 36 matrices of 16 x 1024 weights, combined into 18
`linear_attn.in_proj_ba.weight` tensors in B, A order. The remaining fused
QKV/Z projection keeps the existing H128 DOT4 kernels. Its output buffer
retains QKV, Z, B, A order; the small Q8 tail writes directly into that buffer
without another executor launch or a large concatenation copy.

The candidate has **425,262,272 file bytes**, versus **424,964,864** for the
standard: an increase of **297,408 bytes (0.070%)**. The tensor payload increases
by 294,912 bytes. The audit checks that every retained Q4 payload is byte-for-byte
identical, including the QKV/Z prefix of each split projection.

This recipe changes both precision and basis: the Q8 gates use the original
weight basis instead of H128. Q8 uses scalar absmax/127 fitting with a stored
FP16 scale and 32 signed bytes per group. The existing calibration remains in
effect for the unchanged Q4 weights; it is not used to fit the Q8 gates.
Therefore this experiment cannot isolate a pure bit-width effect. A separate
H128-basis Q8 gate experiment remains a possible follow-up, including reuse of
the prepared H128 activations to reduce input preparation overhead.

## Measurements

| Quality suite | Standard PPL | Q8 gates PPL | Standard KL | Q8 gates KL |
| --- | ---: | ---: | ---: | ---: |
| regression | 16.040874 | 16.017516 | 0.09645640 | 0.09592032 |
| heldout | 4.851430 | 4.854024 | 0.07183477 | 0.07151225 |

KL improves by 0.56% on regression and 0.45% on the held-out screen.
PPL improves by 0.146% on regression but worsens by 0.053% on held-out.
Both document-bootstrap PPL intervals include no change. The KL intervals
are below zero on these bounded suites, supporting a small local KL benefit.

| Workload B / input / output | Standard prefill tok/s | Q8 prefill tok/s | Change | Standard decode tok/s | Q8 decode tok/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 512 / 128 | 2176.95 | 2123.21 | -2.47% | 125.86 | 125.40 | -0.37% |
| 1 / 4096 / 2 | 1920.28 | 1874.48 | -2.39% | n/a | n/a | n/a |
| 4 / 512 / 128 | 2208.79 | 2147.15 | -2.79% | 332.90 | 331.37 | -0.46% |
| 16 / 512 / 128 | 2207.62 | 2145.21 | -2.83% | 639.42 | 637.05 | -0.37% |
| 1 / 8192 / 128 | 1660.10 | 1625.21 | -2.10% | 71.98 | 72.19 | +0.29% |

Speed was measured sequentially with `scripts/benchmark-inference-seq.ps1`,
one warmup and three measured runs per case, using the same frozen executable
for baseline and candidate. The CPU is a Ryzen 9 9955HX3D, with eight threads
and affinity 21845 (`0x5555`). Both candidates use FP16 KV, full-vocabulary
logits, identical forced tokens and no prefix-cache credit. B denotes requests;
batch throughput is compared only with the same batch size. Load and tokenization
are excluded. The two-output-token workload measures long prefill; its single
decode forward is not used to judge sustained decode speed.

The predeclared screen required lower PPL and KL on both suites, correct
arithmetic, and throughput within 3% of baseline. Decode is effectively unchanged
in this series; prefill is about 2-3% slower and near that tolerance boundary.
Three samples are not proof of speed equivalence. Since quality failed the
screen, no extra speed-equivalence series or conditional head experiment was run.

Quality uses identical scoring masks and a common BF16 teacher. Regression is
8,192 scored tokens from 16 WikiText-2 article windows; the separate screening
suite contains six documents and 1,024 scored tokens. Neither is a comprehensive
quality benchmark. Paired document bootstrap intervals in `results.json` use
2,000 resamples and seed 1234; small point-estimate changes should not be read
as broad capability gains.

## Correctness and compatibility

- All 15 CTest tests pass, including a new mixed-projection test against an
  independent FP64 accumulation oracle for scalar and SIMD execution, batches
  1/2/3/4/5/16, and threaded/unthreaded execution.
- Model scheduler/paged-state and prefix-cache checks pass with the candidate.
- Free greedy generation for German `2 + 2` returns `4`; the independent forced
  prefix/answer logit check also passes.
- The current Q4 checkpoint produces byte-identical logits with the old and
  new CLI binaries on the arithmetic regression.
- `.q35h` adds encoding ID 7 for identity-basis Q8_0: row-major blocks, 32 columns
  per group, FP16 scale followed by 32 signed bytes, transform size/seed zero.
  Existing encodings and the container version remain unchanged. Older engines
  reject this new encoding, so experimental checkpoints require this engine.
- `--q8-head` is implemented as an opt-in conversion of the tied embedding/output
  matrix. Its greedy fallback has unit coverage, but **model quality and speed
  have not been measured**. It is not a recommended artifact or a default.

## Reproduction and evidence

Starting from the same BF16 source and existing fitted calibration directory:

```powershell
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-calibrated-q8-gates.q35h --quantizer mse16 --importance-dir benchmarks/plan-calibration-fit --q8-gates
python scripts/audit-q8-artifact.py models/qwen3.5-0.8b/model-calibrated-q8-gates.q35h --out benchmarks/q8-gates-audit.json
# Freeze the built executables in benchmarks/q8-tools before measuring.
python scripts/prepare-q8-benchmarks.py --candidate models/qwen3.5-0.8b/model-calibrated-q8-gates.q35h --label q8-gates --out benchmarks/q8-gates-speed-inputs
pwsh scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/q8-gates-speed-inputs/matrix.json -OutputDir benchmarks/q8-gates-speed -Runs 3 -WarmupRuns 1 -Affinity 21845
```

Use fresh output directories for repeats. Quality commands and all compared
prompt/scoring hashes are preserved in the raw archive. The baseline calibration
and corpus preparation are described in the
[original report](implementation-plan-results-2026-09-06.md).

Candidate SHA-256:
`b1ea388ffa702ac9763918cd1f2ec00c4090a3145ebdd0d7bcc7da1f4d4bf2f0`.

[Results and uncertainty](results/q8-experiments-2026-09-06/q8-gates/results.json) Â·
[Speed samples summarized](results/q8-experiments-2026-09-06/q8-gates/performance.csv) Â·
[Raw profiles, commands, hashes and checks](results/q8-experiments-2026-09-06/q8-gates/raw-results.zip) Â·
[Evidence checksums](results/q8-experiments-2026-09-06/q8-gates/SHA256SUMS).

No experimental weights or build outputs are committed. The published README
comparison and Hugging Face checkpoint continue to describe the validated Q4
standard.
