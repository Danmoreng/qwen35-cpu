# Four-token AVX2 DOT4 prefill on the Intel Core i7-8750H

The four-token DOT4 patch from the [Ryzen trial](decision-tuning-9955hx3d.md#avx2-matrix-kernel)
shares activation broadcasts across two adjacent eight-row weight panels.
This experiment paired stock and patched builds on an Intel Core i7-8750H
with the calibrated qev-0.8b and Qwen3.5-2B H128/Q4-G32-DOT4 checkpoints.
The patch reproduced the Ryzen speedup for qev at 512 prompt tokens and for
Qwen3.5-2B at both tested lengths. At 4,096 tokens, qev improved in five of
six series, with a wider spread in the measured gain.

## Prefill throughput

Each cell shows the median and the minimum–maximum of three measured passes
in tokens/s after one warmup. A and B start with opposite case orders. The
sequential runner reverses each starting order on odd-numbered passes.

This is fixed-token, single-request model-forward throughput
(`prefill_forward_tokens_per_second`). All cases used full logits,
128-token chunks and six threads pinned to different physical cores
(CPUs 0–5, mask 63). The benchmark profiles confirm AVX2/FMA/F16C,
512 or 4,096 new prompt tokens, and zero cached prefix tokens.

The laptop ran on AC with Linux 7.0.0-43-generic, `intel_pstate` in active
mode, the `powersave` governor, `balance_power` preference and
`max_perf_pct=80`. Turbo was enabled. The configured package limits were
30 W long term and 38 W short term. Both variants used GCC 15.2.0,
CMake 4.2.3 and Ninja 1.13.2. Package temperature peaked at 67 °C. The
kernel's core and package thermal-throttle counters stayed at zero.

### Matched v0.3.2 builds

Both builds used `c2affd50b83edb2e677794993b2a6ff02dba2d4e`.

| Checkpoint | Prompt tokens | A stock | A four-token | A change | B stock | B four-token | B change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qev-0.8b | 512 | 454.24 [453.31–464.75] | 481.70 [480.39–491.59] | +6.05% | 468.23 [467.49–471.98] | 481.58 [475.91–482.84] | +2.85% |
| qev-0.8b | 4,096 | 380.20 [364.45–380.86] | 377.02 [376.40–393.73] | -0.84% | 366.50 [366.04–380.16] | 378.78 [371.01–394.93] | +3.35% |
| Qwen3.5-2B | 512 | 205.74 [204.80–206.46] | 216.12 [213.82–218.28] | +5.05% | 205.98 [202.83–206.46] | 213.97 [210.87–218.02] | +3.88% |
| Qwen3.5-2B | 4,096 | 171.34 [152.26–177.63] | 183.65 [183.63–183.77] | +7.19% | 175.42 [174.29–176.79] | 183.38 [182.82–184.27] | +4.53% |

The patch sped up both Qwen3.5-2B workloads and qev at 512 tokens in both
series. At 4,096 tokens, qev changed direction and the stock and candidate
ranges overlapped in both series. A dedicated third series for that pair
gave **+2.98%**, from 366.81 [362.59–368.05] to 377.76 [373.75–379.91]
tokens/s.

### Matched main builds

The same patch applied verbatim to `fc7e413101cea078ebd36cb2e2cb37fd8022938b`,
the tip of `main` at measurement time. The stock and candidate builds
repeated the full matrix with the same compiler, flags and inputs.

| Checkpoint | Prompt tokens | A stock | A four-token | A change | B stock | B four-token | B change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qev-0.8b | 512 | 466.69 [458.10–470.15] | 480.05 [479.44–486.63] | +2.86% | 461.92 [458.28–464.67] | 481.35 [480.76–487.33] | +4.21% |
| qev-0.8b | 4,096 | 377.66 [366.96–381.35] | 379.38 [376.61–394.53] | +0.45% | 365.27 [365.21–376.71] | 392.99 [378.31–394.11] | +7.59% |
| Qwen3.5-2B | 512 | 205.23 [202.02–206.63] | 215.42 [211.31–219.18] | +4.97% | 203.77 [201.94–206.09] | 217.57 [217.23–218.83] | +6.77% |
| Qwen3.5-2B | 4,096 | 177.57 [175.19–178.80] | 184.94 [184.29–185.08] | +4.15% | 175.42 [175.30–177.67] | 184.37 [183.97–184.76] | +5.11% |

The qev 4,096-token ranges overlap in series A. The dedicated third series
gave **+2.85%**, from 364.66 [358.82–364.86] to 375.05 [372.67–376.62]
tokens/s. For the other three cases every measured candidate pass was
faster than every stock pass within each full series on both revisions.
The dispatch evidence covers the AVX2 DOT4 matrix path on this processor.
The EPYC 9V74 replication remains open.

## Logit parity

For each model and prompt length, `qwen35_cpu` dumped four unrounded FP32
vectors with the forced IDs `303,22461,4843,57900`. Each vector contained all
248,320 vocabulary entries. Each dump was 3,973,160 bytes including the
header and target IDs. All eight stock and patched dump pairs matched
byte for byte with `cmp`, four on v0.3.2 and four on `main`. The DOT4
kernel test passed 5,040 cases per build.

## Reproduction

Build both variants from the same revision in separate worktrees. Apply the
[original four-token patch](../results/decision-tuning-2026-11-20/avx2-pair4.patch)
to the candidate worktree. Build `qwen35_cpu_bench`, `qwen35_cpu` and
`q4_dot4_test` in Release mode with the same compiler, then run the kernel
test for both variants.

Use the checkpoint hashes in the manifest. Fetch the pinned qev package with
`scripts/download-qev-model.py` from `fc7e413` and verify `SHA256SUMS`.
The 2B checkpoint is the standard H128/Q4-G32-DOT4 artifact. Both prompt
fixtures are under `docs/results/cpu-refresh-2026-10-22/`.

The manifest records the first matrix and the starting case order for every
series. Use the binaries from the corresponding revision and variant, then
resolve executable and input paths to absolute paths for your workspace.
Run each matrix from the corresponding stock worktree.

```bash
OMP_NUM_THREADS=6 taskset -c 0-5 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix "$matrix" -OutputDir "$output" -WarmupRuns 1 -Runs 3 -Affinity 63
```

Use the recorded CLI commands and forced IDs for the logit check. See the
[individual samples](../results/i7-dot4-pair4-2026-11-20/samples.csv),
[commands and provenance](../results/i7-dot4-pair4-2026-11-20/manifest.json)
and [logit dump hashes](../results/i7-dot4-pair4-2026-11-20/logit-parity.json).
The manifest names the ignored local directory holding raw profiles, commands,
thermal telemetry, dumps and build logs.
