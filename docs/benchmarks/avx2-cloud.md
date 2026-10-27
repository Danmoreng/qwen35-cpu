# AVX2 tuning on an EPYC cloud VM

October 27, 2026. Paired measurements against source revision
`49545867464d3c799aef770ef2685b6ef084881b` on the same VM. This is a separate
AVX2 measurement series; it does not replace the Ryzen AVX-512 comparison in
the README or model card.

## Selected change

The AVX2 four-token DOT4 tile consumes one packed weight segment at a time,
rather than keeping all eight unpacked weight vectors live. An empty GCC/Clang
register dependency prevents integer reassociation from creating many live
products and stack spills. Weight row tiles are processed before moving to the
next four-token group so that each small weight tile remains hot across a prompt.

The format, quantization, scales and per-output floating-point FMA order are
unchanged. Each signed int16 partial sum contains at most sixteen products,
bounded in absolute value by `16 * 15 * 128 = 30720`. The final pair reduction
and offset correction still use int32. Other ISA paths retain their kernels.
No additional weight representation or heap scratch buffer is introduced.

Single-request decode retains its original kernel. The final AVX2 matvec is
byte-identical to baseline (418 bytes; SHA-256
`840ac060eeeff714fcaab029cbee6090aa0a78ce3d54bee9bd2e1f61e8d2840e`).
Four-request batched decode uses the modified matrix kernel and can benefit.

## Paired results

Median tokens/s from three measured runs after one warmup. Higher is better.
Prefill below includes engine overhead; model-forward-only rates are also
retained in the CSV. Decode uses P512/N128 and counts 127 actual decode forwards
per request. B4 is aggregate throughput across four private requests, not the
speed of an individual response.

| Workload | Baseline | Selected | Change |
| --- | ---: | ---: | ---: |
| B1 prefill 512 | 342.71 | 383.23 | +11.82% |
| B1 prefill 4,096 | 298.84 | 325.76 | +9.01% |
| B1 decode | 112.70 | 108.36 | -3.85% |
| B4 prefill 512 per request | 348.90 | 396.27 | +13.58% |
| B4 decode, aggregate | 215.38 | 234.31 | +8.79% |

B1 prompt processing falls from **1,493.96 to 1,336.01 ms** for P512, and
**13,706.28 to 12,573.55 ms** for P4096. The median native peak RSS remains
about 489 MB for B1/P512, 532 MB for B1/P4096 and 584 MB for B4/P512.

There is **no demonstrated single-request decode improvement**. Its three
baseline samples span 105.52–113.32 tokens/s; selected samples span
90.40–117.75. The selected kernel's machine code is unchanged. A separate
P512/N512 check with the exact same selected executable measured medians
111.02 baseline / 109.29 selected tokens/s. Differences between these series
must not be presented as an optimization gain. The B4 decode samples are
213.95–219.48 baseline and 224.52–237.02 selected; that improvement concerns
aggregate concurrency throughput only.

## Rejected experiments

Raw trial sources, binaries and measurements remain under ignored
`benchmarks/avx2-tuning/`:

- Streaming weight segments without constraining compiler reassociation
  increased stack spills and regressed prompt processing.
- Paired-row single-token decode regressed substantially.
- Changing AVX2 instruction scheduling from `skylake` to `znver4` did not
  establish a decode improvement. The build configuration was restored.
- Explicit eight-block-ahead prefetching did not establish a decode improvement
  and was removed.

These prototypes are not included in the selected source.

## Correctness and scope

All **19 CTest tests** pass, including **5,040 DOT4 cases** on this AVX2 host.
The existing DOT4 test checks extreme signed activations and weight patterns,
several reduction lengths, row tails, strided output, matvec, matmul and argmax
against the reference. HTTP smoke checks pass for validation, greedy and seeded
sampling, mixed concurrent requests and prefix parity.

Original and selected CLIs produced byte-identical teacher-forced logit dumps
for prompt lengths **7, 31, 65, 127, 128, 129, 512 and 4,096**, including both
sides of the adaptive prefill chunk boundaries. Four output positions per
prompt give **32 full-vocabulary vectors**. Their hashes are retained in
[logit-parity.json](../results/avx2-cloud-2026-10-27/logit-parity.json).
This establishes equivalence for these inputs; it is not a new PPL/KL study.
Windows, other compilers, other CPUs and AVX-512 performance were not measured
in this series.

## Conditions and reproduction

- AMD EPYC 9V74, x86-64 KVM VM; five visible virtual CPUs, cgroup CPU quota
  `400000/100000` (four CPUs), memory limit 16 GiB.
- Debian 13.6, GCC 14.2.0, CMake Release. Default AVX2 translation-unit flags
  `-mavx2 -mfma -mf16c -mtune=skylake`; runtime `avx2+fma+f16c`.
- Four inference threads, inherited and explicit CPU affinity 0–3 (`0x0f`).
- Standard H128/Q4-G32-DOT4 model revision
  `f6cb5cf04a9094670f9041578a3f4e44b94a1395`, unchanged and SHA-256 verified.
- Fixed prompt/output tokens from the retained CPU comparison; FP16 KV,
  full-vocabulary logits, 8,192-token context, no prefix-cache reuse.
- Sequential benchmark runner, alternating case order, one warmup plus three
  measured runs. Compilation and correctness checks finished before timed runs.
- Loading, tokenization, HTTP and prefix-cache credits are excluded. P4096/N2
  measures prefill; one decode forward is not a reliable decode benchmark.

Build the baseline revision separately and retain its `qwen35_cpu_bench`, then
build the selected source with the ordinary Release commands. No new build flag
is needed. Use the exact matrices retained in the
[manifest](../results/avx2-cloud-2026-10-27/manifest.json), adapting only local
paths. The commands used here, from the repository root, were:

```sh
export XDG_CACHE_HOME=/workspace/.cache
export XDG_CONFIG_HOME=/workspace/.config
export XDG_DATA_HOME=/workspace/.local/share
# Replace pwsh's absolute path with your PowerShell 7 installation if needed.
taskset -c 0-3 /workspace/tools/pwsh/pwsh -NoProfile \
  -File scripts/benchmark-inference-seq.ps1 \
  -Matrix benchmarks/avx2-tuning/final-matrix.json \
  -OutputDir benchmarks/avx2-tuning/final -Runs 3 -WarmupRuns 1 -Affinity 15
taskset -c 0-3 /workspace/tools/pwsh/pwsh -NoProfile \
  -File scripts/benchmark-inference-seq.ps1 \
  -Matrix benchmarks/avx2-tuning/batch4-matrix.json \
  -OutputDir benchmarks/avx2-tuning/batch4 -Runs 3 -WarmupRuns 1 -Affinity 15
```

Use new output directories for a rerun. Raw profiles, logs, exact commands,
binary/input hashes, baseline source archive, selected patch, prototype source
snapshots and validation captures remain under `benchmarks/avx2-tuning/`.
Compact [aggregates](../results/avx2-cloud-2026-10-27/performance.csv),
[all final samples](../results/avx2-cloud-2026-10-27/samples.csv) and the
[manifest](../results/avx2-cloud-2026-10-27/manifest.json) retain the evidence.
