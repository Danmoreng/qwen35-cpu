# Current CPU comparison

Measured on October 22, 2026. The README compares the calibrated
H128/Q4-G32-DOT4 engine with four llama.cpp GGUFs and six ik_llama.cpp GGUFs.
The initial eleven-candidate comparison shares one sequential performance
series and scores all candidates independently against the same frozen BF16
reference. The README now supplements that series with the later
[native prefill optimization](native-prefill.md), measured against the unchanged
native baseline on the same day. Comparator rows retain the original series.
Use the paired native results to quantify that optimization, rather than
attributing differences between series to code alone.

## Source revisions

| Engine | Measured revision | Local changes |
| --- | --- | --- |
| Native, current README supplement | [`f928eec37ab3`](https://github.com/Danmoreng/qwen35-cpu/commit/f928eec37ab34739e03d4fa3d37fe34d44ff7229) | None; sixteen-token prefill reuse committed in this revision |
| Native, initial full comparison | [`6d38cca33e54`](https://github.com/Danmoreng/qwen35-cpu/commit/6d38cca33e548aaeda2dd27b21fef689d47c7d79) | None |
| llama.cpp b11815 | [`7170cad4d02f`](https://github.com/ggml-org/llama.cpp/commit/7170cad4d02f7907517613931840cbb2367399a3) | None |
| ik_llama.cpp | [`98aca8692ae3`](https://github.com/ikawrakow/ik_llama.cpp/commit/98aca8692ae334e6e23e4d39ac43ce0d21029c35) | None for B1, prefill and quality; existing graph-capacity patch for B16 only |

The two comparator revisions were fetched from their upstream default branches
for this measurement. They are frozen identifiers, not a promise to track future
upstream changes automatically. Native engine code was unchanged in the initial
refresh; the subsequent optimization and its evidence are documented separately.

## Performance contract

- Ryzen 9 9955HX3D, Arch Linux / kernel 7.2.9-arch1-1, GCC 16.2.1 Release.
- Eight threads pinned to physical V-Cache cores 0–7 (`0xff`); no SMT siblings.
- Laptop platform profile `max-power`. The separate Linux CPU controls are
  `amd-pstate-epp` active mode, governor `powersave`, EPP `balance_performance`.
  These controls were left unchanged at the user's request. The governor name
  does not mean the laptop is in its low-power platform mode or has fixed clocks.
- Codex minimized, XFCE compositing disabled, display unchanged at
  2560×1600 / 240 Hz. The local wrapper restores the desktop afterward.
- One warmup and three measured runs, alternating case order, through
  `scripts/benchmark-inference-seq.ps1`: 33 cases and 132 raw profiles in the
  initial full comparison; 18 cases and 72 profiles in the native confirmation.
  No concurrent inference or compilation during performance measurements.
- Identical fixed tokens, full-vocabulary logits, FP16 KV, maximum context 16,384.
- B1/P512/N128, B1/P4096/N2 and B16/P512/N128 for every candidate.
  Decode scores use 127 actual decode forwards per request from P512/N128.
  B16 is aggregate throughput across sixteen private requests; long-prefill N2
  decode is not used as a headline decode score.
- CPU-only comparator builds, flash attention, `n_batch=2048`, `n_ubatch=512`,
  no GPU/KQV offload. IK runtime repacking is enabled. Native prefill uses its
  default 128-token chunks and AVX-512/VNNI DOT4 kernel.
- Loading, load-time repacking, tokenization, HTTP, quality-logit capture and
  prefix-cache credit are excluded. Computation inside the timed forward call,
  including IK's temporary prefill weight conversion, remains included.
- Native prefill uses its internal model-forward timer; comparator prefill uses
  synchronized `llama_decode` elapsed time. Both include final full-vocabulary
  logits. These are compute measurements, not end-to-end request latency.

The [performance CSV](../results/cpu-refresh-2026-10-22/performance.csv)
contains medians and min/max; [individual samples](../results/cpu-refresh-2026-10-22/performance-samples.csv)
retain the warmup and every measured run. In that initial series, native P512
prefill is 2,946.40 tokens/s, about 174 ms of model-forward work, and IK IQ4_KS_R4
reaches 3,689.22 tokens/s. The later [native paired series](native-prefill.md)
measures 2,918.72 baseline versus 3,052.57 optimized tokens/s (+4.59%, about
175 ms versus 168 ms). Its B1/B16 decode is unchanged; absolute values differ
between series even for the same binary, so small cross-engine gaps should not
be treated as a definitive ranking.

Stock IK pure-Q4_0 B16 still aborts. Only the B16 comparator uses the existing
seven-line graph-capacity correction; its scope and fresh numerical validation
are documented in the [IK batch report](ik-batch16.md).

## Quality and size

Each quantized candidate was rescored with its measured engine revision on
16 WikiText-2 English test article windows: 256 prompt tokens and 512 scored
tokens per window, 8,192 scored tokens per candidate. Prompts, targets and scoring
masks are identical. This is an English-prose subset, not full-corpus WikiText
perplexity or a broad application-quality evaluation.

The original BF16 reference logits are retained unchanged. Teacher identity is
recorded by binary, checkpoint and per-window logit hashes; the archived teacher
identity does not supply an upstream Git revision, so none is inferred.
Perplexity is `exp(mean(token NLL))`, not the arithmetic mean of window
perplexities. KL is the mean `KL(teacher || candidate)` in nats. Metrics use
float64 calculations. Quality runs are separate from performance runs and their
timings are not published as speed results.

The [quality CSV](../results/cpu-refresh-2026-10-22/quality.csv) identifies the
actual scoring backend for every row, including direct IK measurements of the
three shared Unsloth files. [Window metrics](../results/cpu-refresh-2026-10-22/quality-windows.csv)
and [input identities](../results/cpu-refresh-2026-10-22/quality-inputs.json)
preserve scoring provenance. Backend outputs need not be bitwise identical.

The later native revision reproduces the original native capture hashes at all
8,192 positions, including every vocabulary logit. Its reported PPL and KL
therefore remain exactly the same on this corpus. See the
[native parity evidence](../results/native-prefill-reuse-2026-10-22/quality-parity.json).

All seven model/checkpoint files match the September comparison by SHA-256.
H128 comes from model revision `f6cb5cf04a9094670f9041578a3f4e44b94a1395`;
Unsloth from `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`. The IK-specific pure
quants were generated from the archived BF16 model without an importance matrix
or evaluation-set calibration. IQ4_K_R4 and IQ4_KS_R4 have 424,934,656 and
401,437,952 bytes of on-disk tensors respectively. Their loaded tensor storage
is larger; disk payload is not total resident memory.

The independent mixed-language/code/math calibration validation is retained as
a separate historical corpus and was not rerun for this English refresh.

## Reproduction and evidence

Build the native engine with:

```sh
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j 8
```

Use native revision `6d38cca` for the initial full-comparison matrix below.
Use `f928eec` and the separate [native matrix](../results/native-prefill-reuse-2026-10-22/matrix.json)
to reproduce the optimized supplement and its paired baseline.

Prepare clean upstream checkouts at the exact revisions above, then build the
fixed-token comparators from this repository:

```sh
cmake -S tools/llama-comparison -B build-llama-refresh -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SOURCE_DIR="$PWD/build-llama-refresh-source" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-llama-refresh --target llama-fixed-cpu-bench -j 8
cmake -S tools/ik-llama-comparison -B build-ik-refresh -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIK_LLAMA_SOURCE_DIR="$PWD/build-ik-refresh-source" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ik-refresh --target ik-fixed-cpu-bench -j 8
```

Build IK B16 separately using the [patch instructions](ik-batch16.md#reproduction).
Adapt the local absolute paths in the published [matrix](../results/cpu-refresh-2026-10-22/matrix.json)
to the matching models and builds. Keep the workload flags and input contents:

```sh
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix /path/to/adapted-matrix.json -OutputDir benchmarks/new-cpu-run \
  -Affinity 255 -WarmupRuns 1 -Runs 3
```

The [manifest](../results/cpu-refresh-2026-10-22/manifest.json) records source
revisions/diffs, binary/library/checkpoint hashes, CPU affinity, environment,
desktop snapshots and raw-file hashes. Published files have a
[SHA-256 inventory](../results/cpu-refresh-2026-10-22/SHA256SUMS).
Raw profiles, full commands, scoring scripts, logs, CMake caches, compile
commands and desktop observations remain under ignored
`benchmarks/cpu-refresh-2026-10-22/`. The desktop wrapper is machine-specific;
inspect its paths before reuse. Builds, weights and raw archives are not committed.

## Comparison with the earlier measurement

The [September 7 report](https://github.com/Danmoreng/qwen35-cpu/blob/6d38cca33e548aaeda2dd27b21fef689d47c7d79/docs/benchmarks/cpu.md)
used llama.cpp `73a43d1f69345aee8bb186ef4b3172cef892f2e5` and
IK `fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf`.
The model hashes and workloads are unchanged, but kernel 7.2.2 became 7.2.9 and
the recorded governor/EPP changed from `performance`/`performance` to
`powersave`/`balance_performance`. Both runs used platform profile `max-power`.
Historical speed differences therefore cannot be attributed solely to upstream
code changes. The initial October table compares engines within one new series;
the current native row is explicitly identified as a later paired supplement.

## Why IK IQ4_KS_R4 prefill changed

The source review compares IK `fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf`
with `98aca8692ae334e6e23e4d39ac43ce0d21029c35`.
The September 17 commit
[`dc31024448b8`](https://github.com/ikawrakow/ik_llama.cpp/commit/dc31024448b8f18eac0cd5c2e200b6c7e015ef7a)
adds missing prefill converters for repacked quantizations, explicitly including
IQ4_KS_R4 and IQ4_K_R4. This enables a different matrix-multiplication path:

1. `MulMat::is_dequant_better()` now selects `Q8_K_R16` for IQ4_KS_R4 when
   there are at least 32 activation rows on the AVX-512 build. Dispatch also
   checks matrix dimensions and available kernels.
2. `iqk_mul_mat()` converts a small tile of up to 32 weight rows into a reusable
   thread-local buffer, then multiplies that tile against the activation rows.
3. `iqk_convert_iq4_ks_r4_q8_k_r16()` resolves packed 4-bit indices and scales
   into an interleaved 16-row 8-bit representation. The corresponding Q8 kernel
   uses AVX-512 VNNI integer dot products and reuses the prepared weights.
4. Single-token decode does not cross the conversion threshold and keeps its
   direct quantized path. This makes a large prefill gain with nearly unchanged
   B1 decode plausible.

Source anchors at the measured revision:
[dispatch](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_mul_mat.cpp#L239),
[tile conversion and reuse](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_mul_mat.cpp#L604),
[IQ4_KS_R4 converter](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_gemm_iqk_quants.cpp#L2844),
[Q8 matrix kernel](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_gemm_kquants.cpp#L1853).

Later commits improve this path further:
[`26fea7a3a519`](https://github.com/ikawrakow/ik_llama.cpp/commit/26fea7a3a519bf076ea06466fef4316d1c01bca7)
aligns the Q8 representation and work buffers to 64-byte reads;
[`fdb8e67b1d59`](https://github.com/ikawrakow/ik_llama.cpp/commit/fdb8e67b1d59178cfed04413cab6d57b897f9a1b)
extends the optimized AVX-512 converter helper to these quantizations.
The latest commit's KT optimization concerns different quantization types and
is not the direct explanation for IQ4_KS_R4's new dispatch.

This conversion includes rescaling and rounding, not merely a permutation of
identical bytes. See
[`Q8KRowRepacker`](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_common.h#L699).
The on-disk IQ4_KS_R4 model is unchanged, but numerical results can change;
quality must be evaluated independently from speed.

### Controlled conversion comparison

A separate diagnostic build of the current revision changes exactly one dispatch
case: `IQ4_KS_R4` returns its original type instead of selecting the temporary
Q8 representation. All other source, flags, weights and workload arguments are
unchanged. Both variants run sequentially, alternating order, with one warmup
and three measured runs, eight threads and affinity `0xff`. No quality run or
compilation runs concurrently. CPU controls and desktop setup match the main
series. The diagnostic build is not used in the README comparison.

| Current IK IQ4_KS_R4 variant | Prefill 512 tokens/s | Prefill 4,096 tokens/s | Decode B1 at P512 tokens/s |
| --- | ---: | ---: | ---: |
| Stock, conversion enabled | 3,710.70 | 3,155.60 | 134.09 |
| Only this conversion disabled | 2,068.52 | 1,843.37 | 133.87 |

At P512, enabling this path improves prefill by **79.39%**, while B1 decode
changes by only 0.17%. The three P512 results span 3,695.49–3,713.57 tokens/s
with conversion and 2,062.32–2,072.51 without it. This strongly explains the
historical 2,060-to-3,689 tokens/s change without relying on the older machine
state. The diagnostic and headline series are separate; use paired diagnostic
values for this causal comparison, not a mixture of the two series.

The [diagnostic evidence](../results/cpu-refresh-2026-10-22/ik-converter-ablation.json)
contains the exact one-line patch, commands, all profiles, binary/library/input
hashes, desktop snapshots and upstream revision. Reproduce it by applying that
patch to a separate checkout of the current revision, building with the same
comparator CMake configuration, and running its four-case matrix through the
sequential runner. This isolates the current conversion path as a whole; it is
not a commit-by-commit attribution of later converter improvements. It also does
not establish numerical equality between the two paths. Published quality uses
the stock engine with its conversion enabled.

### Transfer to the native engine

The baseline native DOT4 prefill kernel already used AVX-512 VNNI, sixteen
output rows and eight-token weight reuse. The follow-up
[native experiment](native-prefill.md) tested temporary expanded weight tiles
and wider token tiles. The selected implementation reuses unpacked weights
across sixteen tokens for calls with at least 64 activation rows, without an
additional weight buffer. Smaller calls retain the old path to protect decode.

The paired result is +4.59% P512 and +3.62% P4096 prefill, with unchanged B1/B16
decode and bit-identical quality captures. Native Q4 unpacking is simpler than
IK's nonuniform lookup and scale handling, so IK's roughly 79% gain does not
transfer directly. The selected kernel preserves group scales, integer values
and accumulation order in the existing isolated AVX-512 translation unit.

## Earlier second-CPU comparison

These September 8, 2026 measurements use the older engine revisions documented
in their evidence. They were not rerun as part of the October Ryzen refresh.

**Intel Core i7-8750H · Ubuntu Linux · six physical cores · GCC 15.2.0
Release · AVX2.** Values are median **tokens/s** from three measured runs after
one warmup, pinned to physical cores 0–5 (`0x3f`). The fixed tokens, FP16 KV,
full-vocabulary logits and timing boundaries follow the September Ryzen protocol. IK uses
runtime tensor repacking; its B16 rows use the same documented graph-capacity
correction. The laptop used the `powersave` governor with Intel P-state, so
these results describe the measured system state rather than a fixed CPU clock.

Sorted by **single-request decode speed**, fastest first. MB remains tensor
payload rather than file size or RAM usage. Bold values mark the best speed in
each column.

| Engine / checkpoint | Decode B=1 ↑ | Decode B=16 ↑ | Prefill 512 ↑ | Prefill 4,096 ↑ | MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| **This engine, H128/Q4-G32-DOT4** | **62.43** | **179.22** | **400.62** | **316.36** | 424.9 |
| ik_llama.cpp Q4_0 | 60.22 | 78.31 | 354.25 | 281.05 | 424.9 |
| ik_llama.cpp IQ4_KS_R4 | 56.88 | 70.69 | 278.71 | 226.17 | 401.4 |
| ik_llama.cpp IQ4_K_R4 | 55.10 | 72.38 | 284.66 | 231.44 | 424.9 |
| ik_llama.cpp Unsloth Q4_0 | 52.76 | 77.54 | 357.27 | 280.87 | 496.2 |
| ik_llama.cpp Unsloth IQ4_XS | 51.06 | 73.83 | 321.56 | 257.78 | 481.6 |
| ik_llama.cpp Unsloth Q4_K_M | 49.48 | 74.39 | 369.84 | 292.09 | 521.6 |

The IQ4_K_R4 and IQ4_KS_R4 files were regenerated locally from the pinned
Unsloth BF16 GGUF with the documented `--pure` recipe and ik_llama.cpp revision
`fe215a8c`. Their tensor payloads match the Ryzen artifacts, but their hashes do
not; no Ryzen quality score is attributed to these local files. Speed and
quality therefore remain separate measurements. Absolute throughput between
the two CPU measurements should not be read as architecture-only scaling because the
machines also differ in core count, ISA, compiler, power policy and desktop
conditions.

[Laptop performance CSV](../results/linux-i7-8750h-2026-09-08/performance.csv) ·
[Laptop benchmark provenance](../results/linux-i7-8750h-2026-09-08/manifest.json)
