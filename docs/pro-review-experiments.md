# Local experiments following the independent Pro review

Starting point: `27898c3`, calibrated qev H128/Q4-G32-DOT4, unchanged weights.
All work remains local. Raw evidence is kept under
`benchmarks/qev-pro-review-2026-10-21/`. README/model-card product tables are not
changed by these experiments.

## Measurement and numerical controls

The starting native/browser series was repeated with the existing sequential
runner, eight threads, affinity mask 255, chunk 32, frozen 74/77/579-token
requests, one process warmup and three measured process runs. Resident B1
compute medians were 158.65/159.57/1267.46 ms for WASM and
36.73/38.77/246.50 ms for native. All requests have fresh state and zero decode
forwards. Native uses FP16 KV; browser uses FP32 KV. This comparison is not a
pure measure of virtualization overhead at identical cache precision.

The numerical driver now accepts `--baseline-profile` in addition to the native
reference. It checks FP32 option-logit bits, prompt IDs, usage and complete
answers against the unchanged WASM baseline. The C ABI exports integer logit
bit patterns directly; legacy profiles are compared after reconstructing FP32.
A unit test verifies that a one-bit change fails the gate. The existing
native probability/choice gate remains separate. Instrumentation-only changes
passed the complete 18-request/24-question suite with unchanged bits.

The native oracle now accepts `--chunk`, `--context`, `--profile-ops` and
`--tuning`; configuration is checked before loading. Debug output includes
participants, positions, attention pair counts and tiling details. Tail
executor intervals accumulate across all remainder tokens rather than retaining
only the last call's timing. Linear preparation distinguishes convolution and
head preparation; MLP activation and post-Delta gating have separate fields.
Nested stage/operation fields overlap and must not be summed indiscriminately.

Explicit experiment controls are applied through `qev_tune` before loading or
while the owner is idle. Idle retuning recreates the executor/workspaces while
retaining immutable weights. It does not reuse recurrent/KV request state.
The browser comparison driver supports an alternating `--tuning-grid`: every
candidate warms all three shapes after retuning, and measured requests remain
serial. Retuning/load, cold shape warmups, compute, full worker time, page
roundtrip and heap are recorded separately. The sequential runner still owns
the three process samples and complete process warmup. Two idle-retune controls
passed the FP32 baseline gate; retuning is not counted as an inference speedup.

Archived runtime selection is explicit with `--runtime-path`, so a preserved
baseline bundle can be compared against a rebuilt candidate without relabeling
its binary hashes. No experimental browser flags are used in final speed runs.

Experiments and decisions are recorded below as they complete. A diagnostic
single run is not a published speed claim.

## B: parallel DeltaNet preparation threshold

Thresholds 128/32/16/8/1 were measured separately in an alternating grid with
three process samples and one warmup. Threshold 32 was effectively neutral.
An apparent advantage for threshold 1 correlated with first-trial position, so
a separate fresh-browser A/B series was run. Fresh medians for 128 were
157.40/157.58/1247.95 ms versus 155.91/155.98/1226.38 ms for 1 (see exact JSON).
This small difference does not establish the anticipated larger gain; retain
128 as the product default. The explicit tuning control remains available.
The 24-question candidate suite passed exact WASM bits and native probability
gates. DeltaNet tests now require exact WASM state/output equality and cover
1/3/7/8/16/31/32/33/128-token batches and split head ranges.

## A: exact SIMD packed Q8 activation quantization

A dedicated WASM SIMD translation unit quantizes X1 and X4 activation blocks.
It preserves FP16 scale roundtrips, lane order, integer sums, and the scalar
round-away-from-zero contract. Residual-based tie handling avoids an FP32
addition rounding a value immediately below a half up to the tie. Exceptional
nonfinite or reciprocal-overflow blocks use the existing scalar implementation.
H128 normalization is still folded into the same prepared scales.

Node and actual Chromium field tests cover zeros, signed zeros, half-neighbors,
small/large magnitudes, nonfinite inputs, random blocks, multiple widths and
sign seeds. The complete 24-question model suite has identical FP32 logit bits
and answers; its pre-existing maximum native probability difference is 0.004.

A fresh alternating comparison against the archived 27898c3 runtime used three
measured processes plus one warmup, with unchanged eight-thread/chunk-32
settings and ordinary Chromium flags. Compute medians were
157.63/157.92/1247.61 ms before and 141.32/143.21/1155.91 ms after: about
10.3%/9.3%/7.4% lower latency. Enable SIMD quantization in SIMD builds; scalar
builds and native dispatch remain unchanged. Raw profiles contain binary hashes
and the original baseline archive is retained. Compact results: `results/
qev-pro-review-2026-10-21/A-packed-q8.json`.

## C: MLP SiLU production inside H128 prepare workers

The optional producer runs the existing SiLU function for each four-token tile
immediately before H128/Q8 preparation in the same worker job. Disjoint hidden
rows remain caller-owned. Single-tile and 1--3-token tail paths produce their
rows explicitly. Native execution is unchanged; eligibility is limited to the
WASM H128/DOT4 path. The threshold is independently controlled by
`mlp_prepare_min_tokens` (zero disables it).

Node and actual Chromium tests compare hidden FP32 bits and packed fields
against the serial pipeline across widths and tile counts. The complete model
suite preserves the 24 questions' FP32 option-logit bits. Thresholds 0/4/8/16/32
were measured in an alternating resident-model grid, three runs plus warmup.
The grid is promising but variable; retain zero until fresh-process confirmation.
The fused activation work belongs to prepare time; a near-zero serial activation
subtimer must not be interpreted as eliminating all SiLU cost. Raw evidence:
`C-quality` and `C-performance`; medians in `results/qev-pro-review-2026-10-21/
C-mlp.json`.

## User-requested FP16 versus FP32 KV storage

`kv_cache_bits` selects 16 or 32 for contiguous caches; zero retains the existing
backend choice (native AVX2 family: FP16; WASM: FP32). Shared pages reject an
explicit FP32 setting rather than silently ignoring it. The C ABI validates the
setting before replacing an idle engine. Debug records include the actual
format and per-question allocated KV payload bytes before request release.
This excludes weights, recurrent state, workspaces and allocator high-water
marks. It is not a promise to halve the total browser memory.

The first experiment uses the existing scalar FP16 cache conversion/attention
path. No new precision arithmetic or GPU backend is introduced. Node and actual
Chromium tests compare FP16 reads with exactly widened, already-rounded FP32
values at 1/3/31/32/33 tokens and positions 0/127/1024. The 24-question model
comparison against unchanged FP32 WASM outputs has no choice changes, but the
maximum probability deviation is 0.0081 (0.81 percentage points) and raw logit
difference is 0.08457. The existing 0.005 probability gate correctly failed; it
was not relaxed. This is not a BF16 model quality evaluation.

Three process measurements plus one warmup compare explicit FP32 and FP16 at
the same eight threads/chunk 32, without MLP fusion, to isolate storage choice.
For the 579-token request the actual cache payload halves from 14,229,504 to
7,114,752 bytes (13.57 to 6.79 MiB). The much larger immutable checkpoint and
other runtime state remain. The current FP16 path is slower, consistent with
repeated half conversion in the scalar attention loops. This result concerns
the current implementation and workload, not every possible FP16 WASM kernel.
Keep automatic FP32 for WASM; explicit FP16 remains an experimental comparison
control. Exact timing medians and heap observations: `results/
qev-pro-review-2026-10-21/kv-fp16.json`; raw `kv-fp16-quality` and `kv-performance`.

Reproduce the storage A/B with the browser driver
`scripts/benchmark-browser-decision.mjs --tuning '{"kv_cache_bits":32}'` and
`--tuning '{"kv_cache_bits":16}'`, using the sequential runner, frozen browser
inputs and the other arguments in the raw metadata. Omit the bitwise FP32 gate
for this intentional precision experiment; retain probability/choice reporting.
