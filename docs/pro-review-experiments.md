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
`scripts/benchmark-browser-decision.mjs --tuning '{"kv_cache_bits":32,"mlp_prepare_min_tokens":0}'` and
`--tuning '{"kv_cache_bits":16,"mlp_prepare_min_tokens":0}'`, using the sequential runner, frozen browser
inputs and the other arguments in the raw metadata. Omit the bitwise FP32 gate
for this intentional precision experiment; retain probability/choice reporting.

### C fresh-browser confirmation

A separate alternating A/B series (three measured processes plus one warmup,
all shapes warmed, no experimental Chromium flags) compares serial SiLU with
an eight-token producer threshold. Compute medians are recorded under `fresh`
in `C-mlp.json`; all request outputs pass the strict baseline bit gate. This
confirms the gain independently of the larger resident tuning grid. Enable
fusion for eligible multithreaded WASM H128/DOT4 preparation at eight tokens;
retain the serial path for native, one-thread execution and smaller tails.

## P6: executor spin budgets

With SIMD Q8 and the eight-token MLP producer held fixed, an alternating grid
compared 6,553,600/131,072/16,384/1,024/0 polls with eight threads and chunk 32.
Each candidate warmed every shape, followed by three measured processes and
one process warmup. Reduced budgets are slower; retain 6,553,600. Every
candidate passes the strict option-logit/answer baseline gate. This measures
latency, not power or idle CPU consumption. It does not establish the best
budget for other browsers, processors, thread counts or interactive gaps.
Medians: `results/qev-pro-review-2026-10-21/spin.json`; raw `spin-performance`.

## Diagnostic accounting and remaining candidates

A separate single instrumented C comparison attributes 109.94 ms of the long
serial request to MLP activation. Fusion moves that work into the existing
prepare jobs: serial activation falls to 0.76 ms while Q4 prepare rises from
34.87 to 49.78 ms. The instrumented total falls from 1166.99 to 1086.79 ms.
The fused work has not disappeared; the timing category changes as it becomes
parallel. This diagnostic is not used instead of the uninstrumented three-run
performance series. Compact counters: `results/qev-pro-review-2026-10-21/
C-diagnostic.json`; raw `C-diagnostic`.

The initial implementation pass covered the review's three recommended starting
patches, the executor-spin experiment, and the additional user-requested KV
precision comparison. No speed claim is made for the remaining proposed Q4
instruction/tile changes (P4), weighted attention partitions (P5), a new GQA
attention kernel (P7), selected-option head projection (P8), native unforced
greedy batch dispatch (P9), or merged tail jobs (P10). Those were unimplemented
and unmeasured at that checkpoint; subsequent experiments are recorded below. In particular, no assertion about physical register
spills is inferred from WASM locals; optimized V8 disassembly for a new Q4 tile
experiment has not been captured in this pass.

## First-pass consolidated comparison

The accepted configuration at source commit `93981b2` enables exact SIMD Q8
and multithreaded MLP preparation from eight tokens. DeltaNet threshold 128,
spin count 6,553,600 and automatic WASM FP32 KV remain. A new alternating
three-run/one-warmup series compares this combination with the archived
27898c3 runtime. Compute medians in milliseconds:

| Request | Original WASM | Accepted WASM | Lower latency |
|---|---:|---:|---:|
| Short (74 tokens) | 156.70 | 134.37 | 14.2% |
| Snake (77 tokens) | 157.55 | 133.45 | 15.3% |
| Long (579 tokens) | 1248.80 | 1122.76 | 10.1% |

The final request series varies more than the separate A and C controls,
especially on the later native runs. Its native median is 45.44/45.69/297.08 ms,
while a subsequent direct archived/current native control gives
36.12/36.11/239.61 versus 36.49/36.44/239.74 ms. The slowdown does not recur
against the archived native binary; its environmental cause was not identified.
Both series are retained rather than replacing the noisy one. No stronger
claim about a native speed change is made. Native option logits and answers
match the archived native outputs exactly on all three frozen prompts.

All seven WASM builds pass their DOT4, DeltaNet and quantization tests.
Chromium quantization tests pass. The final 18-request/24-question suites at
one and eight threads preserve exact option-logit bits and full answers; the
existing maximum probability difference to native remains 0.004. Invalid KV
widths/types and unknown tuning keys, malformed requests, context overflow,
protocol fixtures, cancellation and fresh-state recovery pass. All 17 native
CTest checks pass. This is numerical/regression coverage on a small panel, not
a new full task-accuracy benchmark.

Compact samples/hashes: `results/qev-pro-review-2026-10-21/final.json` and
`native-control.json`. Raw evidence, commands, affinity, binary/checkpoint hashes
and environment metadata remain in `benchmarks/qev-pro-review-2026-10-21/`.
All changes and artifacts stay local; no repository/model/Space was published.

## P5 and P10: weighted attention rows and merged remainder jobs

Two independently disabled controls were added: `balance_attention_rows:1` and
`merge_tail_jobs:1`. The former finds contiguous boundaries by cumulative
query-key pair count in token-major/head-minor order, with 64-bit work sums on
WASM32. Every partition retains private score scratch. The latter dispatches
all two/three remainder-token matvecs in one weight-tile job, calling the same
single-token kernel separately for each output. One-token tails keep their
existing path. Defaults remain false for both controls.

Coverage/nonoverlap and work bounds pass for heads 1/2/8/16, token counts
1/2/3/7/8/31/32/33/127/128/129, positions 0/1/31/1024/8192 and 1/2/4/8/16
requested partitions, on native and WASM32. Native row-attention outputs are
bit-identical for FP32/FP16 KV and scalar/AVX2/automatic backends at short and
longer positions. Native tail tests cover counts 1/2/3/4/5/8/15/16/17/31/32
and require exact merged/unmerged output bits. Actual Chromium unit checks and
both complete 24-question WASM suites pass the unchanged logit/answer gate.

The performance grid rotates three candidates over three measured processes
so that each occupies every position once (plus process and per-shape warmups).
Control medians are 134.64/133.04/1075.64 ms; balanced attention gives
132.26/132.65/1077.97 ms; merged tails give 133.44/132.97/1078.42 ms. The small
short-request difference for attention is unproven, and neither establishes a
useful consistent end-to-end gain. Both remain experimental and disabled.
No per-worker wall-time benefit is claimed from the mathematical work bounds.
Compact results: `results/qev-pro-review-2026-10-21/P5-P10.json`; raw
`P5-P10-quality` and `P5-P10-performance`.

## P4: actual optimized code and Q4 instruction/tile experiments

Chromium's code-print flag reports TurboFan headers and sizes in this build,
but no instruction listing. A diagnostic run with `--perf-prof`, an explicit
`--perf-prof-path` and `--no-liftoff` captures actual optimized bytes in Linux
jitdump records. `scripts/inspect-wasm-jit.py` extracts the exact
`wasm-function[index]-index-turbofan` record and uses objdump offline. The
format follows the [Linux jitdump specification](https://raw.githubusercontent.com/torvalds/linux/master/tools/perf/Documentation/jitdump-specification.txt).
No browser or model memory is read through a live debugger. The extractor
records skipped truncated files; the selected renderer record is complete.

The control WASM function map identifies defined function 312 / absolute
function index 390, with 26 relaxed FMA sites. Captured x86 code contains VEX
`vpdpbusd` and `vfmadd*ps`. In the eight-token block loop (offsets
0x6740 through the backedge at 0x7178), there are 67 static vector stack-store
sites and 74 stack-load sites, 32 VNNI sites and 16 FMA sites. These are static
sites in real machine code, not dynamic counts or measured cycle costs.
The source-local count alone was not used as evidence of physical spills.
Raw bytes, assembly, code hash and tier are retained in `P4-jit-dump`; counters
are in `results/qev-pro-review-2026-10-21/P4-jit-control.json`.

Three compile-time experiments preserve each output's block order and float
operations: byte shift versus masked word shift; eight tokens with one
four-row half completed across all blocks before the other; and four-row
integer dots completed separately inside each shared weight block. The latter
two are mutually exclusive. All build commands/options and binary hashes are
retained under the `P4-build-*` evidence directories.

The first three-run/one-warmup series gives control 131.35/131.19/1065.89 ms,
word shift 130.29/130.39/1056.67 ms, and separate row halves
138.42/139.44/1129.80 ms. Word shift is an unproven sub-1% difference; row
halves regress about 6%. The second direct A/B gives control
130.92/131.38/1065.80 ms and half-sized integer dots
139.85/139.83/1134.18 ms, another regression. Kernel tests and every candidate's
complete 24-question bit gate pass. Speed runs use the ordinary browser,
without diagnostic JIT flags.

Keep all three compile-time experiment defaults OFF and retain the existing
eight-by-eight tile. After two plausible narrowing variants failed, no wider
blind tile search is performed. Reduced register pressure is not by itself a
measured performance improvement. Compact results: `results/
qev-pro-review-2026-10-21/P4.json`; raw `P4-quality`, `P4-performance`,
`P4-halfdots-quality` and `P4-halfdots-performance`.


## P7: shared-KV four-head FP32 WASM attention

The new isolated SIMD translation unit computes four query heads sharing one KV
head. Query transposition, QK lanes and PV channel vectors reuse KV data while
preserving each output's accumulation order, softmax and gate arithmetic.
It requires head width 256, four query heads per KV head and contiguous FP32 KV.
Other layouts, native execution and FP16 retain their existing kernels.
The accepted minimum batch size is eight tokens; zero disables the experiment.

The rotating three-candidate grid gives control 129.07/132.24/1072.66 ms,
minimum eight 126.82/130.00/982.14 ms and minimum 32
127.12/131.11/986.64 ms. A separate fresh-browser A/B confirms control
130.85/131.02/1066.85 versus 128.26/128.75/981.77 ms (compute latency,
three measured runs and one warmup). Adopt minimum eight: approximately 8%
lower long-request latency, with no short-request regression.

Kernel tests cover token counts 1/3/7/8/31/32/33, positions 0/37/1024,
whole and split group ranges, extreme finite inputs and output guards.
Node and actual Chromium pass exact FP32 bit parity against the scalar kernel.
All 17 native tests and the full 24-question WASM logit/answer gate pass.
The probability difference to native remains at most 0.004 on this small panel.
An instrumented diagnostic retains separate worker sums for packing, QK,
softmax, PV and gating; these sums are not wall latency or additive stage times.
Raw evidence: P7-quality, P7-performance, P7-fresh and P7-diagnostic.
Compact samples: results/qev-pro-review-2026-10-21/P7.json.


## P8: selected option output projection

The Decision C ABI now requests only the needed option token logits. The engine
prepares the normalized final hidden vector once and evaluates unique eight-row
DOT4 tiles, restoring the caller's token order and duplicates. The restricted
API has a separate callback, requires exactly one forced output slot and rejects
unsupported heads or invalid token IDs. Full-vocabulary callbacks and general
sampling keep their existing contract. Complete and partial prefix hits are
covered. `selected_option_head:0` restores the full projection for comparison.

Three measured runs plus one warmup give these median compute latencies (ms):

| Engine | Full head: short / Snake / long | Selected head: short / Snake / long |
|---|---:|---:|
| WASM | 129.01 / 129.15 / 980.09 | 125.16 / 126.65 / 980.92 |
| Native | 38.39 / 37.03 / 239.69 | 33.98 / 34.62 / 236.60 |

The benefit is concentrated in short requests: about 3% for the first WASM
request, and 6-11% for the native short requests in this series. No measurable
long-request WASM benefit is established. Adopt the selected Decision readout;
this does not change HTTP server code or claim a server-level speed gain.

All 24 browser questions retain exact option bits and answers. Native full and
selected outputs match bitwise on all three measured prompts. A real-model API
test covers eight prompt lengths, four prefix lengths, vocabulary endpoints,
unsorted/duplicate IDs, seven rejected contracts and recovery; all 17 native
CTest checks pass. Weights and probability calibration are unchanged.
Raw P8-quality/P8-performance, compact results/P8.json under the study folders.
