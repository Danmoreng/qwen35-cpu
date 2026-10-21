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
157.40/157.62/1248.02 ms versus 155.93/156.00/1226.43 ms for 1 (see exact JSON).
This small difference does not establish the anticipated larger gain; retain
128 as the product default. The explicit tuning control remains available.
The 24-question candidate suite passed exact WASM bits and native probability
gates. DeltaNet tests now require exact WASM state/output equality and cover
1/3/7/8/16/31/32/33/128-token batches and split head ranges.
