# Native prefill weight reuse

October 22, 2026, Ryzen 9 9955HX3D. This experiment follows the
[IK converter analysis](cpu.md#why-ik-iq4_ks_r4-prefill-changed).
The selected [native revision `f928eec37ab3`](https://github.com/Danmoreng/qwen35-cpu/commit/f928eec37ab34739e03d4fa3d37fe34d44ff7229)
reuses unpacked Q4 weights across sixteen tokens
instead of eight for matrix calls with at least 64 activation rows. Smaller
calls retain the existing eight-token kernel and tails. The checkpoint layout,
32-value scales, integer dot products and floating-point accumulation order are
unchanged. No temporary weight buffer is added to the selected implementation.

## Candidates and selection

The baseline engine code is the same as the preceding CPU refresh (repository
revision `68156de3cadd4e5f373b0d9729529ec299596481`; its retained baseline binaries
were built at `6d38cca33e548aaeda2dd27b21fef689d47c7d79`). The experiment considered:

- Temporary expanded weight buffers containing 16, 32 or 128 quantization groups
  for sixteen output rows, reused across token tiles. The middle size performed
  best in the first screen: about 3% faster at P512 and 2.7% at P4096. Larger
  scratch storage did not improve performance further.
- Sixteen-token register tiles without a temporary buffer. They improved
  prefill, but using them for a sixteen-request decode batch regressed that
  workload. The final dispatch retains the old path below 64 activation rows.
- Combining the middle-size buffer with sixteen-token tiles. This did not
  improve the P512 result over the simpler register-only candidate.

The first three series are selection evidence, not the published final speed
claim. A separate paired confirmation series measures the exact selected build
against the unchanged baseline, including short prompts, B16/B32 decode and
native pure-Q4_0 execution. Every series uses one warmup and three measured runs
with alternating case order through `scripts/benchmark-inference-seq.ps1`.

## Correctness

The DOT4 regression suite includes 60/64/68 and 124/128/132-token cases around
the dispatch threshold and tile tails, padded output strides, eight-row tails,
negative/extreme quantized values and multiple reduction lengths. SIMD outputs
must match the existing reference bit for bit where AVX2 is available.

All 17 local CTest tests pass, including 10,080 DOT4 regression cases. Full-model
batch checks for both the standard H128 artifact and native pure-Q4_0 cover
B4/B8/B12/B16/B20/B32, with uneven worker
partitions and prompts spanning chunk boundaries. Every batched full-vocabulary
logit matches its single-request reference bit for bit.

All 16 original quality windows were rerun with the selected native CLI:
8,192 scored positions and 2,034,237,440 float logits. Every capture matches its
baseline SHA-256 exactly, including target IDs and the full vocabulary. Thus the
published H128 PPL **15.79649** and KL **0.060100** are unchanged on this corpus.

Separate 4,096-token prompts followed by 16 scored positions also match bit for
bit for H128 and native pure Q4_0: 3,973,120 float logits per model. These checks
establish equality for the tested workloads, not for every possible input.

## Final paired measurement

| H128 workload | Baseline tokens/s | Selected tokens/s | Change |
| --- | ---: | ---: | ---: |
| B1 prefill 32 | 1,641.61 | 1,652.34 | +0.65% |
| B1 prefill 64 | 1,991.00 | 2,104.23 | +5.69% |
| B1 prefill 128 | 2,279.22 | 2,422.12 | +6.27% |
| B1 prefill 512 | 2,918.72 | 3,052.57 | +4.59% |
| B1 prefill 4,096 | 2,589.49 | 2,683.16 | +3.62% |
| Decode B1, P512/N128 | 123.04 | 123.03 | -0.01% |
| Decode B16, P512/N128 | 639.66 | 639.63 | <0.01% |
| Decode B32, P512/N128 | 730.41 | 733.96 | +0.49% |

Native pure-Q4_0 uses the same DOT4 kernel:

| Q4_0 workload | Baseline tokens/s | Selected tokens/s | Change |
| --- | ---: | ---: | ---: |
| B1 prefill 512 | 2,768.95 | 2,848.82 | +2.88% |
| B1 prefill 4,096 | 2,432.06 | 2,509.78 | +3.20% |
| Decode B1, P512/N128 | 124.18 | 123.41 | -0.62% |

H128 P512 model-forward latency falls from 175.42 to 167.73 ms.
The unchanged P32 and decode paths show small measurement variation; no
decode improvement is attributed to this change. In particular, the earlier
full-engine comparison measured native B1 decode at 118.33 tokens/s, whereas
this paired baseline reaches 123.04 with the same executable. Absolute
differences between those series are not an optimization result.

The README updates the native row from this confirmation series and retains
the earlier comparator measurements with an explicit source note. Use the
paired baseline above to quantify this change; small cross-engine differences
across the two series should not be interpreted as a definitive ranking.

## Conditions and evidence

Eight threads, physical V-Cache cores 0–7 (`0xff`), GCC 16.2.1 Release,
AVX-512/VNNI, FP16 KV, full-vocabulary logits and fixed prompts/outputs. The laptop
profile remains `max-power`; Linux uses `amd-pstate-epp` active mode, governor
`powersave`, EPP `balance_performance`. No CPU controls were changed. Codex is
minimized and XFCE compositing disabled at 2560×1600 / 240 Hz during speed runs.
No compilation, model validation or other benchmark runs concurrently.

Prefill reports native model-forward time including the final vocabulary head;
loading, tokenization and prefix-cache credit are excluded. Decode scores use
P512/N128 and 127 actual decode forwards per request. B16/B32 report aggregate
throughput, not single-request latency. P4096/N2 decode is not a headline score.
The short-prompt checks use the same forced 128 outputs.

[Aggregates](../results/native-prefill-reuse-2026-10-22/performance.csv),
[all samples](../results/native-prefill-reuse-2026-10-22/samples.csv),
[quality parity](../results/native-prefill-reuse-2026-10-22/quality-parity.json)
and the [manifest](../results/native-prefill-reuse-2026-10-22/manifest.json)
retain the measurement contract, source identities, binary/checkpoint hashes,
exact final matrix, environment and raw-file identities. Prototype source
snapshots, patches, build logs, raw profiles and command lines remain under
ignored `benchmarks/native-prefill-reuse-2026-10-22/`.

Reproduce with the source revisions in the manifest, Release builds and the
published final matrix after adapting local paths. Use a new output directory:

```sh
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix /path/to/matrix.json -OutputDir benchmarks/new-native-prefill-run \
  -Affinity 255 -WarmupRuns 1 -Runs 3
```

The measured gain applies to this CPU and workload. IK's roughly 79% converter
speedup is not reproduced here: native Q4 unpacking was already simple, used VNNI
and reused weights across eight tokens. Doubling that reuse saves some work;
temporary conversion and extra cache traffic are not automatically beneficial.
