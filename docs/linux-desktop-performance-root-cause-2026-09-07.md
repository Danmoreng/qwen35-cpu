# Linux/Ryzen decode check (2026-09-07)

This is the earlier native-only desktop experiment. The README now uses the
[subsequent complete five-candidate series](readme-comparison-linux-ryzen-2026-09-07.md),
including a new native measurement under the same desktop policy. The values
below remain the historical results of this separate experiment.

The unchanged Linux native engine reaches **127.48 tok/s single-request decode**
and **638.53 tok/s aggregate batch-16 decode** with the Codex window minimized
and XFCE compositing disabled. The display remains at 2560×1600 / 240 Hz.
These are medians of three measured runs after one discarded warmup.

## Measurement contract

- AMD Ryzen 9 9955HX3D; Arch Linux, kernel 7.2.2-arch1-1; GCC 16.2.1 Release.
- Eight threads on physical V-Cache cores 0–7 (`0xff`), AVX-512/VNNI dispatch.
- H128/Q4-G32-DOT4 B+C 256, exact published P512/N128 token fixtures,
  full-vocabulary logits, FP16 KV, maximum context 16384.
- 127 actual decode forwards per request. B16 is aggregate throughput across
  sixteen private requests; it is not compared with a single-request baseline.
- Sequential measurements through `scripts/benchmark-inference-seq.ps1`, one
  warmup plus three measured runs, reversing case order on alternate passes.
- Desktop changes and settling delays precede executable startup. Results use
  the engine's internal decode time, excluding model load, prefill and the wrapper.
- CPU affinity and maximum-performance power policy remain fixed across cases.
- No production-code changes or rebuild were needed. No operation instrumentation
  was enabled. Model quality was not rerun; forced tokens fix the workload rather
  than establish generation quality.

## Results

| Workload / desktop state | Median tok/s | Measured range |
|---|---:|---:|
| B1, 240 Hz, Codex visible, compositor on | 110.71 | 110.39–110.77 |
| B1, 240 Hz, Codex minimized, compositor on | 126.42 | 126.13–126.69 |
| **B1, 240 Hz, Codex minimized, compositor off** | **127.48** | **127.36–127.59** |
| B1, 60 Hz, Codex minimized, compositor off | 126.25 | 125.85–126.41 |
| B16 aggregate, 240 Hz, Codex visible, compositor on | 544.00 | 524.82–546.01 |
| **B16 aggregate, 240 Hz, Codex minimized, compositor off** | **638.53** | **637.43–639.15** |

The README uses the matching minimized/compositor-off conditions for B1 and B16.
This is a native-engine decode check, not a new five-candidate comparison or a
fresh matched Windows/Linux A/B series. The archived Windows results are
120.05 tok/s B1 and 619.16 tok/s B16.

## Desktop interference

Initial Linux measurements were around 87 tok/s. A display-rate experiment
reproduced a slower visible-window regime: two measured 240 Hz runs reached
81.47 and 81.91 tok/s, while a third reached 110.19. At 60 Hz the three runs
clustered at 114.08–114.62 tok/s. The unchanged standalone Q4 streaming probe
improved under the same desktop intervention. All individual values, including
this variability, are retained in the [summary CSV](results/linux-desktop-2026-09-07/summary.csv).

The second series above separates window visibility from refresh rate. Minimizing
Codex alone restores Windows-level B1 performance at 240 Hz with compositing
still enabled. Per-process DRM counters also show rendering activity falling
substantially when the window is minimized. Moving GUI CPU threads to another
CCD or disabling only the compositor does not isolate application GPU rendering.

The graphical environment is a demonstrated causal contributor, not evidence of
an inherently slower Linux CPU build. Shared-memory contention is the leading
explanation for the streaming effect; independent DRAM traffic and the precise
driver/fabric/power mechanism were not measured. The transition between the two
visible-window speed regimes remains unresolved, so no fixed 240 Hz penalty is
claimed. GPU busy endpoint readings alone are not a reliable attribution.

For comparisons, minimize the Codex window throughout the benchmark series and
record consistent desktop conditions for every candidate. Original settings were
restored after testing: 240 Hz, compositor enabled, Codex window active.

## Provenance

The [manifest](results/linux-desktop-2026-09-07/manifest.json) records revisions,
executable/checkpoint/tool hashes and the local raw-archive hash. Full profiles,
commands, matrices, source hashes, desktop observations and case wrappers remain
in the ignored `benchmarks/astra-investigation-20260907/` directory; raw archives,
weights, build outputs and diagnostic source additions are not published.

The manifest supplements the runner metadata with hashes of the actual inference
and probe executables, because the runner launches a Python desktop wrapper.
All recorded hashes were rechecked after the series, and all 40 warmup/measured
profiles were validated for workload identity. The exact wrappers are local,
session-specific diagnostics: verify tool paths, CPU topology, X11 window ID and
original display settings before reusing them with the sequential runner.
