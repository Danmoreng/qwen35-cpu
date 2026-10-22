# Published evidence

These directories retain the compact measurements and provenance used by the
current documentation. Dates identify frozen measurement sets, not current
build versions. Existing evidence files are preserved byte for byte.

| Directories | Used by |
| --- | --- |
| `native-prefill-reuse-2026-10-22` | Current native README/model-card speed, paired baseline and quality parity; [methodology](../benchmarks/native-prefill.md) |
| `cpu-refresh-2026-10-22` | Comparator README/model-card speed, all quality scores and preceding native baseline; [methodology](../benchmarks/cpu.md) |
| `current-cpu-2026-09-07`, `ik-batch16-2026-09-07` | September CPU speed baseline and earlier B16 validation |
| `readme-bc256-2026-09-07`, `ik-linux-2026-09-07` | September quality baseline and comparator provenance |
| `linux-i7-8750h-2026-09-08` | [Earlier second-CPU comparison](../benchmarks/cpu.md#earlier-second-cpu-comparison) |
| `g32-large-calibration-2026-09-07` | [Standard calibration](../benchmarks/calibration.md) |
| `qev-calibration-2026-10-21`, `qev-api-2026-10-21` | [Decision quality](../benchmarks/decision-quality.md), model-package provenance |
| `qev-http-selected-2026-10-22` | [Decision HTTP latency](../benchmarks/decision-http.md) |
| `qev-browser-cpu-2026-10-21`, `qev-wasm-local-2026-10-21` | [Browser comparison](../benchmarks/browser.md) and initial WASM baseline |
| `qev-wasm-optimization-2026-10-21`, `qev-wasm-kernels-2026-10-21` | Retained WASM FMA/kernel measurements |
| `qev-space-2026-10-22` | [Public Space deployment](../browser/hosting.md) |

For new measurements, keep raw profiles, captures and logs under ignored
`benchmarks/`. Publish compact aggregates with commands, binary/checkpoint
hashes, CPU affinity, upstream revisions and raw-evidence locations. Use the
sequential benchmark runner with one warmup and three measured runs. Do not
replace old measurements with numbers from a different protocol.

Superseded raw result trees remain in [Git history](../history.md). Runtime
release archives exclude this evidence directory; use the matching source
revision to inspect it.
