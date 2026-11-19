# Published evidence

These directories retain the compact measurements and provenance used by the
current documentation. Dates identify frozen measurement sets, not current
build versions. Existing evidence files are preserved byte for byte.

| Directories | Used by |
| --- | --- |
| `qwen35-2b-adaptive-draft-2026-11-19` | [Adaptive drafting on Ryzen](../benchmarks/adaptive-drafting-ryzen.md), paired 2B latency, acceptance, output parity and stress fixtures |
| `qwen35-4b-ryzen-2026-11-19` | [Native Qwen3.5-4B on Ryzen](../benchmarks/qwen35-4b-ryzen.md), throughput, memory, task panel and BF16 reference |
| `qwen35-text-panel-2026-11-19` | [Native 0.8B and 2B text tasks](../benchmarks/qwen35-text-panel.md), frozen tasks, outputs, timings and provenance |
| `zen4-batch-decode-2026-11-19` | [Parallel batch decode on Zen 4](../benchmarks/zen4-batch-decode.md), paired 0.8B and 2B throughput and logit parity |
| `avx2-prefill-i7-2026-11-18` | [Long-context AVX2 prefill](../benchmarks/avx2-prefill-i7.md), i7-8750H timing samples and logit parity |
| `when2call-comparison-2026-11-10` | [Published When2Call comparisons](../benchmarks/when2call-comparison.md), AA cohort re-scoring, independently re-counted Jev/Snap outputs and English charts |
| `when2call-2026-11-10` | [qev on NVIDIA When2Call](../benchmarks/when2call.md), complete-test candidate-choice accuracy, CPU/Vulkan agreement and frozen provenance |
| `jevbench-v1-2026-11-10` | [qev on JevBench v1](../benchmarks/jevbench-v1.md), CPU precision/latency comparison, paired accuracy and provenance audit, English chart and published Gemma 4/H100 baselines |
| `batch-api-2026-11-18` | [Native Batches API](../benchmarks/batch-api.md), end-to-end throughput, CPU calibration and output parity |
| `text-drafting-2026-11-17` | [Text drafting](../benchmarks/text-drafting.md), source documents, acceptance, output parity and latency |
| `deltanet-xeon-2026-11-17` | [AVX-512 DeltaNet decode](../benchmarks/deltanet-xeon.md), paired timings and numerical validation |
| `qev-avx2-cloud-2026-10-27` | [qev AVX2 tuning](../benchmarks/qev-avx2-cloud.md), large-chunk parity and paired timings |
| `avx2-cloud-2026-10-27` | [AVX2 cloud VM tuning](../benchmarks/avx2-cloud.md), paired measurements and logit parity |
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
