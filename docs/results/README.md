# Published evidence

These directories retain the compact measurements and provenance used by the
current documentation. Dates identify frozen measurement sets, not current
build versions. Existing evidence files are preserved byte for byte.

| Directories | Used by |
| --- | --- |
| `qev-ryzen-servers-2026-11-22` | [Ryzen and i7 Qev server comparison](../benchmarks/qev-ryzen-server-comparison.md), paired latencies, throughput, answer distributions and provenance |
| `qev-julia-2-probe-2026-11-21` | [Julia-2 text port feasibility](../julia-2-port.md), native H128 decision parity, probability drift and provenance |
| `qev-heaven-precision-i7-2026-11-21` | [Qev precision comparison on the i7](../benchmarks/qev-heaven-precision-i7.md), 924 scored predictions, repeated timing samples and provenance |
| `qev-heaven-i7-2026-11-21` | [Qev Q4 quality on Benchmark Heaven](../benchmarks/qev-heaven-i7.md), 1,386 scored predictions, paired comparisons and provenance |
| `qev-i7-servers-2026-11-21` | [Qev decision servers on the i7](../benchmarks/qev-i7-server-comparison.md), matched-size Q4 speed, Q8_0 measurements, response agreement and provenance |
| `i7-ryzen-comparison-2026-11-21` | [Core i7-8750H vs Ryzen 9 9955HX3D](../benchmarks/i7-8750h-vs-ryzen-9955hx3d.md), six-core throughput, qev HTTP latency and response parity, and verified checkpoint provenance |
| `i7-avx2-followup-2026-11-20` | [AVX2 prefill follow-up on the i7-8750H](../benchmarks/avx2-prefill-followup-i7.md), paired and instrumented timings, build provenance and logit parity |
| `i7-dot4-pair4-2026-11-20` | [Four-token AVX2 DOT4 on the i7-8750H](../benchmarks/dot4-pair4-i7.md), sequential samples, build provenance and logit parity |
| `decision-tuning-2026-11-20` | [qev latency, concurrent serving and AVX2 prototypes](../benchmarks/decision-tuning-9955hx3d.md) |
| `laptop-refresh-2026-11-20` | [Current laptop benchmarks](../benchmarks/laptop-refresh-9955hx3d.md), fixed-token throughput, qev HTTP decisions and the README chart |
| `q4-prefill-2026-11-20` | [Ryzen Q4 matrix kernel](../benchmarks/q4-prefill-9955hx3d.md), paired prefill rates, Q4 operation timings and full-vocabulary logit parity |
| `prefill-profile-2026-11-20` | [Ryzen prefill study](../benchmarks/prefill-9955hx3d.md), operation timings, runtime comparisons and full-vocabulary logit parity |
| `ryzen-large-2026-11-20` | [2B and 4B on the Ryzen 9 9955HX3D](../benchmarks/ryzen-9955hx3d-large.md), matched tensor sizes, per-run CPU throughput and reproduction matrix |
| `decision-gpu-2026-11-20` | [GPU decision-model comparison](../benchmarks/decision-models-gpu.md), predictions, quality scores and checkpoint hashes |
| `decision-followup-2026-11-20` | [Decision-model follow-up](../benchmarks/decision-model-followup.md), full JevBench scores, frozen German inputs, paired quantization results and CPU latency |
| `qev-08-comparison-2026-11-20` | [BA Research and Labqoat Qev at 0.8B](../benchmarks/qev-08-comparison.md), paired predictions, context coverage and checkpoint provenance |
| `readme-2026-11-19` | [Ryzen 9 9955HX3D comparison](../benchmarks/ryzen-9955hx3d.md), fixed-token throughput, inputs and binary hashes |
| `qwen35-4b-agent-2026-11-19` | [4B Pi Coding Agent on Ryzen](../benchmarks/qwen35-4b-pi-agent.md), per-request phase rates, browser scores, cache controls and BF16 quantization comparison |
| `qwen35-4b-mtp-coding-2026-11-19` | [4B MTP depth for coding and tool calls](../benchmarks/qwen35-4b-mtp-coding.md), source fixtures, HTTP traces and shared-target timing samples |
| `qwen35-4b-mtp-2026-11-19` | [Native 4B MTP on Ryzen](../benchmarks/qwen35-4b-mtp.md), sidecar sizes, forward errors, output parity and sequential timings |
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
| `native-prefill-reuse-2026-10-22` | October native speed, paired baseline and quality parity; [methodology](../benchmarks/native-prefill.md) |
| `cpu-refresh-2026-10-22` | October comparator speed, WikiText-2 quality scores and native baseline; [methodology](../benchmarks/cpu.md) |
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
