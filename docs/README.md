# Documentation

Start with the [project README](../README.md) for downloads, native builds and
headline benchmarks. The guides below describe the supported engine and tools.

## Use and integrate

- [Fine-tune import and task evaluation](finetunes.md) covers HF snapshots, LoRA adapters and single-token classifiers.
- [HTTP server](server.md): Responses, offline batches, CPU tuning, chat completions and decision requests.
- [C++ engine API](cpu-engine-api.md): batching, scheduling and prefix reuse.
- [Quantization](quantization.md): H128/Q4-G32-DOT4 format and offline calibration.
- [GGUF loading](q4km-native.md): native pure Q4_0 and experimental Q4_K_M.

## Browser playground

- [Build and validate WebAssembly](browser/build.md).
- [Interactive decision demos](browser/playground.md).
- [Package and publish the Hugging Face Space](browser/hosting.md).

## Benchmarks and model cards

- [qev CPU serving profiles and AVX2 kernel trials](benchmarks/decision-tuning-9955hx3d.md) covers short decisions, concurrent routing and two DOT4 prototypes.
- [Current laptop throughput and qev decision latency](benchmarks/laptop-refresh-9955hx3d.md) covers the three base-model sizes and the before/after decision tests.
- [Core i7-8750H vs Ryzen 9 9955HX3D](benchmarks/i7-8750h-vs-ryzen-9955hx3d.md) compares the README workloads on six Intel cores and eight Ryzen V-Cache cores with identical checkpoints.
- [Qev decision servers on the i7](benchmarks/qev-i7-server-comparison.md) compares native H128/Q4-G32-DOT4, llama.cpp and ik_llama.cpp over System One HTTP.
- [Ryzen and i7 Qev server comparison](benchmarks/qev-ryzen-server-comparison.md) pairs HTTP latency, throughput and per-ticket choices across the same three servers and 66 inputs.
- [Qev Q4 quality on Benchmark Heaven](benchmarks/qev-heaven-i7.md) compares six checkpoint and server combinations on 231 labeled tasks.
- [Qev precision comparison on the i7](benchmarks/qev-heaven-precision-i7.md) compares native H128 with llama.cpp Q4_K_M, Q8_0 and BF16 for decision quality and speed.
- [0.8B on the Ryzen 9 9955HX3D](benchmarks/ryzen-9955hx3d.md) compares the native engine with llama.cpp and ik_llama.cpp.
- [2B and 4B on the same laptop](benchmarks/ryzen-9955hx3d-large.md) compares the three engines with matched tensor sizes, single requests and batches of 16.
- [Hadamard cost and DeltaNet prefill on the same laptop](benchmarks/prefill-9955hx3d.md) profiles the three model sizes and compares the optimized kernel with ik_llama.cpp. The [Q4 matrix kernel follow-up](benchmarks/q4-prefill-9955hx3d.md) isolates the next change.
- [October CPU comparison](benchmarks/cpu.md), [IK prefill analysis](benchmarks/cpu.md#why-ik-iq4_ks_r4-prefill-changed)
  and [IK batch-16 methodology](benchmarks/ik-batch16.md).
- [Fine-tune serving](benchmarks/finetune-serving.md) compares Prism and QEV against BF16 and tests shared decision prefill.
- [Batches API throughput](benchmarks/batch-api.md) compares serial and concurrent processing of 16 German text corrections.
- [Speculative text decoding](benchmarks/text-drafting.md) compares prompt and input-source lookup on single-stream editing and generation.
- [Adaptive drafting on Ryzen](benchmarks/adaptive-drafting-ryzen.md) measures 2B text drafting and the low-acceptance fallback.
- [AVX-512 DeltaNet decode](benchmarks/deltanet-xeon.md) measures single-stream generation on a Xeon VM.
- [Parallel batch decode on Zen 4](benchmarks/zen4-batch-decode.md) measures the DeltaNet dispatch change with 0.8B and 2B.
- [Native prefill optimization](benchmarks/native-prefill.md): paired measurements and bitwise quality checks.
- [Qwen3.5-2B on the i7-8750H](benchmarks/qwen35-2b-i7.md) covers GTX 1070 Max-Q teacher collection, quantization quality and matched 0.8B/2B CPU measurements against ik_llama.cpp.
- [Native 0.8B and 2B text tasks](benchmarks/qwen35-text-panel.md) measures German correction, Python edits and JSON extraction on a Ryzen 5 7600.
- [Native Qwen3.5-4B on Ryzen](benchmarks/qwen35-4b-ryzen.md) compares 2B and 4B throughput, memory and task results and validates 4B against BF16.
- [Native 4B MTP on Ryzen](benchmarks/qwen35-4b-mtp.md) compares BF16, Q8_0 and H128 sidecars, draft depths and lookup drafting.
- [4B MTP depth for coding and tool calls](benchmarks/qwen35-4b-mtp-coding.md) tests one through four proposals with shared verification on the native API.
- [4B Pi Coding Agent on Ryzen](benchmarks/qwen35-4b-pi-agent.md) compares native and ik_llama.cpp agent sessions, cache reuse, quantization fidelity and [matched 4K–12K decode](benchmarks/qwen35-4b-pi-agent.md#matched-long-context-mtp-decode).
- [AVX2 prefill on the i7-8750H](benchmarks/avx2-prefill-i7.md) measures query tile selection for long prompts.
- [Four-token AVX2 DOT4 on the i7-8750H](benchmarks/dot4-pair4-i7.md) measures the paired-panel patch and checks full-vocabulary logit parity.
- [AVX2 prefill follow-up on the i7-8750H](benchmarks/avx2-prefill-followup-i7.md) compares paired DOT4 scheduling, DeltaNet state reuse and attention PV tiles.
- [AVX2 cloud VM tuning](benchmarks/avx2-cloud.md): prompt processing, decode experiments and bitwise checks on an EPYC VM.
- [Standard model calibration](benchmarks/calibration.md).
- [QEV prompt order on 50 German tickets](benchmarks/qev-ticket-order.md) compares placing the ticket before or after the question and all options.
- [Decision quality](benchmarks/decision-quality.md), [HTTP latency](benchmarks/decision-http.md) and [exact-prefix caching with fixed rubrics](benchmarks/qev-exact-prefix.md).
- [Four decision models on GPU](benchmarks/decision-models-gpu.md) compares public JevBench accuracy, probability consistency and When2Call tool routing before CPU integration.
- [Liquid and Qev on full JevBench, German routing and CPU](benchmarks/decision-model-followup.md) compares Qev encodings, Liquid Q4 versus Q8 and resident llama.cpp latency.
- [BA Research and Labqoat Qev at 0.8B](benchmarks/qev-08-comparison.md) compares both decision models at BF16, the Labqoat Q8 configuration and the effect of BA's released context limit.
- [qev on JevBench v1](benchmarks/jevbench-v1.md): Q4/Q8/BF16 accuracy and Ryzen CPU latency, paired precision audit, training-provenance limits, published Gemma 4/H100 comparisons and an English chart.
- [qev on NVIDIA When2Call](benchmarks/when2call.md): complete MCQ test accuracy across Q4/Q8/BF16 and CPU/Vulkan backend agreement; [published-model comparisons](benchmarks/when2call-comparison.md) with English charts.
- [qev AVX2 tuning](benchmarks/qev-avx2-cloud.md): CPU measurements and attention boundaries for larger chunks.
- [Browser comparison](benchmarks/browser.md), [WASM FMA](benchmarks/wasm-fma.md)
  and [WASM kernels](benchmarks/wasm-kernels.md).
- [Base model card](model-card.md) and [decision model card](qev-model-card.md).
- [Evidence inventory](results/README.md): retained measurements and provenance.

Reports describe their recorded checkpoints, source revisions and measurement
dates. They are not measurements of every subsequent commit.

## Contribute and release

- [Liquid d1-2B CPU port feasibility](liquid-d1-port.md) covers the text architecture, the first native mixer test and a proposed implementation scope.
- [Julia-2 text port feasibility](julia-2-port.md) covers the native H128 trunk probe, the Julia decision head and the serving changes.
- [Validation](validation.md): automated checks and optional model/browser suites.
- [Tooling](../scripts/README.md): build, calibration, evaluation and publishing scripts.
- [Publication](publishing.md): model packages, binary releases and CI requirements.
- [Historical work](history.md): immutable access to superseded studies and raw evidence.
