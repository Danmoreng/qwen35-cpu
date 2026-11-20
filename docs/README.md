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

- [0.8B on the Ryzen 9 9955HX3D](benchmarks/ryzen-9955hx3d.md) compares the native engine with llama.cpp and ik_llama.cpp.
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
- [AVX2 cloud VM tuning](benchmarks/avx2-cloud.md): prompt processing, decode experiments and bitwise checks on an EPYC VM.
- [Standard model calibration](benchmarks/calibration.md).
- [Decision quality](benchmarks/decision-quality.md) and [HTTP latency](benchmarks/decision-http.md).
- [Four decision models on GPU](benchmarks/decision-models-gpu.md) compares public JevBench accuracy, probability consistency and When2Call tool routing before CPU integration.
- [Liquid and Qev on full JevBench, German routing and CPU](benchmarks/decision-model-followup.md) compares Qev encodings, Liquid Q4 versus Q8 and resident llama.cpp latency.
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

- [Validation](validation.md): automated checks and optional model/browser suites.
- [Tooling](../scripts/README.md): build, calibration, evaluation and publishing scripts.
- [Publication](publishing.md): model packages, binary releases and CI requirements.
- [Historical work](history.md): immutable access to superseded studies and raw evidence.
