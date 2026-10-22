# Documentation

Start with the [project README](../README.md) for downloads, native builds and
headline benchmarks. The guides below describe the supported engine and tools.

## Use and integrate

- [HTTP server](server.md): text completions, decision requests, sampling and limits.
- [C++ engine API](cpu-engine-api.md): batching, scheduling and prefix reuse.
- [Quantization](quantization.md): H128/Q4-G32-DOT4 format and offline calibration.
- [GGUF loading](q4km-native.md): native pure Q4_0 and experimental Q4_K_M.

## Browser playground

- [Build and validate WebAssembly](browser/build.md).
- [Interactive decision demos](browser/playground.md).
- [Package and publish the Hugging Face Space](browser/hosting.md).

## Benchmarks and model cards

- [Current CPU comparison](benchmarks/cpu.md), [IK prefill analysis](benchmarks/cpu.md#why-ik-iq4_ks_r4-prefill-changed)
  and [IK batch-16 methodology](benchmarks/ik-batch16.md).
- [Native prefill optimization](benchmarks/native-prefill.md): paired measurements and bitwise quality checks.
- [Standard model calibration](benchmarks/calibration.md).
- [Decision quality](benchmarks/decision-quality.md) and [HTTP latency](benchmarks/decision-http.md).
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
