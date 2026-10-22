# Tooling

Run scripts from the repository root. Build products, model downloads and raw
measurements belong in ignored local directories. Use each command's `--help`
(or PowerShell parameters) for its input requirements.

| Workflow | Entry points |
| --- | --- |
| Native build | `build.ps1`, or CMake on Linux |
| Browser build and local serving | `build-wasm.sh`, `serve-wasm.py` |
| Model download | `download-model.sh`, `download-model.ps1`, `download-qev-model.py` |
| Sequential measurements | `benchmark-inference-seq.ps1`; HTTP, WASM and browser benchmark drivers run through it |
| CPU comparisons | `build-llama-comparison.ps1`, `prepare-comparison-data.py`, `prepare-comparison-matrix.py`, `record-comparison-inputs.py`, `validate-llama-batching.py` |
| Quality scoring | `evaluate-comparison-quality.py`, `evaluate-native-gguf.py`, `compare-logit-dumps.py`, `compare-quality-pair.py`, `quality-report.py` |
| Calibration | `prepare-diverse-calibration.py`, `collect-resident-calibration.py`, `fit-dot4-quant.py`, `fit-block-covariance.py`, `run-large-calibration-study.py` |
| Decision-model calibration | `prepare-qev-calibration.py`, `collect-qev-calibration.py`, `qev-study.py`, `split-qev-study.py` |
| Model packaging | `prepare-hf-package.py`, `prepare-qev-hf-package.py`, `upload-hf-package.py` |
| Binary packaging | `package-release.py`, `verify-release.py` |
| Space packaging | `prepare-qev-space.py`, `serve-space-preview.py`, `publish-qev-space.py` |

Audit, fitting, capture-compaction and summary helpers support these workflows.
Offline teacher collection can use a separate CUDA tool; the inference engine
remains CPU-only. Calibration study drivers retain their documented local layout
and must be configured with the matching inputs before use.

See [quantization](../docs/quantization.md), [benchmark reports](../docs/README.md#benchmarks-and-model-cards),
[validation](../docs/validation.md) and [publication](../docs/publishing.md).
One-off generators for superseded head/Q8/README experiments are available in
[the historical snapshot](../docs/history.md), together with their reports.
