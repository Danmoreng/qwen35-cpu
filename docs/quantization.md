# Standard quantization recipe

The standard [Hugging Face download](https://huggingface.co/danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4/tree/59f422b2d410fdaf4a9efc71ff12f278abc2a5d1) is **activation-weighted MSE16
H128/Q4-G32-DOT4**, also called calibrated MSE16. The file format is unchanged:
`.q35h`, signed H128 transforms, Q4 groups of 32 with FP16 scales, and the same
CPU DOT4 byte layout. No new loader or kernel is required to interpret its
weights. The format identifier describes storage; `quantization.json` describes
how the values were fitted. A better recipe does not require a new format name.

## What changed

Legacy conversion chooses a scale from the largest absolute weight in a group.
MSE16 instead searches both signs of the scale, uses all 16 signed codes
`[-8,7]`, refines candidate scales by least squares, and scores reconstruction
using the FP16 scale that will actually be stored. It keeps the legacy candidate
when no candidate improves the fitting objective. Code ties round away from
zero, and equal objective scores retain the first candidate for determinism.

The calibrated recipe weights squared error by the second moment of real BF16
teacher inputs. Those inputs are captured after the relevant normalization or
nonlinearity and transformed with exactly the weight transform's seed and H128
basis. The tied output head uses identity. Fitting also retains the unweighted
MSE16 candidate. This is diagonal activation weighting, not full-covariance GPTQ.
Improvement of this local objective alone is not proof of better perplexity;
the published candidate was separately evaluated.

Only offline conversion does more work. Inference reads the same number of
weight bytes and performs the same operations, explaining why the quality
improvement does not incur an inference-speed cost. The measured speed varies
slightly between runs. Both new artifacts have 424,934,656 tensor bytes and
424,964,864 total file bytes.

## Defaults and reproduction

There are two explicit defaults:

- **Download:** the calibrated artifact, SHA-256
  `8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d`.
- **Source conversion:** unweighted `mse16` when no calibration directory is
  supplied. Calibration is never silently inferred from a local directory.

```powershell
# Default conversion, no calibration corpus required
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-mse16.q35h

# Calibrated conversion after collecting and fitting teacher inputs
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-calibrated-mse16.q35h --importance-dir benchmarks/plan-calibration-fit

# Exact historical recipe, explicitly selected
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-legacy.q35h --quantizer legacy-absmax15
```

The packer writes quantization and calibration provenance sidecars. Publication
embeds those records in `quantization.json`; runtime does not need calibration
inputs. The versioned Q35CAL1 records validate columns, transform seed, basis,
sample count, finite nonnegative weights and exact file length.

See [the measured report](implementation-plan-results-2026-09-06.md) for
calibration collection, BF16 source hashes, raw results and limitations. Its
statements that Legacy was the default describe the initial experiment, before
the new recipe was promoted to the standard download and converter default.
The four-document English calibration pilot is not general quality certification.

The old public artifact remains reproducible by pinning Hugging Face revision
`cc7df08da7ef7ac15db62e80b4eda85e19a143da`. Updated download examples pin the
new revision rather than mutable `main`; tokenizer and model checksums are
verified together. The runtime's model identity changes with the checkpoint,
so prefix state must not be shared across old and new weights.

The promoted model revision is `59f422b2d410fdaf4a9efc71ff12f278abc2a5d1`.
Its metadata references engine implementation commit
`97c72de`, which includes the evaluated quantizer, tests and benchmark report.
The rebuilt default converter reproduces the calibrated artifact hash exactly.

## Optional mixed Q8 experiment

`--q8-gates` promotes the small recurrent B/A projections to identity-basis
Q8_0; `--q8-head` promotes the tied embedding/output matrix. Both flags default
to off. These artifacts introduce tensor encoding ID 7 and require a new
engine; older readers reject them. The packer writes a `.precision.json`
sidecar identifying these choices. They do not alter the published standard.
See the [selective Q8 experiment](q8-experiments-2026-09-06.md) for measured
gate results and format details. The subsequent
[Q8 embedding/output experiment](q8-head-experiment-2026-09-06.md) measures the
large tied matrix separately, with the recurrent gates retained in Q4.
