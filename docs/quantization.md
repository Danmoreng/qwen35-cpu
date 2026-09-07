# Standard quantization recipe

The standard [Hugging Face download](https://huggingface.co/danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4/tree/f6cb5cf04a9094670f9041578a3f4e44b94a1395) is **H128/Q4-G32-DOT4 with 256-document
calibration, weighted MSE16 and block-128 error compensation**. The `.q35h`
format, signed H128 transform, groups of 32 with FP16 scales and CPU DOT4 layout
are unchanged. No new runtime loader or GPU backend is required.

## Format and calibration names

`H128/Q4-G32-DOT4` names the representation: a signed Hadamard transform over
128 channels, 4-bit weights in groups of 32 with FP16 scales, and DOT4 packing
for CPU kernels. The tied embedding/output matrix uses the identity basis.
The calibration procedure is activation-weighted MSE16 fitting with error
compensation within 128-channel blocks. MSE16 refers to the sixteen signed
4-bit levels, not 16-bit weight storage.

Older study reports and artifact IDs use `B+C` for the combination of data and
sampling work (B) and bounded error compensation (C). The suffix `256` counts
calibration documents. These are experiment identifiers, not a separate weight
format; hashes and pinned revisions identify the exact checkpoint.

## Offline fitting

MSE16 searches both scale signs and all signed codes `[-8,7]`, evaluates stored
FP16 scales, and weights reconstruction error by real BF16 teacher activations
in the correct H128 basis (identity for the tied embedding/output matrix).
The error-compensation step uses second moments within 128-channel blocks, damping 0.01,
and error propagation across remaining channels in each block. It retains the
MSE16 candidate if the undamped reconstruction objective does not improve.
Cross-block covariance is omitted; this is not full-matrix GPTQ or sequential
layer recalibration.

The published recipe uses 256 independent selected documents / 262,144 tokens,
with approximately 25% German, 25% English, 20% Python code, 20% rendered dialogs
and 10% mathematics. A separate resident CUDA teacher collected activations;
CPU inference requires no CUDA. Dataset revisions, selection hashes and held-out
separation are recorded in the [study](g32-large-calibration-2026-09-07.md).
More data alone did not help MSE16, and the larger corpus is not better on every
suite. The method and the resulting checkpoint were evaluated separately.

## Defaults and reproduction

- **Download:** H128/Q4-G32-DOT4 with the calibrated recipe above, SHA256
  `013fbfaa03760e759181301ddaf964bb5c200c50c50617afe72557fd65bcbf0a`.
- **Source conversion:** unweighted `mse16` unless explicit fitting directories
  are supplied. The converter never silently discovers calibration files.

```powershell
# Published recipe, after preparing its fitting statistics
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model.q35h --importance-dir benchmarks/bc-large-study/full-importance --covariance-dir benchmarks/bc-large-study/full-covariance

# Unweighted conversion without calibration inputs
build/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-mse16.q35h
```

The [study driver](../scripts/run-large-calibration-study.py) and archived
source/provenance describe preparation, fitting, quality comparisons and matched
CPU speed measurements. Diagonal and covariance fitting statistics are offline
inputs, not runtime dependencies. The complete file is 424,964,864 bytes with
424,934,656 tensor bytes, unchanged from the earlier Q4 recipe.

The independent mixed final set measures PPL **12.71898** and teacher KL
**0.064178**; the English 8,192-position regression subset measures **15.80467**
and **0.060190**. These are different test sets and must not be compared directly.
See the [README](../README.md) for the matched llama.cpp/Unsloth comparison.

The previous four-document MSE16 artifact remains at Hugging Face revision
`59f422b2d410fdaf4a9efc71ff12f278abc2a5d1`; legacy remains at
`cc7df08da7ef7ac15db62e80b4eda85e19a143da`. Current download commands in the README
pin the new immutable revision. Prefix state must never be shared across
checkpoint identities. The source converter default has not changed.

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

## Optional head experiments

`--head-g16` gives each sixteen-weight half of the tied vocabulary matrix its
own FP16 scale. `--head-basis h128` rotates that matrix locally and applies the
inverse during embedding lookup. Both options default off. G16 requires the MSE16
fitter and now accepts identity-basis `--importance-dir` calibration; rotating the
head still rejects calibration inputs. See the
[head experiment report](head-experiments-2026-09-06.md) for format contracts,
quality tradeoffs and matched timing. These options do not change the published
calibrated standard.
