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
separation are recorded in the [study](benchmarks/calibration.md).
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
**0.064178**; the original calibration study's English 8,192-position regression
subset measured **15.80467** and **0.060190**. These are different test sets and
must not be compared directly. The [current CPU comparison](benchmarks/cpu.md)
reports the separately refreshed engine measurements.
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
Both modes can be combined with `--covariance-dir`: promoted Q8 tensors bypass
the Q4 fitting inputs, while the remaining Q4 tensors use the supplied diagonal
and covariance statistics. See the [qev decision study](benchmarks/decision-quality.md)
for a model-specific comparison; results do not transfer automatically between
the base/chat model and its decision fine-tune.
See the [selective Q8 experiment](https://github.com/Danmoreng/qwen35-cpu/blob/3680efb33bf6975ce6722d64d20c641cb7490b06/docs/q8-experiments-2026-09-06.md) for measured
gate results and format details. The subsequent
[Q8 embedding/output experiment](https://github.com/Danmoreng/qwen35-cpu/blob/3680efb33bf6975ce6722d64d20c641cb7490b06/docs/q8-head-experiment-2026-09-06.md) measures the
large tied matrix separately, with the recurrent gates retained in Q4.

## Optional head experiments

`--head-g16` gives each sixteen-weight half of the tied vocabulary matrix its
own FP16 scale. `--head-basis h128` rotates that matrix locally and applies the
inverse during embedding lookup. Both options default off. G16 requires the MSE16
fitter and now accepts identity-basis `--importance-dir` calibration; rotating the
head still rejects calibration inputs. See the
[head experiment report](https://github.com/Danmoreng/qwen35-cpu/blob/3680efb33bf6975ce6722d64d20c641cb7490b06/docs/head-experiments-2026-09-06.md) for format contracts,
quality tradeoffs and matched timing. These options do not change the published
calibrated standard.

## Qwen3.5-2B

Qwen3.5-2B is the second native H128/Q4-G32-DOT4 model option. Use the
official 2B tokenizer and the model-specific calibrated MSE16 fit. The
validated `model.q35h` is 1,060,442,880 bytes with SHA-256
`1084c3a2751d3813a2aa686337f717e56291e6a1a79ff8be1b6d1665d22892a3`.
The following command serves the calibrated file once it has been placed in
the model directory alongside the tokenizer assets.

```sh
build/qwen35_cpu_server --model-dir models/qwen3.5-2b \
  --weights models/qwen3.5-2b/model.q35h --threads 6
```

The [Ryzen text panel](benchmarks/qwen35-text-panel.md) compares the two native
models on fixed German correction, Python edit and JSON extraction tasks.

The following Linux commands build an unweighted MSE16 baseline.

```sh
hf download Qwen/Qwen3.5-2B --revision 15852e8c16360a2fea060d615a32b45270f8a8fc --local-dir models/qwen3.5-2b
build/qwen35_cpu_pack --hf-model-dir models/qwen3.5-2b --output models/qwen3.5-2b/model-mse16.q35h
build/qwen35_cpu --model-dir models/qwen3.5-2b --weights models/qwen3.5-2b/model-mse16.q35h --prompt "Once upon a time" --max-new-tokens 32
```

Calibration uses a text-only BF16 GGUF converted from the same checkpoint.
Pass its path via `--teacher` to `scripts/collect-resident-calibration.py` and use
the 2B tokenizer when preparing the corpus. Fit importance and block covariance
from these captures and supply both directories to the packer. The optional
CUDA collector runs offline. The native inference executable uses the CPU.

For a Pascal GPU with compute capability 6.1, the following example uses
CUDA 12.4, GCC 13 and the pinned llama.cpp checkout under `.cache/llama.cpp`.
Set the cuBLAS compute type to FP32 so that the collector can multiply the
original BF16 weights on this device. Use a Python environment with the
llama.cpp converter dependencies, Transformers, PyArrow, fsspec HTTP support,
NumPy and threadpoolctl. The capture cleanup script requires PowerShell 7.

```sh
cmake -S tools/llama-calibration-cuda -B build-llama-calibration-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-13 -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_C_COMPILER=gcc-13 \
  -DGGML_CUDA_FA=OFF -DGGML_CUDA_NCCL=OFF -DLLAMA_SOURCE_DIR="$PWD/.cache/llama.cpp"
cmake --build build-llama-calibration-cuda --target llama-calibration-resident-cuda
teacher=models/llama-comparison/Qwen3.5-2B-BF16.gguf
mkdir -p models/llama-comparison
python .cache/llama.cpp/convert_hf_to_gguf.py models/qwen3.5-2b --outtype bf16 --outfile "$teacher"
work=benchmarks/2b-onboarding
python scripts/prepare-diverse-calibration.py --large --model-dir models/qwen3.5-2b --out "$work/corpus"
python scripts/audit-calibration-corpus.py "$work/corpus"
GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 python scripts/collect-resident-calibration.py \
  --teacher "$teacher" --backend cuda --threads 6 --corpus "$work/corpus" --out "$work/collection" --block-covariance
python scripts/fit-dot4-quant.py "$work/collection/full" --out "$work/importance"
python scripts/fit-block-covariance.py "$work/collection/covariance-full" --out "$work/covariance"
build/qwen35_cpu_pack --hf-model-dir models/qwen3.5-2b --output models/qwen3.5-2b/model.q35h \
  --importance-dir "$work/importance" --covariance-dir "$work/covariance"
```

The 2B pilot, final held-out quality and native CPU timings are in the
[2B measurement report](benchmarks/qwen35-2b-i7.md).

The CPU reference collector builds from `tools/llama-comparison` as target
`llama-calibration-resident-cpu` and accepts the same jobs file. The collection
wrapper selects it with `--backend cpu`. Use separate output directories for
each run.
