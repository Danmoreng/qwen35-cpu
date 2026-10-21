# Local browser decision-engine comparison

This experiment compares the existing H128/Q4-G32-DOT4 engine with external
llama.cpp and Transformers.js / ONNX Runtime builds. All primary measurements
use browser CPU execution, the same qev fine-tune, fresh state, and the same
three prompts. There are no generated tokens or cross-request batches.
Nothing in this experiment adds a GPU backend to the native CPU engine, and
no code, models, or results have been published.

llama.cpp supports Emscripten/WASM builds. The small binding under
`tools/llama-wasm-benchmark` links unmodified upstream llama.cpp and exposes
model loading and final-slot option logits. Its model loader sets zero GPU
layers. The build explicitly selects the `wasm32` processor so ggml includes
its WASM SIMD quantization kernels rather than its generic fallback.

Transformers.js 4.3.1 supports Qwen3.5 and uses ONNX Runtime for inference.
The experiment uses its optimized ONNX graph layout, with separate embedding
and decoder sessions. The public graph templates describe the Qwen3.5 base
architecture; they do not contain the qev fine-tune. `prepare-qev-onnx.py`
replaces every external text-weight initializer with the corresponding local
qev source weight, applies Qwen's norm offsets and negative exponentials, and
transposes projection matrices into the graph's layout. It preserves the
template's fixed architecture constants. Every replacement is shape checked
and listed in the generated export manifest.

The ONNX Q8 and Q4 fixtures use ordinary asymmetric weight-only block
quantization with groups of 32 and FP32 activations, including quantized
embeddings and output projections. They use ONNX `MatMulNBits` and
`GatherBlockQuantized`; these are different formats from GGUF Q8_0, GGUF Q4_0,
and H128/Q4-G32-DOT4. Quantization names alone do not establish equivalent
quality. ONNX quantization here does not use our covariance calibration.

## Reproduction

Use the same local model files and SDK as [the WASM demo](wasm-local.md).
Keep generated weights, packages and builds in the ignored directories.

```sh
# Browser tooling (tested versions are also recorded with the results).
npm install --prefix .cache/wasm-browser --save-exact \
  playwright@1.64.1 @huggingface/transformers@4.3.1
uv pip install --python .cache/qev-venv/bin/python \
  numpy onnx==1.23.2 onnxruntime==1.31.0 onnx-ir==1.1.0

# Export the exact local qev text weights, not the base model weights.
.cache/qev-venv/bin/python scripts/prepare-qev-onnx.py --dtypes q8 q4

# Build the external CPU-only binding from the pinned local upstream checkout.
source .cache/emsdk/emsdk_env.sh
bash scripts/build-llama-wasm-benchmark.sh

# Create the GGUF comparison files if they are not already present.
build-qev-server-current/bin/llama-quantize --pure \
  models/qev-0.8b/qev-0.8b-bf16.gguf models/qev-0.8b/qev-0.8b-Q4_0.gguf Q4_0 8
build-qev-server-current/bin/llama-quantize \
  models/qev-0.8b/qev-0.8b-bf16.gguf models/qev-0.8b/qev-0.8b-Q4_K_M.gguf Q4_K_M 8

python3 scripts/stage-browser-benchmarks.py
python3 scripts/serve-wasm.py
```

The staging script links local files where possible, avoiding another copy of
large checkpoints. The external fixtures are served under `/llama-bench/`
and `/onnx-bench/`. Browser inference does not call an HTTP inference server.

Use `scripts/benchmark-browser-decision.mjs` only through
`scripts/benchmark-inference-seq.ps1`. The frozen input file is
`configs/qev-browser-bench.json`; its prompt IDs come from the native protocol
oracle for `configs/qev-wasm-bench.jsonl`. The inputs contain 74, 77 and 579
tokens. The custom engine receives the original raw System One JSON as well.
All cases use eight threads and process affinity mask 255 (CPUs 0–7).

Each fresh browser/engine warms each of the three shapes once before the
measured requests. The runner adds one complete process warmup and retains
three measured process runs. Downloading, model loading and graph setup are
excluded. All requests start with empty recurrent/KV state and compute only
the final prompt slot; there is no decode forward. The custom engine's worker
latency includes its JSON preparation and tokenization. External engine
worker latencies start from the frozen prompt IDs. Report this scope difference
alongside any comparison, and retain prefill-only timings when available.

The driver accepts `--engine custom`, `--engine llama.cpp`, or the default
Transformers.js engine. Supply `--provider wasm`, `--threads 8`,
`--dtype LABEL`, `--inputs-file configs/qev-browser-bench.json`, the installed
Playwright module, and a Chromium executable. llama.cpp also takes `--weights`
and `--ubatch`. ONNX takes `--ort-build plain`, `jsep`, or `asyncify`.
The sequential runner supplies `--profile-json`.

## CPU results, 2026-10-21

This table records the original custom engine before direct-FMA optimization.
See [WASM FMA optimization](wasm-fma-optimization.md) for the newer implementation
and a fresh comparison against the same llama.cpp builds. A further
[Q4 kernel optimization](wasm-kernel-optimization.md) records the current
browser timings and a new comparison with native execution.


AMD Ryzen 9 9955HX3D, Chromium 156.0.8078.4, Emscripten 6.0.13,
llama.cpp revision `9b56ccf00d755b0bf780140f6352a67db1ac0378`.
Each row uses eight threads and affinity 0–7. Values are medians of three
measured process runs following one process warmup and the in-session shape
warmups described above. The custom baseline and external candidate series
ran as separate sequential matrices; neither ran concurrently with another
inference benchmark or build.

| Browser CPU engine | Short routing, 74 tokens | Snake, 77 tokens | Long routing, 579 tokens |
| --- | ---: | ---: | ---: |
| llama.cpp Q8_0 | 316.40 ms | 326.92 ms | 2,471.19 ms |
| llama.cpp Q4_0 | 395.54 ms | 405.58 ms | 3,085.71 ms |
| Custom H128/Q4-G32-DOT4 | 417.31 ms | 421.81 ms | 3,233.03 ms |
| llama.cpp Q4_K_M | 441.46 ms | 449.20 ms | 3,370.58 ms |
| Transformers.js Q4, ONNX/JSEP | 490.64 ms | 504.73 ms | 2,583.82 ms |
| Transformers.js Q8, ONNX/JSEP | 493.17 ms | 483.51 ms | 2,592.73 ms |

For the custom engine and llama.cpp these are **prefill plus final logit
projection** timings, excluding tokenization and state preparation. ONNX
timings cover a pretokenized forward and option readout, including fresh
tensor/state allocation. Loading and main-thread IPC return are excluded.
Worker latencies, individual samples and input-token throughput are also
retained in [performance.csv](results/qev-browser-cpu-2026-10-21/performance.csv).
No number in this table measures generated tokens per second.

On these inputs the custom Q4 engine takes 4.0–5.5% longer than llama.cpp
Q4_0 and 4.1–6.1% less time than Q4_K_M. Its latency is 1.29–1.32 times the
Q8_0 latency. Thus it is competitive with these Q4 configurations, while
Q8_0 is faster. ONNX/JSEP is slower on the short requests and about 20%
faster on the long one. These three requests do not establish a universally
fastest browser engine.

The selected llama.cpp configuration uses upstream automatic Flash Attention
(confirmed active in the raw logs), `n_ubatch=512`, context 2048 and one
sequence. The initial Flash-Attention-disabled, `n_ubatch=1024` series was
slower on the long request and is retained in
[candidates.json](results/qev-browser-cpu-2026-10-21/candidates.json).
The custom engine uses 32-token chunks and no prefix reuse. The ONNX graph
uses its fused LinearAttention and GroupQueryAttention operators with fresh
state and one last-slot logit output.

For ONNX CPU execution the separate JSEP build was faster than the plain
WASM build in this test. The Transformers.js default `asyncify` build failed
to load the Q8 fixture on CPU with a missing `GatherBlockQuantized` kernel.
This failure is preserved in the raw diagnostic profile; it is excluded from
the timing table. No CPU performance claim is made for that failed build.
The same fixture completed WebGPU inference with `asyncify`.

### Size and memory

| Engine artifact | Model-file bytes | Maximum measured WASM linear memory |
| --- | ---: | ---: |
| Custom H128/Q4-G32-DOT4 | 424,964,864 | 548.56 MiB |
| llama.cpp Q4_0 | 435,898,528 | 1,412.88 MiB |
| llama.cpp Q4_K_M | 529,299,616 | 1,501.94 MiB |
| llama.cpp Q8_0 | 811,845,792 | 1,771.44 MiB |
| Transformers.js Q4 | 647,133,030 | Not captured |
| Transformers.js Q8 | 1,165,946,356 | Not captured |

These file sizes cover weights and model graphs, excluding the shared
tokenizer/runtime assets. ONNX has separate embedding and output-projection
storage. GGUF contains vocabulary metadata. The custom artifact and pure
Q4_0 have exactly the same tensor payload size (424,934,656 bytes); their
file-size difference is metadata/layout overhead.

Linear memory is not total browser RAM. The measurements use different
prefill chunk sizes, which affect scratch memory; they do not establish an
inherent memory ratio between the libraries. See
[memory.json](results/qev-browser-cpu-2026-10-21/memory.json).

### Probability checks

The quality reference is **native llama.cpp CPU BF16**, scored through
`/v1/systemone` with the same three requests. All browser candidates retained
the reference's selected choices on all three questions. The maximum absolute
probability difference is over every option in those questions. Tensor
payload sizes determine the row order; these are not accuracy rankings.

| Scoring backend and artifact | Snake `left` | Maximum probability difference from BF16 |
| --- | ---: | ---: |
| Native llama.cpp BF16 reference | 70.65% | Reference |
| Browser ONNX/JSEP Q8 | 70.79% | 0.14 percentage points |
| Browser llama.cpp Q8_0 | 73.50% | 2.85 percentage points |
| Browser ONNX/JSEP Q4 | 58.97% | 13.13 percentage points |
| Browser llama.cpp Q4_K_M | 66.85% | 3.80 percentage points |
| Browser H128/Q4-G32-DOT4 | 66.08% | 4.57 percentage points |
| Browser llama.cpp Q4_0 | 79.56% | 8.91 percentage points |

Backend numerics matter even with identical GGUF bytes. The native Q8_0
reference selected `left` with 70.38%, versus 73.50% in its WASM build, a
3.12-percentage-point difference. Q4_K_M also differed between native and
WASM. The respective contributions of reduction order, activation quantization,
Flash Attention and math implementations have not been isolated.

This is a three-question numerical smoke check, not a rerun of the complete
qev evaluation set. It cannot establish relative benchmark accuracy or
probability calibration across tasks. The earlier 24-question check compared
our WASM engine with our native Q4 engine; its 0.40-percentage-point limit is
a different comparison from the BF16 differences above.

Full probabilities and native quantized references are retained in
[quality.json](results/qev-browser-cpu-2026-10-21/quality.json).
Commands, upstream revisions, model/source/binary hashes, affinity and raw
profile locations are in
[provenance.json](results/qev-browser-cpu-2026-10-21/provenance.json).
Raw profiles and the earlier Flash-Attention-disabled binary remain locally
under ignored `benchmarks/qev-browser-cpu-2026-10-21/`.

## WebGPU and WebGL

WebGPU is GPU execution, including when the GPU is integrated into the CPU's
package. It is a separate resource/performance comparison from WASM CPU.
Transformers.js / ONNX Runtime can use WebGPU. The qev Q8 and Q4 fixtures both
completed real inference on this machine's NVIDIA Blackwell adapter with
`isFallbackAdapter=false`. The local Chromium needed a headful window and
`--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan`.
These flags and adapter details are retained in the diagnostic profiles.

For these GPU diagnostics the adapter uses the `asyncify` runtime, which is
also Transformers.js 4.3.1's default runtime choice. The older `jsep` binary
does not expose the newer native WebGPU initialization API. GPU timings are
kept separate from the primary CPU comparison.

ONNX Runtime documents WebGL as maintenance mode and recommends WebGPU.
An arbitrary Qwen3.5 graph cannot be assumed to work in WebGL: the specialized
linear-attention operators also require backend support. This experiment did
not implement or benchmark a WebGL path. Our H128 packed weights would require
dedicated GPU kernels or conversion to another representation for GPU use.

Sources: [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md),
[Transformers.js 4.3.1](https://github.com/huggingface/transformers.js/releases/tag/4.3.1),
[optimized ONNX graph templates](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX-OPT/tree/9a4c21d8f724176b0a7edaaed36303044451886e),
[ONNX Runtime Web backends](https://onnxruntime.ai/docs/get-started/with-javascript/web.html).
