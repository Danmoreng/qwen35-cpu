# Experimental native GGUF execution

Source builds support the tested Qwen3.5-0.8B **Q4_0 `--pure`** and **Q4_K_M**
GGUF files. Released v0.1.1 binaries predate this feature; build current source.
This remains a CPU-only engine for this one text architecture. It does not
delegate inference to llama.cpp or require ggml libraries at runtime.

## Q4_0: reuse the optimized DOT4 path

The pure checkpoint contains 187 Q4_0 matrices and 133 retained F32 tensors.
The loader losslessly rearranges Q4_0 nibbles and FP16 scales into the existing
eight-row DOT4 layout. It does **not** requantize weights. Concatenated gate/up,
QKV and linear-attention projections retain that layout. Their transient
canonical block copies are released after loading.

Prefill, decode, compact Q8 activations, VNNI projections, fused greedy heads,
the shared executor, sampling, scheduling and prefix reuse use the existing
paths. H128 transforms are disabled. GGUF normalization offsets and the directly
stored `ssm_a = -exp(A_log)` are translated to this runtime's conventions.

Packing currently happens **once at model load**, not once per token or request.
This differs from the already CPU-packed public H128 checkpoint. There is no
new offline pure-Q4 artifact or published Hugging Face model in this change.

The downloaded unsloth file named `Q4_0` is **not** pure Q4_0: it contains
129 Q4_0, 36 Q8_0, 18 Q5_K, three Q4_1, one Q6_K and 133 F32 tensors. Its Q4_1
tensors are unsupported. Use the exact pure checkpoint below, not that file.

## Run

Config and tokenizer files are still required in `--model-dir`; GGUF tokenizer
metadata is not used. They must match Qwen3.5-0.8B. For an existing local setup:

```powershell
./scripts/build.ps1
./build/qwen35_cpu.exe --model-dir models/hf-download-test --weights models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf --prompt "Once upon a time" --threads 8 --max-new-tokens 128
./build/qwen35_cpu_server.exe --model-dir models/hf-download-test --weights models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf --threads 8
```

The server reports `Qwen3.5-0.8B-GGUF` from `/v1/models`. Omit the request's model
field or use that ID. The benchmark executable accepts `--cpu-gguf FILE`; CLI
and server use `--weights FILE`. The loader detects file magic. The historical
C++ field `CpuLoadOptions::cpu_q4_h128_path` accepts either format.

The comparison's pure file was created with the pinned llama.cpp quantizer:

```powershell
./build-llama/bin/llama-quantize.exe --pure models/llama-comparison/Qwen3.5-0.8B-BF16.gguf models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf Q4_0 12
```

See the [original comparison](comparison-2026-09-06.md) for pinned BF16 source,
quantizer revision and setup. The downloaded mixed `Q4_0` is not interchangeable.

## Paused Q4_K_M experiment

The earlier prototype also remains available via `--weights` with the tested
Q4_K_M GGUF. It retains original packed Q4_K/Q5_K/Q6_K/Q8_0 bytes and shares
immutable row segments for mixed-type projection groups. It uses native scalar
or AVX2 integer kernels, with four activation rows sharing each decoded weight
group, and Q8_K-style signed FP32 activation scales for K quants. It has no
H128 transformation, whole-model FP32 dequantization, or hidden requantization.
Greedy selection currently falls back to full logits for this path.

This first implementation is substantially slower at prefill than H128 or
native pure Q4_0. It has **not** received the mature DOT4/VNNI kernel tuning.
Further K-quant optimization and its full quality sweep were paused when the
implementation priority changed to pure Q4_0. A three-run P512/N128/B1/T8 trial
measured H128 2,134.95/120.89, native Q4_K_M 372.83/88.80, and llama.cpp Q4_K_M
685.36/83.58 tokens/s (prefill/decode). Those are a separate measurement group.

## Validation and measurements

Ten CTest tests pass, including independent GGML dequantization fixtures,
scalar/SIMD dots, batch/stride equivalence and malformed GGUF payloads. The
existing model scheduler test passes with both native formats: private logits,
queued requests, cancellation, backpressure, single-flight prefix builds and
exact cold/restored branch logits.

The [native Q4_0 report](native-q4_0-2026-09-06.md) records the bounded quality
check, arithmetic regression, new sequential speed measurements and raw data.
Repacking preserves weights, but activation quantization, accumulation order,
normalization and recurrent/attention implementations can produce differences
from llama.cpp. Engine logits are not claimed to be bit-identical to llama.cpp.
