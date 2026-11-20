# Liquid d1-2B CPU port feasibility

Liquid d1-2B is a feasible second architecture for native text decisions. Our H128/Q4-G32-DOT4 projection kernels ran both weight matrices of its first ShortConv mixer on CPU. The full model still needs a Liquid-specific loader, forward pass and decision readout. The existing [CPU results](benchmarks/decision-model-followup.md) used llama.cpp.

## What ran locally

The isolated November 20, 2026 probe used the original BF16 language-layer-zero input projection (6144 × 2048), seven-tap depthwise filter (2048 × 1 × 7) and output projection (2048 × 2048). Both projections used the repository's H128 quantizer, eight-row DOT4 packer, activation preparation and matrix-vector kernel. The filter used its original coefficients expanded to FP32. The input was 12 synthetic token vectors of width 2048 from NumPy's default random generator, seed 20261120. This spans both left padding and a complete filter window. [Checkpoint](https://huggingface.co/LiquidAI/d1-2B/tree/a6bce774e6dde338dcc3036ceb5d91f78bf4c19e) · [H128 primitives](../include/qwen35x/cpu/q4_h128.h) · [DOT4 primitives](../include/qwen35x/cpu/q4_dot4.h)

The input projection splits into B, C and X. The mixer filters B × X over the current and six preceding positions, multiplies by C and applies the output projection. An independent NumPy sliding-window calculation reproduced every FP32 value immediately before the output projection from the same quantized projection outputs. Across the full mixer the flattened output cosine was 0.983 against FP32 arithmetic on the original BF16 weights. This single uncalibrated synthetic input tests the mixer wiring. End-to-end decision fidelity and latency remain open.

## What needs implementing

| Component | Native port |
| --- | --- |
| Weights and export | Add an explicit LFM3 text profile, tensor mapping and architecture discriminator with backward-compatible Qwen loading. The dense projections have input widths of 2048 or 7680, both divisible by 128. Start from the original BF16 text weights, handle the tied embedding/output table explicitly and retain suitable precision for small filters and norms. Collect Liquid activations for calibration. [Converter](../src/q4_h128_convert_main.cpp) · [Artifact reader and writer](../src/weights/q4_h128_artifact.cpp) |
| Forward pass | Dispatch 13 ShortConv blocks, five causal sliding-attention blocks with a 256-token window and six global attention blocks. Match the checkpoint's pre/post RMSNorm, Q/K normalization, per-head sigmoid gate and distinct rotary settings. Reuse the existing matrix kernels and CPU dispatch. [Checkpoint configuration](https://huggingface.co/LiquidAI/d1-2B/blob/a6bce774e6dde338dcc3036ceb5d91f78bf4c19e/config.json) · [llama.cpp graph](https://github.com/ggml-org/llama.cpp/blob/ffc87c3a7989cb657641a6459a186f303e176af8/src/models/lfm3.cpp) |
| Prompt processing | For one question the six global layers read the whole prompt bidirectionally. A later token can change an earlier token's representation. Process the complete sequence layer by layer, with a bounded attention workspace. Existing causal Qwen prefix-cache assumptions do not hold here. Multiple questions packed around a shared state have additional mask rules. [Liquid mask semantics](https://huggingface.co/LiquidAI/d1-2B/blob/a6bce774e6dde338dcc3036ceb5d91f78bf4c19e/attention.py) · [Current prefill](../src/runtime/prefill.inl) |
| Decisions | Match Liquid's tokenizer, special tokens, prompt layout and answer positions. Group the appropriate label logits and apply the checkpoint's calibration before reporting probabilities. For ordinal scores the official default readout is the median of the predicted distribution. The existing HTTP request envelope can serve a Liquid-specific implementation. [Liquid decision API](https://huggingface.co/LiquidAI/d1-2B/blob/a6bce774e6dde338dcc3036ceb5d91f78bf4c19e/api.py) · [Existing Qev decision code](../src/systemone.h) |

## Proposed first milestone and effort

1. Freeze one short text-only multiple-choice request with expected token IDs, intermediate activations and label logits from a high-precision reference. Implement all 24 layers in a correctness-first CPU path and compare checkpoints through the first convolution, sliding and global attention blocks, then the final readout.
2. Export the text weights to H128/Q4-G32-DOT4 and replay the same request. Check how quantization changes option margins and probabilities. Add calibration only after the higher-precision path agrees.
3. Expose the validated choice path through the decision endpoint. Cover prompt length, request isolation, artifact compatibility and existing Qev models before expanding the supported decision types.

For one developer familiar with this engine, allow roughly 5–10 working days for the first reference-checked native text decision and another 2–4 weeks for calibrated artifacts, serving and regression coverage. The largest uncertainty is agreement on global attention and calibrated option probabilities. The estimate covers one text question per request. Joint question sets, vision and long-context performance tuning would need separate estimates.
