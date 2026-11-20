# Hadamard cost and native prefill on the Ryzen 9 9955HX3D

On November 20, 2026 we profiled the three size-matched native H128/Q4-G32-DOT4 checkpoints from the [laptop comparison](ryzen-9955hx3d-large.md). The largest measured cost was the Q4 matrix products. The signed Hadamard transform on activations accounted for an estimated 1.3–2.1% of native prefill time. The weight-side transform is already applied when packing the checkpoint.

## Where prefill time goes

| Model | Prompt tokens | Q4 input preparation | Estimated Hadamard share | Q4 matrix products |
| --- | ---: | ---: | ---: | ---: |
| 0.8B | 512 | 5.0% | 2.1% | 55.1% |
| 0.8B | 4,096 | 4.4% | 1.9% | 45.7% |
| 2B | 512 | 3.9% | 1.7% | 71.7% |
| 2B | 4,096 | 3.6% | 1.7% | 65.8% |
| 4B | 512 | 3.3% | 1.4% | 73.6% |
| 4B | 4,096 | 3.0% | 1.3% | 66.5% |

Percentages are fractions of the native model-forward wall time. “Preparation” covers the batched Q4 projection's full input preparation, including Hadamard, Q8 activation packing, and fused MLP work where enabled. “Q4 matrix” measures the subsequent parallel matrix call including dispatch and waiting. The transform estimate is part of preparation. The remaining time includes DeltaNet, full attention, other layer operations and the final vocabulary head.

To isolate the activation transform, we wrapped the signed AVX2 transform call inside `q4_h128_prepare_activations_4` in a four-iteration loop and rebuilt the binary with the baseline flags. This diagnostic computed the same signed 128-element Hadamard transform four times on the same input, overwriting the same destination before quantization. The subsequent Q8 packing and matrix product still ran once. For each paired run we subtracted stock preparation time from diagnostic preparation time, divided by three, then normalized by stock forward time. The table uses the median of the three estimates. Repeated calls reuse hot inputs and outputs, so this is an estimate of marginal cost. The complete preparation timer independently bounds how much that stage could save. Dropping the runtime transform would also put activations in a different basis from the stored weights.

## Optimization and comparison

The selected change processes 16 value rows of a DeltaNet head together. It transposes an 8 KiB slice of recurrent state, keeps it local while scanning the prompt in token order, and restores the standard state layout at the end. AVX-512 lanes span independent value rows. Each row preserves the eight accumulation chains and the floating-point reduction order of the existing path. Dispatch selects this implementation for prompt chunks with at least 32 tokens, 128 key columns and a value width divisible by 16. Single-token decode keeps its existing kernel.

| Model | Prompt tokens | Previous H128 | Optimized H128 | H128 gain | IK IQ4_K_R4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 512 | 3,062.5 | 3,251.6 | +6.2% | 3,763.0 |
| 0.8B | 4,096 | 2,696.7 | 2,854.5 | +5.9% | 3,209.9 |
| 2B | 512 | 1,519.0 | 1,585.4 | +4.4% | 1,786.2 |
| 2B | 4,096 | 1,429.3 | 1,472.6 | +3.0% | 1,670.0 |
| 4B | 512 | 622.4 | 650.9 | +4.6% | 717.1 |
| 4B | 4,096 | 565.0 | 588.6 | +4.2% | 650.1 |

Rates are median tokens/s. Baseline, candidate and ik_llama.cpp ran in the same confirmation series. Each timing case used a single prompt with an empty KV cache, full-vocabulary logits for its first output and zero decode steps. Native uses its model-forward prefill timer. IK measures the prompt loop with a synchronized `llama_decode`. Model loading and tokenization are outside both timers. The native format, calibration and on-disk tensor sizes are identical before and after the change.

We also tested larger native prefill chunks (256 and 512), two AVX-512 integer dot-product changes individually and in combination, and an eight-row recurrent tile using the existing decode kernel. Larger chunks slowed all three P512 cases. The DOT4 experiments produced smaller gains than the 16-row recurrent tile in the three-model screen. The measured IK revision has a separate prefill path for IQ4_K_R4 that can convert a tile to eight-bit weights and reuse it across multiple prompt positions. A useful next experiment would compare the current DOT4 kernel with a prefill variant that unpacks each weight tile once and reuses it across a larger block of prompt tokens. Attributing the remaining gap to a specific IK kernel would require profiling both matrix implementations on matching layer shapes. [IK dispatch](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_mul_mat.cpp#L288) · [IK conversion](https://github.com/ikawrakow/ik_llama.cpp/blob/98aca8692ae334e6e23e4d39ac43ce0d21029c35/ggml/src/iqk/iqk_gemm_iqk_quants.cpp#L3528)

A separate short-prompt control compared the two native builds at the dispatch threshold and at the default chunk size. Entries are changes in median prefill tokens/s.

| Model | 32 tokens | 128 tokens |
| --- | ---: | ---: |
| 0.8B | +5.6% | +4.7% |
| 2B | +3.0% | +2.5% |
| 4B | +3.4% | +2.7% |

## Validation and reproduction

The DeltaNet unit test compares full recurrent state and outputs byte for byte across head partitions, including batch lengths 31, 32, 33 and 128. Full-model checks used both prompt lengths and all three native checkpoints. Each pair produced identical FP32 logits across the entire vocabulary for four forced output positions. [Capture hashes](../results/prefill-profile-2026-11-20/logit-parity.json)

The laptop ran on AC power in the max-power profile. Eight threads were pinned to physical cores 0–7 on the 96 MiB L3 CCD. Each timing case had one warmup and three measured runs, with the order reversed in alternate rounds. All runs used the same token fixtures, FP16 KV and a 16,384-token per-request context limit. Profiling and timing series ran separately. Results apply to the tested laptop and build. [Per-run data](../results/prefill-profile-2026-11-20/samples.csv) · [Summary](../results/prefill-profile-2026-11-20/summary.csv) · [Revisions, hashes and environment](../results/prefill-profile-2026-11-20/manifest.json)

To reproduce the confirmation, build the baseline and selected revisions in Release mode and the pinned IK comparator. Extract the confirmation array from the [recorded matrices](../results/prefill-profile-2026-11-20/matrices.json) and adjust the executable and model paths for the local checkout.

```sh
OMP_NUM_THREADS=8 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix confirmation.json -OutputDir benchmarks/prefill-repeat \
  -Affinity 255 -WarmupRuns 1 -Runs 3
```
