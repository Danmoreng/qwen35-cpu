# Q4 matrix kernel on the Ryzen 9 9955HX3D

This November 20, 2026 follow-up compares the [DeltaNet-optimized baseline](prefill-9955hx3d.md) with a change to the AVX-512 VNNI kernel used by H128/Q4-G32-DOT4. The same single-request prompts and size-matched checkpoints ran on eight physical cores of the laptop's 96 MiB L3 CCD.

| Qwen3.5 | Prompt tokens | Previous H128, tok/s | New H128, tok/s | Gain | ik_llama.cpp IQ4_K_R4, tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 512 | 3,251.1 | 3,351.1 | +3.1% | 3,773.1 |
| 0.8B | 4,096 | 2,857.4 | 2,931.5 | +2.6% | 3,219.3 |
| 2B | 512 | 1,584.9 | 1,645.7 | +3.8% | 1,798.9 |
| 2B | 4,096 | 1,480.6 | 1,533.8 | +3.6% | 1,674.7 |
| 4B | 512 | 656.5 | 676.6 | +3.1% | 723.2 |
| 4B | 4,096 | 592.7 | 611.4 | +3.2% | 648.0 |

The table uses median prefill throughput from three measured runs after one warmup. Each run started with an empty KV cache, returned one set of full-vocabulary logits and performed zero decode steps. Native rates use the model-forward timer. The IK timer covers its synchronized prompt loop. The [per-run samples](../results/q4-prefill-2026-11-20/samples.csv) include warmups and the [summary](../results/q4-prefill-2026-11-20/summary.csv) includes paired gains.

## How the kernel works

The previous 16-row kernel broadcast each group of four activation bytes for one pair of eight-row weight blocks. The new kernel computes 32 output rows for eight prompt tokens together. Each activation broadcast feeds both row pairs. Four-bit weights are unpacked in registers, and the integer accumulator starts with the correction for the Q4 zero point. Each output still visits the 32-column blocks in order and applies the original floating-point scale and FMA once per block.

The 32-row kernel runs on AVX-512 VNNI for packed batches of at least 64 vectors divisible by eight. Smaller batches, row tails and other instruction sets use the existing kernels. There is no additional packed weight copy. Existing checkpoints load directly.

## Matrix time

A separate instrumented series measured the elapsed time inside the packed Q4 matrix calls for a 512-token prompt. This includes dispatch and worker completion. Activation preparation, Hadamard, DeltaNet and the final vocabulary head are outside this timer.

| Qwen3.5 | Previous Q4 matrix time | New Q4 matrix time | Time saved |
| --- | ---: | ---: | ---: |
| 0.8B | 92.92 ms | 88.43 ms | 4.8% |
| 2B | 238.30 ms | 228.04 ms | 4.3% |
| 4B | 598.36 ms | 572.81 ms | 4.3% |

We also timed unpacking weight panels into scratch once for all prompt tokens, widening the existing token tile to 24, changing row and token traversal order, alternative 32–64-row register tiles and broadcasting eight activation bytes with a horizontal reduction. The scratch variants lost throughput at all three sizes. The 32-row, eight-token register tile improved throughput consistently across the screening series.

## Correctness and reproduction

The DOT4 test passed 10,080 cases on this host. It compares the packed matrix outputs byte for byte with the AVX2 reference. It exercises nontrivial strides, saturated and random input bytes, multiple block counts, partial row tiles, and batches on either side of the dispatch boundary. For each model and both prompt lengths we also compared the complete FP32 vocabulary logits with the baseline for four forced output positions. [Logit hashes](../results/q4-prefill-2026-11-20/logit-parity.json)

All runs used FP16 KV, an eight-thread affinity mask of `0xff` and a context limit of 16,384. The runner reversed case order on alternating rounds. Build the baseline at `2498e51` and the optimized revision in Release mode. The [manifest](../results/q4-prefill-2026-11-20/manifest.json) records the IK revision, checkpoint and executable hashes, candidate configurations and raw profile locations. Update executable and model paths in the [comparison matrix](../results/q4-prefill-2026-11-20/confirmation.json) for your checkout.

```sh
OMP_NUM_THREADS=8 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/q4-prefill-2026-11-20/confirmation.json \
  -OutputDir benchmarks/q4-prefill-repeat -Affinity 255 -WarmupRuns 1 -Runs 3
```
