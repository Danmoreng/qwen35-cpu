# Qwen3.5-2B and 4B on the Ryzen 9 9955HX3D

On November 20, 2026, the native engine, ik_llama.cpp and llama.cpp ran the same fixed-token workloads on the laptop used for the [0.8B comparison](ryzen-9955hx3d.md). The [README](../../README.md#cpu-performance) gives the median rates. The [results CSV](../results/ryzen-large-2026-11-20/performance.csv) also gives the minimum and maximum for each workload. [Individual samples](../results/ryzen-large-2026-11-20/samples.csv)

With 16 simultaneous requests, native decode exceeded IK by 56% at 2B and 53% at 4B. For one request, IK led native decode by 3.4% and 2.4%, respectively. All three engines recorded their fastest single-request decode in the final round at both sizes. Individual rates varied more across rounds than the differences between single-request medians.

## Checkpoints

| Qwen3.5 text model | Tensor bytes in each format | Native H128/Q4-G32-DOT4 weights |
| --- | ---: | --- |
| 2B | 1,060,412,672 | Activation-calibrated checkpoint from the [2B study](qwen35-2b-i7.md) |
| 4B | 2,369,040,384 | Unweighted MSE16 checkpoint from the [4B study](qwen35-4b-ryzen.md) |

Both native files were copied from the desktop and verified against the earlier SHA-256 records. The comparison files use pure IQ4_K_R4 for IK and pure Q4_0 for llama.cpp, quantized from BF16 text weights from the same [2B](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc) and [4B](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a) snapshots. Each format has the same on-disk tensor payload at a given model size. File sizes include additional metadata, and runtime memory depends on the loader and KV cache. This run measures speed. Earlier [2B quality results](qwen35-2b-i7.md) and [4B quality results](qwen35-4b-pi-agent.md) used separate evaluation fixtures.

## Timing contract

The Ryzen 9 9955HX3D ran on AC power in its max-power platform profile. Eight threads used physical cores 0–7 on the 96 MiB L3 CCD. The Release binaries and shared libraries match the November 19 comparison. The same prompt and continuation IDs, FP16 KV, 16,384-token context limit per request and full-vocabulary logits were used throughout. IK used runtime repacking and its [graph-capacity patch](ik-batch16.md#reproduction) for batch 16. Each process loaded a single model. Loading, tokenization and HTTP are outside the timers.

- **Single request, 512 input and 128 output tokens.** Prefill processes one prompt. Decode counts 127 forward passes because prefill produces the first output.
- **Single request, 4,096 input and two output tokens.** The README reports prompt processing from this case.
- **Sixteen private requests, each with 512 input and 128 output tokens.** Decode reports 2,032 forward passes divided by total decode time. Each request processes its own prefix. Speculative decoding is disabled.

Native prefill uses the model-forward timer. The comparator prefill timer measures synchronized `llama_decode`. The runner executes one case at a time and reverses the order on alternate rounds. The median excludes the initial warmup. The [manifest](../results/ryzen-large-2026-11-20/manifest.json) records revisions, hashes, quantizer commands, host settings and the raw profile directory. All inference arguments are in the [matrix](../results/ryzen-large-2026-11-20/matrix.json).

```sh
OMP_NUM_THREADS=8 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/ryzen-large-2026-11-20/matrix.json \
  -OutputDir benchmarks/ryzen-large-repeat -Affinity 255 -WarmupRuns 1 -Runs 3
```
