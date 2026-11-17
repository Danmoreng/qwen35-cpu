# AVX-512 DeltaNet single-stream decode on a Xeon VM

November 17, 2026. Qwen3.5-0.8B H128/Q4-G32-DOT4 running greedy text generation on three threads, pinned to guest vCPUs 0, 2 and 4 of an Intel Xeon Platinum 8573C. GCC 14.2 Release. Both engines used automatic AVX-512/VNNI dispatch for the rest of the model. The stock single-token DeltaNet used AVX2. The 128-column candidates preserve the eight-lane accumulation and reduction order. Other widths and the prefill scan use the existing routines.

The row-pair variant processes eight value rows at a time, assigning independent rows to the two halves of each vector. The register variant processes two rows and retains the decayed 128-column slice until the final update. Each writes recurrent state once. The engine selects the row-pair implementation for 128-column single-token AVX-512 updates.

## Complete model

Fresh processes used one warmup and six measured pairs for row-pairs, four for registers. One prompt has 38 tokens (a lighthouse story), the other uses the existing 512-token fixture. Both generated all 128 output IDs identically to stock. The first comes from prefill. Rates cover 127 subsequent decode forwards. Ranges and sample-level data are in the summary. Independent process series exhibited substantial timing noise. Cross-series comparisons use their own controls.

| Candidate | Prompt | Fresh stock / candidate, median tokens/s | Paired windows, stock / candidate, median tokens/s | Three process-level paired changes |
|---|---|---:|---:|---|
| row-pairs | story | 35.80 / 34.23 | 35.49 / 35.46 | -0.1%, +2.5%, +3.5% |
| row-pairs | p512 | 32.97 / 34.50 | 33.80 / 34.36 | +2.0%, +2.3%, +1.6% |
| registers | story | 34.27 / 35.42 | 31.06 / 31.78 | -1.2%, +2.3%, +0.2% |
| registers | p512 | 34.03 / 31.30 | 18.55 / 19.11 | -0.7%, +3.0%, +2.2% |

The closer comparison interleaves windows in the same process. Two independent requests share a model and executor. Each advances alone for 12 decode forwards, alternating which runs first across 16 pairs of windows. A Linux link wrapper selects the baseline or candidate only at the DeltaNet call. Three measured processes follow a warmup; each produces the same 193 IDs per request. Individual windows within a process are correlated.

The row-pair kernel improved all three paired P512 sessions, by 1.6–2.3%. Story results were mixed across methods. The register variant had the lowest isolated latency at 18 MiB, but the P512 median in fresh processes fell from 34.03 to 31.30 tokens/s. Medians of rates in separate columns can differ from the median of paired ratios because the same process need not supply both medians.

## Kernel and validation

In the 18-MiB isolated state test, median time per layer update was 53.9 µs for AVX2, 49.5 µs for row-pairs, and 39.9 µs for registers. At 432 MiB these were 107.1, 97.0 and 98.3 µs. Each uses 16 heads, 128 key columns, 128 value rows, and the production three-thread executor. Five timed samples followed one warmup. Output checksums matched. In three instrumented P512 processes, the 18 layer updates together took a median 2.30 ms per decode token with stock and 2.01 ms with row-pairs. Other engine operations dominate token latency.

For each experimental implementation, the same 24 full-vocabulary vectors (12 per prompt) matched stock byte for byte: 5,959,680 FP32 logit values. Both passed 504 direct row-range comparisons against AVX2, checking the full state and output after four updates. The selected changes also run through the ordinary engine entry points. Tests and measurements are for this CPU and configuration.

## Reproduction

Use the [compact evidence](../results/deltanet-xeon-2026-11-17/summary.json) for parent revision, input and binary hashes, individual process rates and validation. Build the parent and selected versions in Release mode. For each binary, run the repository sequential benchmark runner with affinity 21, and the following arguments for each prompt. Substitute local paths for the checkpoint and token fixtures.

```sh
qwen35_cpu_bench --hf-model-dir MODEL --cpu-q4-h128 MODEL/model.q35h \
  --cpu-batch 1 --cpu-threads 3 --cpu-isa auto --cpu-isa-strict \
  --prompt-tokens-file PROMPT.csv --max-context 2048 --max-new-tokens 128 \
  --temperature 0 --repeat-penalty 1 --profile-json PROFILE.json
```

The [story input](../results/deltanet-xeon-2026-11-17/story.csv) and [512-token input](../results/cpu-refresh-2026-10-22/prompt-512.csv) are fixed.
