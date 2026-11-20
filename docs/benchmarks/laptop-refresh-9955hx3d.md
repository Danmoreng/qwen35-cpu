# Ryzen 9 9955HX3D CPU benchmarks after the prefill changes

## qev-0.8b decision latency

On November 20, 2026, the calibrated Labqoat qev-0.8b H128/Q4-G32-DOT4 checkpoint ran through `POST /v1/systemone` on eight physical cores (0–7) of the laptop's 96 MiB L3 CCD. All three servers used the same HTTP code, tokenizer, weights, adaptive chunk size and option-letter readout. Only the linked inference core varied. The baseline predates the two kernel changes, the middle column adds the [DeltaNet kernel](prefill-9955hx3d.md), and the final column also includes the [Q4 matrix kernel](q4-prefill-9955hx3d.md).

| Decision workload | Before | DeltaNet | DeltaNet + Q4 | Latency reduction vs. before |
| --- | ---: | ---: | ---: | ---: |
| First ticket, 74 tokens | 43.60 ms | 41.22 ms | 40.29 ms | 7.6% |
| First ticket, 579 tokens | 194.86 ms | 184.67 ms | 180.07 ms | 7.6% |
| 64 German routing requests, 178–216 tokens | 63.08 ms | 59.35 ms | 57.50 ms | 8.9% |

The ticket rows time the first request after a fresh server reports ready. The routing row is the median of three within-run p50 latencies, each from 64 distinct requests served consecutively by one resident server. Its p95 fell from 68.26 to 62.86 ms. All 64 routing inputs had a lower median latency with the optimized core when compared individually. All 66 inputs produced identical returned choices, probabilities and token usage across builds and repetitions. Each request generated zero tokens and read zero cached prefix tokens. These timings include localhost HTTP, tokenization, model inference and response formatting. Model startup is outside the timer. [Ticket fixtures](../results/qev-api-2026-10-21/) · [German requests](../results/decision-followup-2026-11-20/routing-de-inputs.jsonl) · [Decision summary and provenance](../results/laptop-refresh-2026-11-20/decisions.json) · [Per-request latencies](../results/laptop-refresh-2026-11-20/decision-requests.csv)

## Base-model throughput

The [README table and chart](../../README.md#cpu-performance) use a separate run of all 11 engine/checkpoint combinations. It repeats the November [0.8B](ryzen-9955hx3d.md) and [2B/4B](ryzen-9955hx3d-large.md) workloads with the optimized native engine. The 33 cases ran sequentially for one warmup and three measured rounds, reversing order on alternate rounds. All comparators, model artifacts, fixed token IDs, thread affinity, FP16 KV and full-vocabulary readout match those earlier series.

For one request, prefill processes 512 or 4,096 input tokens. The corresponding workloads force 128 or two output tokens. Decode uses the 512-input workload and counts 127 forward steps per request because prefill produces the first output. B=16 reports aggregate decode throughput across 16 private requests. Native prefill uses its model-forward timer and the comparators measure synchronized prompt processing. All three exclude model loading, tokenization and HTTP. The README chart displays the three equal-size tensor payloads at each model size. The table also includes the two differently sized 0.8B comparators. [Medians and ranges](../results/laptop-refresh-2026-11-20/performance.csv) · [All throughput samples](../results/laptop-refresh-2026-11-20/samples.csv) · [Revisions, binaries, model hashes and host settings](../results/laptop-refresh-2026-11-20/manifest.json)

## Reproduction

Use Release builds and the pinned model files. The decision series linked the same server object and support libraries against the three recorded inference cores. The decision manifest records hashes for each server and core. Adjust the local executable and model paths in each matrix and use a fresh output directory for each run.

```sh
OMP_NUM_THREADS=8 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/laptop-refresh-2026-11-20/decision-matrix.json \
  -OutputDir benchmarks/repeat-qev-latency -Affinity 255 -WarmupRuns 1 -Runs 3

OMP_NUM_THREADS=8 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/laptop-refresh-2026-11-20/matrix.json \
  -OutputDir benchmarks/repeat-laptop-throughput -Affinity 255 -WarmupRuns 1 -Runs 3
```
