# Choosing a local decision model

November 20, 2026. This extends the [four-model GPU comparison](decision-models-gpu.md) with JevBench-240, an authored German routing set and a paired Liquid quantization comparison. We measured quality on the RTX 5080 Laptop GPU and response latency separately on the Ryzen CPU. For German tool routing, use Liquid d1-2B Q4_K_M through the existing llama.cpp decision endpoint. It kept 55 of 64 German cases correct and took 274 ms for the median German request. The CPU measurements give a concrete latency target for any native implementation.

## Does Qev benefit from one question per prompt?

| Model | Coherence / 100 | 95% interval | Gold accuracy | Batch isolation | Probability laws |
| --- | --- | --- | --- | --- | --- |
| Liquid d1-2B Q8_0 | 72.1 | 71.3–73.1 | 88.4% | 100.0 | 39.6 |
| Qev Q8_0 joint | 62.6 | 61.5–63.8 | 84.9% | 63.9 | 35.5 |
| Qev Q8_0 per question | 70.0 | 69.0–71.0 | 84.7% | 100.0 | 34.9 |

Each configuration completed all 12,000 frozen tests across 50 relations and 240 cases. Accuracy uses the 1,938 base questions with known answers. The interval is the upstream case-bootstrap interval for coherence. The frozen transformations reuse the base cases, so the 12,000 tests do not represent 12,000 independent customer requests. Qev's two publisher GGUFs have identical names, shapes, quantization types and bytes for all 320 weight tensors. The vision-tagged file selects per-question evaluation for text in stock llama.cpp. This run sent text only. [Full scores by relation and domain](../results/decision-followup-2026-11-20/jevbench-full.json) · [Upstream suite](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3)

## German routing

| Model | Correct / 64 | Call / 16 | Ask / 16 | Unsupported / 16 | Direct / 16 |
| --- | --- | --- | --- | --- | --- |
| Liquid d1-2B Q8_0 | 58 | 16 | 15 | 11 | 16 |
| Qev Q8_0 joint | 25 | 6 | 10 | 5 | 4 |
| Qev Q8_0 per question | 25 | 6 | 10 | 5 | 4 |
| Liquid d1-2B Q4_K_M | 55 | 15 | 13 | 11 | 16 |

The panel has 16 scenarios, each with the same tool description and fact sheet for four requests. In one scenario a caller might ask for the status of a specific order, omit the required order number, request a change that the read-only tool cannot make, or ask a question covered by the fact sheet. One action is correct per request. Correct option positions are balanced within each action. We froze both files before any German inference. The inference client read only the input file, and scoring joined its predictions to a separate key by opaque item ID. For example, the music-school lookup requires both course number and lesson date. Given only course M15, Liquid Q8 asked for the date and Liquid Q4 chose the lookup. On the cinema case requesting a change to the audio track, Q4 correctly declined the requested change with only a read-only tool available, while Q8 selected a call. These synthetic cases measure the tested contrasts. They do not give an estimate for production traffic. [German inputs](../results/decision-followup-2026-11-20/routing-de-inputs.jsonl) · [Answer key](../results/decision-followup-2026-11-20/routing-de-gold.jsonl) · [Predictions](../results/decision-followup-2026-11-20/routing-de-predictions.csv)

## Does the routing advantage survive four-bit quantization?

| Liquid weights | File (GiB) | Benchmark Heaven / 231 | Heaven Brier ↓ | Mini coherence / 100 | When2Call / 3,652 | German / 64 |
| --- | --- | --- | --- | --- | --- | --- |
| Q8_0 | 1.72 | 177 | 0.285 | 73.8 | 3,295 (90.2%) | 58 |
| Q4_K_M | 1.02 | 184 | 0.273 | 72.5 | 3,282 (89.9%) | 55 |

On the complete When2Call test, Q8 and Q4 chose the same reply on 3,580 items. Only Q8 was correct on 37 and only Q4 was correct on 24. The comparison uses the original 3,652-item denominator and the repository's frozen candidate order and tie rule. All Q4 probabilities came from the same llama.cpp endpoint as Q8. [Quality and gate results](../results/decision-followup-2026-11-20/routing-and-quantization.json) · [Paired When2Call predictions](../results/decision-followup-2026-11-20/when2call-paired.csv) · [Paired Benchmark Heaven predictions](../results/decision-followup-2026-11-20/heaven-paired.csv)

## Resident CPU latency

| CPU model | German p50 ms | German p95 ms | When2Call p50 ms | When2Call p95 ms | Peak RSS (GiB) |
| --- | --- | --- | --- | --- | --- |
| Liquid d1-2B Q8_0 | 470.0 | 492.2 | 2,197.8 | 4,831.4 | 2.23 |
| Liquid d1-2B Q4_K_M | 273.9 | 285.1 | 1,258.5 | 2,821.6 | 2.19 |
| Qev Q8_0 joint | 187.1 | 200.1 | 922.2 | 2,108.7 | 1.20 |

Liquid Q4 passed the predeclared routing gate on both datasets. The first warmup stopped when Liquid rejected a 652-token prompt at the stock physical batch limit of 512. We reran the entire matrix with both batch sizes set to 8,192 for every model and kept all inputs intact. The stock CPU-only llama.cpp server ran on the Ryzen 9 9955HX3D with eight physical cores on the 96 MiB L3 CCD (logical CPUs 0–7). Each process loaded one model and served 88 distinct, interleaved, serial HTTP requests with zero generated tokens. Every model saw all 64 German cases and the same 24 When2Call examples, sampled at eight length quantiles within each target class. The sequential PowerShell harness ran one warmup and three measured passes, reversing the case order on alternate passes. The table reports the median across runs of the within-run p50, nearest-rank p95 and Linux server peak RSS. HTTP, tokenization and scoring are included. Model loading is outside the request timer. The p95 for the smaller tool panel is sensitive to individual long prompts. Both Liquid configurations reproduced their GPU choices on all 88 CPU requests in each measured pass. Qev reproduced 85 of 88, with the same three discrepancies on the German panel each time. Liquid Q4 cut median response time by about 42% on German cases and 43% on the tool panel relative to Q8. Its smaller GGUF cut peak server RSS by only 0.04 GiB with these batch settings. [Per-run timing, token ranges, CPU/GPU agreement and binary hashes](../results/decision-followup-2026-11-20/cpu-latency.json)

[Protocol frozen before the German and Q4 runs](../results/decision-followup-2026-11-20/protocol.md) · [Model hashes, GPU settings and dataset revisions](../results/decision-followup-2026-11-20/provenance.json)
