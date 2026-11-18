# Fine-tune serving on a four-core Xeon VM

The study goal was to take an independent Qwen3.5-0.8B fine-tune from
safetensors to a working CPU service, measure task quality against its original
weights, open the selected output head to single-token classifiers, and test
whether short independent QEV requests benefit from shared prefill.

## Document relevance and ticket routing

Prism uses the first-step logits for the tokens `yes` and `no` to score a
query/document pair. We used the prompt from its model card, 12 new queries,
one relevant document and two distractors per query. The examples were written
before scoring. They include English, German and French. Each engine scored
the same 36 prompts, and all prompt token counts matched. The engine's new
package used uncalibrated MSE16 H128/Q4-G32-DOT4. It occupies 424,964,864 bytes
of weights. The reference ran the original BF16 checkpoint in Transformers on CPU.

The QEV check used the published calibrated H128/Q4-G32-DOT4 package and its
actual System One endpoint. Its 16 distinct support tickets cover four
departments evenly. The BF16 reference uses the same class tokens and QEV
choice temperature of 1.7. Both tested upstream QEV revisions have identical
safetensors contents.

| Panel | BF16 accuracy | Native Q4 accuracy | Macro-F1 BF16 / Q4 | Top-1 agreement | Mean absolute class probability difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prism document relevance, 36 pairs | 33/36 | 33/36 | 0.908 / 0.911 | 34/36 | 0.1007 |
| QEV support routing, 16 tickets | 15/16 | 15/16 | 0.937 / 0.937 | 16/16 | 0.0122 |

Both Prism models ranked the relevant document first for all 12 queries
(MRR 1.000). At a binary decision threshold, Q4 gained one correct answer on
paper statements (BF16 tied exactly at 0.5, first profile label wins) and lost
one on a bus serving an old clinic. The largest probability change was 0.2159
for Prism and 0.1004 for QEV. For every Prism example, native selected-row
and native full-vocabulary projections gave identical returned probabilities.
The BF16/native comparison includes both quantization and backend numerics.
These are small functional panels. Evidence generation, ASR correction quality,
and production retrieval corpora were outside these quality measurements.

[Prism model and prompt](https://huggingface.co/infgrad/Prism-Qwen3.5-Reranker-0.8B/tree/c60729783cb2f0662c0b054a02964402a211e3f4) ·
[QEV model](https://huggingface.co/labqoat/qev-0.8b/tree/9a095f35fe0e27a888cfed9eb1517718dc9dc59e) ·
[Evaluation inputs](../../tests/data/finetunes) · [Scores and probabilities](../results/finetune-serving-2026-11)

## Adapter import

A second end-to-end import used a deterministic nonzero rank-2 LoRA adapter on
Prism's `q_proj` layers. One command resolved the pinned Hub base, merged in
FP32, packed the sharded text-only export, started the server and compared
three examples against the merged FP32 reference. The adapted weight checksum
differed from the original Prism package and native and reference top choices
agreed on all three examples. This fixture exercises the adapter workflow.
The document relevance scores above are for the public Prism checkpoint.

[Adapter smoke evidence](../results/finetune-serving-2026-11/adapter-smoke.json)

## Concurrent QEV decisions

Intel Xeon Platinum 8573C, four vCPUs of quota, affinity mask 15, four native
inference threads. Each run started a local server and sent the same 16 distinct
60–71-token requests, maintaining 2, 4 or 8 outstanding clients. The prefix
cache was disabled. The measurements include HTTP, tokenization, waiting in
the server, inference and serialization. They exclude server startup. The
sequential benchmark runner used one warmup and three measured runs per
configuration, reversing case order on alternating runs. Throughput is the
median of three runs. p95 is the nearest-rank percentile of 48 client
latencies per configuration.

| Concurrent requests | Decisions/s, ordinary → shared | Throughput change | p95 latency, ordinary → shared |
| ---: | ---: | ---: | ---: |
| 2 | 3.040 → 3.103 | +2.1% | 875 → 822 ms |
| 4 | 2.678 → 3.050 | +13.9% | 2,485 → 1,652 ms |
| 8 | 2.613 → 1.494 | −42.8% | 4,560 → 6,339 ms |

All measured samples produced the same choices and probabilities to QEV API
precision (four decimals). Each grouped configuration recorded actual
multi-request prefill ticks. Four concurrent requests improved throughput in
all three paired runs. At eight, the shared path ranged from 1.45 to 3.22
decisions/s. Keep width 1 as the default. Width 4 is worth testing on an
installation serving several simultaneous cold decisions. These results do
not measure a single cached decision, text decode, or a second CPU family.

[Per-run timings and counters](../results/finetune-serving-2026-11/performance.json) ·
[Raw profiles](../results/finetune-serving-2026-11/profiles.json.gz) ·
[Commands, versions and hashes](../results/finetune-serving-2026-11/provenance.json) ·
[Import and reproduce](../finetunes.md)
