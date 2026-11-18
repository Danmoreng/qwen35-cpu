# Native Batches API on the Xeon VM

The native H128/Q4-G32-DOT4 server corrected 16 individually authored German transcript-style sentences in one OpenAI Batch job on November 18, 2026. Each request used greedy generation, a shared correction instruction, a 48-token cap and natural EOS. The comparison keeps the thread count and file API fixed and varies concurrent batch rows. Every run produced the same outputs across all 16 IDs.

| Batch width | Threads | Seconds per job | Requests/min | Aggregate output tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 4 | 12.80 | 75.0 | 20.5 |
| 8 | 4 | 8.46 | 113.4 | 31.0 |

Values are medians from three sequential measured runs after one warmup. Each timed job includes file upload, submission, prefill, decode, polling and result download. The resident server performed one warmup request with the same instruction before the timer. Model loading and CPU calibration occur outside the timed interval. Reported output tokens include EOS. The wider batch processed these sentences at 1.51 times the serial throughput.

The guest exposed five logical CPUs and a cgroup quota of four. A separate automatic startup calibration chose 4 inference threads and width 8. Both timed configurations explicitly fixed the selected thread count. Performance depends on prompt lengths, generation settings and other CPU load. These are text-correction timings. The qev decision path has a separate calibration and correctness test.

[Workload and client](../../scripts/benchmark-batches.py) · [Samples](../results/batch-api-2026-11-18/samples.csv) · [CPU probe, hashes and runner matrix](../results/batch-api-2026-11-18/manifest.json). To reproduce, adapt the executable and model paths in the matrix and run `scripts/benchmark-inference-seq.ps1` with one warmup and three measured runs. The client requires the OpenAI Python SDK.
