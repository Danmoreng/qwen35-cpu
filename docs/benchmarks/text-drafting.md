# Speculative text decoding with token lookup

On November 17, 2026, the native Qwen3.5-0.8B H128/Q4-G32-DOT4 engine ran four authored, coherent chat tasks on the Xeon Platinum 8573C KVM guest, three threads on CPUs 0, 2 and 4. They cover German transcript correction, revising an email, a Python rename and original prose. The German fixture is an illustrative transcript, not output from a recorded speech recognition system. The checkpoint is the base text model. Each request ran as a single stream with an EOS stop and a 128-token output cap.

The table shows median committed decode positions per second from three measured runs after one warmup. Each generated EOS, if present, is verified too. The first output comes from prefill. The sampler uses temperature 0.65, top-k 40, top-p 0.9, seed 83 and repetition penalty 1.05. Every mode emitted exactly the same token IDs as ordinary decode within each workload, including both sampled tasks. Free prose reaches the output cap.

| Task | Decoding | Off | N-gram | Input | Input accepted / proposed | Resident request ms, off / n-gram / input |
|---|---|---:|---:|---:|---:|---:|
| transcript | greedy | 32.4 | 55.0 | 51.5 | 46/49 | 2570 / 1615 / 1664 |
| editing | greedy | 30.7 | 46.2 | 57.4 | 44/45 | 2499 / 1799 / 1514 |
| editing | sampled | 26.3 | 41.7 | 47.6 | 44/45 | 2933 / 1934 / 1722 |
| code | greedy | 31.9 | 51.4 | 55.0 | 45/48 | 2504 / 1682 / 1699 |
| prose | greedy | 31.2 | 32.9 | 31.2 | 0/6 | 4249 / 4118 / 4286 |
| prose | sampled | 28.3 | 28.3 | 28.6 | 0/0 | 4670 / 4690 / 4723 |

Resident latency sums prompt processing and decode per observation before taking the median. Model loading, tokenization, HTTP and output rendering are excluded. These are small single-machine samples. Free-prose timings with few or zero proposals are sensitive to VM noise. Acceptance counts and all measurements are in the [results](../results/text-drafting-2026-11-17/performance.csv), [samples](../results/text-drafting-2026-11-17/samples.csv) and [manifest](../results/text-drafting-2026-11-17/manifest.json).

The general lookup searches prompt and generated history for the most recent prior trigram with a known continuation. Input mode follows the source cursor, can use a shorter local match to rejoin after an edit, and falls back to a trigram lookup in the source. Editing workloads supply their source document separately. For free prose, input mode uses the rendered prompt. Neither mode uses future reference output.

## Reproduction

Build the CLI and CPU benchmark in Release mode and use the pinned model from the manifest. The script renders the [prompts](../results/text-drafting-2026-11-17/prompts.json) and writes the matrix for the repository sequential runner. Run on the same guest with fresh output paths.

```sh
python scripts/prepare-draft-benchmark.py --cli build/qwen35_cpu --bench build/qwen35_cpu_bench --model-dir models/qwen3.5-0.8b --out benchmarks/draft-inputs
taskset -c 0,2,4 pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/draft-inputs/matrix.json -OutputDir benchmarks/draft-timing -WarmupRuns 1 -Runs 3 -Affinity 21
```
