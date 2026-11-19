# Adaptive token drafting on Ryzen 5 7600

The calibrated native Qwen3.5-2B H128/Q4-G32-DOT4 checkpoint ran the existing
German transcript, editing, Python edit and prose fixtures. Input drafting
received the source document for the three copy tasks. The prose tasks supplied
only the rendered prompt. The synthetic conflicting-source cases used the
ordinary prose outputs with a decoy token inserted after each pair of tokens
to exercise repeated rejections.

The verifier now pauses draft lookup for 16 generated tokens when fewer than
one third of a window of at least 12 proposed tokens are accepted. It then
resumes lookup in the same request with fresh counters. A preliminary warmup
with a six-token window triggered a pause on the Python n-gram case and
reduced accepted tokens from 46 to 37. The longer window protects that case
and still bounds verifier work on the conflicting sources.

## Results

| Task | Decoding | Ordinary decode | Input drafting before | Adaptive input drafting | Adaptive accepted / proposed |
| --- | --- | ---: | ---: | ---: | ---: |
| German transcript | greedy | 1,517 ms | 737 ms | 705 ms | 44/49 |
| German editing | greedy | 1,423 ms | 630 ms | 626 ms | 44/45 |
| German editing | sampled | 1,546 ms | 745 ms | 741 ms | 44/45 |
| Python edit | greedy | 1,587 ms | 680 ms | 687 ms | 49/51 |
| Prose | greedy | 2,603 ms | 2,622 ms | 2,601 ms | 1/3 |
| Prose | sampled | 2,921 ms | 2,910 ms | 2,915 ms | 0/3 |
| Prose with conflicting source | greedy | 2,603 ms | 2,873 ms | 2,702 ms | 1/67 |
| Prose with conflicting source | sampled | 2,921 ms | 3,158 ms | 2,981 ms | 1/67 |

The authored cases triggered zero cooldowns in both input and n-gram mode.
The Python n-gram edit accepted 46 of 54 proposals in both builds and moved from 744 to
745 ms. Each conflicting source triggered five cooldowns and reduced attempted
verification rounds from 63 to 23. Resident latency with the conflicting
source fell by 6.0% for greedy and 5.6% for sampled decoding.

Every mode and measured run emitted the exact ordinary-decode token sequence
for its task and sampling configuration. Model-backed tests exercised the
cooldown through actual rejections with 0.8B, 2B and 4B. They also verified
full acceptance, rejection rollback, stop sequences, backpressure and seeded
sampling with both contiguous and paged KV.

## Method

The Release builds used GCC 16.2.1, six physical cores on CPUs 0–5, one
resident request, an EOS stop and a 128-token output cap. Seeded sampling
used temperature 0.65, top-k 40, top-p 0.9 and repetition penalty 1.05
with seed 83. The sequential runner alternated order over one warmup and
three measured runs. Resident time sums prompt processing and decode in
each measurement before taking the median. Loading, HTTP, tokenization and
rendering are outside these timers.

The [run manifest](../results/qwen35-2b-adaptive-draft-2026-11-19/results.json)
records exact command lines, per-run measurements, token-sequence hashes and
build and input hashes. Prepare the authored inputs with
`scripts/prepare-draft-benchmark.py` and restore the synthetic source CSVs
from [the saved fixtures](../results/qwen35-2b-adaptive-draft-2026-11-19/stress-fixtures.json).
Save the manifest's `.paired.matrix` as a standalone matrix and run the
sequential benchmark from the repository root with
`-Runs 3 -WarmupRuns 1 -Affinity 63`.
