# CPU tuning for qev decisions on the Ryzen 9 9955HX3D

## Single requests

The lowest median latency in the 4/6/8-thread sweep came from eight inference
threads on physical V-Cache cores 0–7. For a dedicated qev service on this
laptop, use this server command.

```sh
taskset -c 0-7 ./build/qwen35_cpu_server \
  --model-dir models/qev --threads 8 --prefill-chunk 128
```

For deployments on another CPU, `--prefill-chunk auto` measures both schedules
on the loaded qev model before the server accepts requests. The server reports
the selected chunk and probe results through `GET /v1/batch-config`. A fixed
numeric setting skips chunk calibration. In an independent comparison on this
laptop the automatic mode selected 128 on all eight starts, including two
warmups. Compared with the numeric 128 setting, calibration added about 0.50 s
per process outside the timed HTTP interval. Calibration also warms the model
before the first real request. In the final release build, the first 74-token
request after readiness took 38.40 ms with adaptive chunks, 33.87 ms with
numeric 128, and 28.82 ms with the startup probe. These numbers include the
effects of that warmup.

The adaptive schedule processes a 74-token request as 64 + 10. A chunk of 128
processes the whole request in one forward. At 129 tokens and above both settings
use 128-token chunks. Token counts include the rendered decision instructions
and choices. The samples below include localhost HTTP, tokenization, queuing
and scoring. Model loading and server startup are outside the timed interval.

| Threads | 74 tokens adaptive → 128, ms | 579 tokens adaptive → 128, ms | 64 German tickets adaptive → 128, median ms |
| ---: | ---: | ---: | ---: |
| 4 | 56.45 → 50.75 | 279.47 → 279.16 | 96.38 → 96.44 |
| 6 | 44.03 → 39.85 | 204.85 → 205.54 | 70.78 → 71.31 |
| 8 | 38.27 → 33.81 | 160.57 → 162.95 | 56.27 → 56.18 |

Each process served either five fixed lengths or 64 distinct German
routing requests (178–216 input tokens). Each cell is the median of three
measured process runs. Fixed-length timings are reported for their position in
the five-request panel. For routing, each process contributes its median over
64 requests. The prefix cache was disabled throughout the sweep.

An independent panel repeated each input 20 times in shuffled order per
process. Each cell below covers 60 timed responses (three measured processes),
including the first response after readiness. At 74 tokens the fixed chunk
reduced median latency by 14.2%. At 127 tokens the fixed chunk improved the
median while its observed p95 was higher. At 129 and 579 tokens the two
settings execute the same chunk schedule.

| Input tokens | Adaptive p50, ms | Chunk 128 p50, ms | Adaptive p95, ms | Chunk 128 p95, ms |
| ---: | ---: | ---: | ---: | ---: |
| 74 | 27.47 | 23.56 | 28.66 | 25.10 |
| 127 | 40.91 | 39.44 | 42.61 | 52.78 |
| 128 | 38.32 | 35.40 | 39.70 | 37.21 |
| 129 | 40.82 | 40.73 | 43.97 | 45.00 |
| 579 | 164.73 | 164.51 | 167.17 | 167.14 |

## Concurrent requests

On the 64 distinct German tickets, width 2 improved throughput by 1.7% with
four active clients. Width 4 increased p95 latency. The table reports the
median group throughput and median per-run nearest-rank HTTP p95, including
time queued by the server. Each client sent its next request after receiving
the previous response. All rows used eight threads and adaptive chunks.

| Clients | Prefill width | Decisions/s | HTTP p95, ms | Shared scheduler steps per 64 requests |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1 | 17.63 | 117.9 | 0 |
| 2 | 2 | 17.82 | 119.5 | 32 |
| 4 | 1 | 17.58 | 234.8 | 0 |
| 4 | 2 | 17.87 | 231.6 | 32 |
| 4 | 4 | 17.85 | 244.0 | 16 |


The grouped implementation combines the linear-attention input projection
and the full-attention Q/K/V projection across ready requests. Each request
then reads its own slice of projected rows and updates its own attention,
DeltaNet and convolution state. The existing eligibility limit is 128 uncached
tokens remaining per request. Raising it to 256 in a separate prototype
increased median p95 from 118.0 to 137.6 ms for the two-client German panel.
The shipped scheduler uses the 128-token eligibility limit.

The second corpus used the 16 short, distinct English tickets from the earlier
Xeon study, cycling through them four times per process with the prefix cache
off. Their prompts span 60–71 tokens. Comparing the existing and grouped
input projection at the **same adaptive chunk and prefill width 2** isolates
the projection change. The separate fixed-chunk measurements show the combined
effect of the two settings.

The independently built shipping server and v0.3.2 produced these medians:

| Corpus | Clients | Prefill width | Chunk | Decisions/s, v0.3.2 → shipping | HTTP p95, ms, v0.3.2 → shipping |
| --- | ---: | ---: | --- | ---: | ---: |
| Short tickets | 2 | 2 | Adaptive | 43.17 → 44.38 | 66.0 → 62.1 |
| Short tickets | 2 | 2 | 128 | 47.87 → 48.12 | 45.8 → 60.4 |
| Short tickets | 4 | 2 | Adaptive | 44.13 → 45.46 | 122.1 → 117.5 |
| Short tickets | 4 | 2 | 128 | 48.55 → 48.65 | 86.6 → 84.7 |
| German routing | 4 | 2 | Adaptive | 17.83 → 18.01 | 234.6 → 230.8 |

With four clients on short tickets, v0.3.2 at width 1 and adaptive chunks
served **41.45 decisions/s**. The shipping server with `--prefill-requests 2
--prefill-chunk 128` served **48.65 decisions/s** (+17.4%). At matched
width 2 and adaptive chunks, the projection implementation itself contributed
3.0%. For two short-ticket clients with chunk 128, width 1 already served
47.99 decisions/s at 44.3 ms p95 in the width sweep. Width 2 offered little
extra throughput and its p95 depended on how many requests arrived together.

## AVX2 matrix kernel

Two prototypes shared each four-byte activation broadcast between adjacent
eight-row DOT4 weight panels. One processed two tokens at a time and the other
four. With eight threads and strict AVX2 dispatch, median full-model prefill
rates on the same Ryzen are shown below.

| Checkpoint | Input tokens | v0.3.2, tokens/s | Two-token prototype | Four-token prototype |
| --- | ---: | ---: | ---: | ---: |
| qev-0.8b | 512 | 1,737.15 | 1,698.24 | 1,784.32 |
| qev-0.8b | 4,096 | 1,477.64 | 1,447.09 | 1,514.62 |
| Qwen3.5-2B | 512 | 748.60 | 729.58 | 774.50 |
| Qwen3.5-2B | 4,096 | 696.35 | 679.05 | 719.05 |

The four-token prototype gained 2.5–3.5%. An independent series in reverse
case order confirmed gains of 2.55% and 3.05% for qev, and 3.58% and 3.37% for
2B (512 and 4,096 tokens respectively). GCC generated more YMM stack accesses
for the prototypes. Static disassembly of the AVX2 matrix function contained 0,
38 and 64 YMM stack-reference sites for baseline, two tokens and four tokens
respectively. Counts refer to distinct instructions in the disassembly. The two-token
version regressed on all four workloads.

Both prototypes passed 10,080 DOT4 kernel cases. For the four-token prototype
all four forced-step vocabulary logit vectors were byte-identical to v0.3.2 on
both checkpoints at both lengths. These are single-request model-forward
measurements on fixed tokens, with strict AVX2/FMA/F16C for the quantized
matrix dispatch. The laptop normally selects AVX-512/VNNI. The
[i7-8750H follow-up](dot4-pair4-i7.md) measures the four-token patch on an
AVX2-only CPU. EPYC 9V74 timings are pending. The implementation is retained
as an [applicable patch](../results/decision-tuning-2026-11-20/avx2-pair4.patch)
for measurements on an AVX2 deployment. To reproduce the paired build on an
i7-8750H or an AVX2 VM, build v0.3.2 in two separate checkouts with the same
compiler and flags. Apply the patch in the candidate checkout, build
`qwen35_cpu_bench` and `q4_dot4_test`, and run the latter before timing.
Replace the executable paths in the AVX2 confirmation matrix with these two
builds and set both the thread count and affinity for the available cores.

## Validation

Across the retained HTTP series, all **14,096 responses** (including warmups)
matched the reference answers, returned probabilities and usage for their
**85 distinct request bodies**. Every response reported zero generated tokens
and zero cached tokens. The separate HTTP regression checked 38 mixed fixtures,
16 parallel requests and an interleaved text completion against v0.3.2, forced
full projection and the selected-option readout. The model-backed scheduler
tests also compare unrounded option logits for variable-length segments with
prefix hits, prefix registration, contiguous KV and paged KV.

The regular AVX-512/VNNI path also ran 512- and 4,096-token prompts on the
0.8B, 2B and 4B base models. Shipping median prefill rates were 0.01–0.57%
below v0.3.2 across the six pairs. These single-request cases use private
attention state and have no cross-request projection to share. All four
full-vocabulary logit records matched v0.3.2 byte for byte on each of the six
inputs.
[Base-model samples](../results/decision-tuning-2026-11-20/base-prefill-samples.csv)

## Reproduction

Run the matrices in the [manifest](../results/decision-tuning-2026-11-20/manifest.json)
from the repository root. For example, `jq '.phases.production.cases' manifest.json`
extracts the shipping comparison. The manifest records the case arguments, model
and binary hashes, affinity, the runner command, and local raw-profile directories.
Recreate the fixed-length panels with the Python version recorded in the manifest.

```sh
python3 docs/results/decision-tuning-2026-11-20/make-fixtures.py benchmarks/decision-tuning-nov20
```

For AVX2 vocabulary checks, the validation CLI set `CpuLoadOptions::cpu_q8_backend`
to `avx2` in both executables and dumped logits with the four forced IDs
`303,22461,4843,57900`. The native base-model checks use the CLI's normal ISA
selection with the same forced IDs. The 512- and 4,096-token prompt files are
the ones named in the matrices.

Each matrix used the sequential runner, one warmup and three measured passes,
reversing case order on alternating passes. HTTP profiles were collected with
eight physical cores available, the same quantized model and the same native
selected-option head throughout. A preliminary projection run overlapped a local
indexing process and was excluded. The projection and repeated-length matrices
were restarted after that process became idle. The short fixed-length panels use the published
74- and 579-token ticket fixtures and synthetic boundary extensions. The German
panel is the frozen [routing input set](../results/decision-followup-2026-11-20/routing-de-inputs.jsonl).

[HTTP samples](../results/decision-tuning-2026-11-20/http-samples.csv) ·
[Per-length latencies](../results/decision-tuning-2026-11-20/latency-by-length.csv) ·
[AVX2 samples](../results/decision-tuning-2026-11-20/avx2-samples.csv) ·
[Output checks](../results/decision-tuning-2026-11-20/validation.json)
