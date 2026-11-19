# Native Qwen3.5-4B on Ryzen 5 7600

The [pinned Qwen3.5-4B snapshot](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a)
was packed as a native H128/Q4-G32-DOT4 text checkpoint with unweighted MSE16.
The packer wrote and checksum-verified 306 tensors. The 2B comparison uses
the existing calibrated H128/Q4-G32-DOT4 checkpoint.

## CPU throughput and memory

GCC 16.2.1 Release, six physical Ryzen 5 7600 cores pinned to CPUs 0–5 with
FP16 KV. The sequential runner alternated case order across one warmup and
three measured runs. Rates and peak RSS are medians. The 512-token cases use
128 forced output tokens, with 127 decode forwards per request. Transfers and
model loading are outside the throughput timers. B16 rates are aggregate.

| Scenario | 2B prefill tok/s | 4B prefill tok/s | 2B decode tok/s | 4B decode tok/s | 2B peak RSS | 4B peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1, 512-token prompt | 766.1 | 282.8 | 48.92 | 21.93 | 1,083 MiB | 2,396 MiB |
| B16, 512-token prompts | 758.5 | 296.5 | 363.13 | 156.08 | 1,535 MiB | 3,504 MiB |
| B1, 4,096-token prompt | 685.0 | 265.2 | | | 1,124 MiB | 2,504 MiB |

The unmodified engine and 4B-enabled engine also ran the same 2B checkpoint
in this series. B1 decode moved from 49.31 to 48.92 tok/s (−0.80%), B16 decode
from 361.45 to 363.13 tok/s (+0.46%), and 4,096-token prefill from 684.76 to
684.97 tok/s (+0.03%).

## Text tasks

The native server ran the same [24 strict tasks](qwen35-text-panel.md) in
one-request, greedy, non-thinking mode. The output text was identical across
all four passes for each model.

| Model | German correction | Python edits | JSON extraction | Total | HTTP p50 | HTTP p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-2B | 1/8 | 3/8 | 2/8 | 6/24 | 460 ms | 1,838 ms |
| Qwen3.5-4B | 6/8 | 7/8 | 3/8 | 16/24 | 876 ms | 1,479 ms |

Each latency is the median of three measured per-pass percentiles. The 2B
outputs included a response at the 256-token cap. The longest 4B response
was 89 tokens.

## BF16 reference

The independent CPU reference used the pinned
[llama.cpp revision](https://github.com/ggml-org/llama.cpp/commit/73a43d1f69345aee8bb186ef4b3172cef892f2e5)
with a BF16 text export from the same Qwen snapshot and its MTP head excluded.
Each engine received the same prompt and forced continuation for each window.
Scores use all 248,320 logits per position.

| Window | Prompt positions | Scored positions | BF16 PPL | H128 PPL | KL from BF16 | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English continuation | 512 | 128 | 13.6030 | 11.6763 | 0.248515 | 85.9% |
| German editing source | 32 | 35 | 15.7253 | 16.8934 | 0.048564 | 85.7% |
| German transcript source | 32 | 34 | 14.5324 | 15.3914 | 0.038343 | 88.2% |
| Python code source | 32 | 41 | 2.7811 | 3.0539 | 0.022635 | 92.7% |
| Combined | | 238 | 10.6716 | 10.1788 | 0.150174 | 87.4% |

Combined perplexity exponentiates the pooled mean target NLL. The English
window drives the lower pooled H128 perplexity and has the highest divergence,
including a maximum single-position KL of 13.12. These values describe
fidelity on four fixed windows.

## Runtime parity

4B batch sizes 4, 8, 12, 16, 20 and 32 reproduced every tested single-request
full-vocabulary logit bit for bit. The chunk and prefix-restore tests also
matched bit for bit with both contiguous and paged FP16 KV. 4B speculative
verification agreed with ordinary generation for greedy and seeded sampling,
including partial acceptance, stops and backpressure. The 0.8B and 2B
checkpoints each reproduced their earlier-build full logit dumps over the same
512-token English prompt and 128 forced continuation positions.

The [throughput and task samples](../results/qwen35-4b-ryzen-2026-11-19/performance.json)
and [BF16 reference](../results/qwen35-4b-ryzen-2026-11-19/reference.json)
record the benchmark matrices, source revisions, commands and artifact hashes.
The [reference tokens](../results/qwen35-4b-ryzen-2026-11-19/reference-windows.json)
are the prompt and forced continuation used in each quality window.

To reproduce the timing series, save `.throughput.matrix` from the performance
JSON as a standalone matrix, build the indicated executables and run
`scripts/benchmark-inference-seq.ps1 -Runs 3 -WarmupRuns 1 -Affinity 63`
from the repository root with that matrix and a new output directory.
The corresponding server matrix is at `.text_panel.matrix`.
