# Qwen3.5-4B MTP depth for coding and tool calls

The native 4B H128/Q4-G32-DOT4 model and the H128 MTP sidecar ran through
the local Chat Completions API on a Ryzen 5 7600. The comparison uses
ordinary decode and MTP depths one through four. A depth-three round
verifies the three proposed tokens and the pending token with one target
forward over four causal rows. Depth four uses five rows. The profile
counts actual target batches by proposal count, including short rounds.

MTP drafts in a structured request use an independent copy of the JSON
grammar. The target sampler applies its live grammar as it selects the
verified output. The tool fixture runner checks the emitted function name
and typed arguments before returning an authored fixture result. Tool
results are simulated. Code edits are scored by comparing Python ASTs.

## Results

Each cell sums the request latencies in that panel and reports the median
over three measured passes. The prose panel contains one greedy and one
sampled request.

| Draft depth | Ten Python edits | Seven tool workflows | Two prose prompts |
| --- | ---: | ---: | ---: |
| Off | 16.69 s | 15.49 s | 8.95 s |
| 1 | 13.90 s | 14.57 s | 8.94 s |
| 2 | 12.30 s | 13.80 s | 10.65 s |
| 3 | 9.89 s | 12.50 s | 10.61 s |
| 4 | 10.80 s | 12.44 s | 13.25 s |

Depth three cut Python latency by 40.7%. Effective throughput including
HTTP and prefill rose from 18.93 to 31.94 output tokens/s. The model accepted
229 of 252 proposals over 84 three-proposal target forwards. Depth four
accepted 242 of 280 over 70 four-proposal forwards and had higher total
latency. All five modes passed nine Python edits. The remaining response
contained an invalid signature and call in the keyword-argument task.

The tool panel completed six of seven workflows. All seven expected function
calls used the correct function and arguments. The read/check workflow used
two simulated tools and correctly emitted `PASS`. Its median time was 4.29 s
without drafting, 3.67 s at depth three and 3.55 s at depth four. The incomplete
workflow read the correct source and repeated the call as text on the final
turn, leaving the requested edit incomplete.

Depth three cut total tool latency by 19.3% and accepted 124 of 134 proposals.
The target made 41 full three-proposal forwards and seven shorter forwards.
Depth four cut tool latency by 19.7% and accepted 133 of 145. Its median
advantage over depth three was about 59 ms across 11 HTTP requests.
The response text, finish reasons, tool arguments and token usage matched
ordinary decode across all depths and passes.

## Measurement

The ten Python edits combine the eight frozen
[Python text-panel cases](../results/qwen35-text-panel-2026-11-19/panel.json)
with two longer module edits. The seven tool cases include function routing,
typed arguments, direct answers under automatic selection, and a two-call
read/check sequence. Two additional prompts cover free greedy and sampled
prose. The sampled cases use temperature 0.65, top-k 40, top-p 0.9,
repetition penalty 1.05 and seed 83. The rest use temperature zero.

A fresh server ran each pass with one resident request, six threads on
physical CPUs 0–5, paged target KV and prefix caching disabled. The
sequential runner alternated the cases over one warmup and three measured
passes. Reported times measure the complete local HTTP request, including
prefill and response handling. Server startup and fixture handling occur
outside the timer. A multi-turn case sums the times of its requests.

The [fixture definitions](../results/qwen35-4b-mtp-coding-2026-11-19/fixtures.json)
record the mock tool results and expected replies. The
[manifest](../results/qwen35-4b-mtp-coding-2026-11-19/manifest.json)
and [samples](../results/qwen35-4b-mtp-coding-2026-11-19/samples.csv)
retain the matrix, hashes, measured samples and per-fixture outcomes.
Run the matrix through `scripts/benchmark-inference-seq.ps1` from the
repository root using `-Runs 3 -WarmupRuns 1 -Affinity 63`.
