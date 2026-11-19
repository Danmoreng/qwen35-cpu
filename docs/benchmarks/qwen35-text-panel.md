# Native Qwen3.5-0.8B and Qwen3.5-2B text tasks

The calibrated H128/Q4-G32-DOT4 models ran the same 24 tasks on a Ryzen 5 7600.
2B passed more Python edits and JSON extractions. 0.8B passed more German
corrections and had lower HTTP latency and peak memory use in this panel.

| Model | German correction | Python edits | JSON extraction | Total | HTTP p50 | HTTP p95 | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 3/8 | 1/8 | 1/8 | 5/24 | 189 ms | 357 ms | 518 MiB |
| Qwen3.5-2B | 1/8 | 3/8 | 2/8 | 6/24 | 458 ms | 1,842 ms | 1,131 MiB |

The panel measures strict compliance with the requested edits and response
format. German responses must match the target after Unicode and whitespace
normalization. Python answers must match the target AST and may use one code
fence. JSON answers must match the target object, with exact types and no
duplicate keys. These are eight examples per category. Equivalent rewrites
outside the stated constraints can fail the check.

The server used six pinned physical cores, one resident, batch width one,
greedy generation, a 256-token output limit, disabled thinking, and no prefix
cache. The sequential runner used a fresh server for every panel and ran one
warmup plus three measured passes per model. Outputs reproduced word for word
on all four passes. HTTP timing covers each local request after server startup.
The table reports the median of the three per-pass percentiles and high-water
RSS samples.

The [task definitions](../results/qwen35-text-panel-2026-11-19/panel.json)
and [outputs and run manifest](../results/qwen35-text-panel-2026-11-19/results.json)
include the prompts, scoring targets, exact responses, commands, revisions
and hashes. Use the two saved matrix entries with
`scripts/benchmark-inference-seq.ps1 -Runs 3 -WarmupRuns 1 -Affinity 63`
to repeat the run from the repository root. The
[2B instructions](../quantization.md#qwen35-2b) describe the calibrated
checkpoint and how to start the server with it.
