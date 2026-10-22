# Native decision HTTP readout, 2026-10-22

The qev HTTP server now projects only the option-letter logits required by each
question, using the engine's restricted DOT4 readout. Previously it projected
the entire vocabulary and discarded almost all values. The final hidden vector,
selected logits, temperatures, probabilities and checkpoint are unchanged.
This integrates the previously measured browser/native C ABI optimization into
the actual HTTP service; these results supersede the
[previous HTTP measurements](qev-systemone-api-2026-10-21.md).

## Behavior and validation

`--decision-head auto` is the default. The model exposes its readout capability,
and unsupported output formats (including Q8 heads) automatically retain full
projection. `--decision-head full` forces the reference implementation. Normal
completion sampling and vocabulary callbacks are unchanged. No continuation is
generated: decisions still use zero decode forwards and zero output tokens.

The native default prefill chunk remains adaptive: 64 tokens for prompts of
at most 128 tokens, otherwise 128. `--prefill-chunk N` provides an explicit
1..2048-token override. The 32-token browser C ABI default is a separate setting.
Chunk 128 results below do not represent an additional kernel change.

- All 17 native CTests pass.
- Forty protocol fixtures match upstream tokenization/formatting.
- Nineteen smoke questions have zero rounded probability delta to the original
  quality study; this is regression evidence, not a repeat of its full scoring.
- Previous server, forced full head and automatic selected head have identical
  HTTP answers and usage on 38 fixtures each: both encodings, mixed question
  types and up to 255 options.
- Sixteen parallel HTTP calls match serial answers; interleaved completion and
  decision requests recover correctly. Q8-head tests verify automatic full-head
  dispatch and exact answer parity.
- Every fixed-work benchmark response matches across previous/current servers
  and concurrency levels, including all 16 distinct inputs.

The new standard-library HTTP regression test and pinned model downloader also
run in Linux and Windows release CI.

## Current HTTP latency

AMD Ryzen 9 9955HX3D, GCC 16.2.1 Release, native AVX-512/VNNI, eight physical
V-cache cores 0–7 (affinity mask 255), FP16 KV. One warmup and three measured
process runs, sequentially through `scripts/benchmark-inference-seq.ps1`.
Each sample starts a fresh server and measures its first B1 decision after
readiness. Loading is excluded; localhost HTTP, tokenization, scheduling and
answer formatting are included. No resident-model warmup or prefix-cache credit.
llama.cpp is CPU-only at `9b56ccf00d755b0bf780140f6352a67db1ac0378`.

| Server / checkpoint | Short, 74 tokens | Long, 579 tokens |
|---|---:|---:|
| Previous native, H128/Q4-G32-DOT4 | 48.46 ms | 206.58 ms |
| **Current native, same checkpoint, adaptive chunk** | **44.38 ms** | **204.09 ms** |
| Current native, same checkpoint, explicit chunk 128 | 40.02 ms | 204.41 ms |
| llama.cpp CPU / BF16 | 77.04 ms | 541.14 ms |
| llama.cpp CPU / Q8_0 | 92.56 ms | 828.73 ms |

At unchanged defaults, selected readout reduces short-request latency by about
8.4% and long-request latency by 1.2% in this series. Explicit chunk 128 reduces
the current short median by another 9.8%; its long difference is effectively
neutral. Defaults remain unchanged. The previous short-server samples are
57.72/48.46/46.79 ms; current samples are 44.38/44.09/45.51 ms. Three runs do
not establish a general speedup distribution across workloads or machines.
These are decision latencies, separate from the base model's decode tokens/s
and from quantization quality. No model quality gain is claimed.

## Same work, serial versus concurrent

Each sample sends the same 16 distinct state strings, serially or with 16
simultaneous calls, to one fresh server. These inputs have 76/581 tokens.
Throughput covers the whole group; latency is the mean per-request HTTP time.

| Native server | Input | Concurrent calls | Decisions/s | Mean HTTP latency |
|---|---|---:|---:|---:|
| Previous | Short | 1 | 28.72 | 34.78 ms |
| Current | Short | 1 | 31.08 | 32.13 ms |
| Previous | Short | 16 | 26.87 | 507.57 ms |
| Current | Short | 16 | 29.03 | 486.16 ms |
| Previous | Long | 1 | 5.08 | 196.64 ms |
| Current | Long | 1 | 5.21 | 191.90 ms |
| Previous | Long | 16 | 4.99 | 2965.75 ms |
| Current | Long | 16 | 5.11 | 2911.85 ms |

The readout improvement carries through to the fixed-work test. Concurrency
still provides no useful throughput gain: the scheduler time-slices prefill
rather than batching it across requests. Do not compare aggregate throughput
with a B1 first-request latency or reinterpret it as faster individual calls.

## Reproduction and provenance

```sh
python tests/systemone_readout_parity.py --server build-linux/qwen35_cpu_server \
  --reference-server benchmarks/qev-http-selected-2026-10-22/server-before \
  --model-dir dist/qev-h128-public-final --weights dist/qev-h128-public-final/model.q35h \
  --output benchmarks/qev-http-selected-2026-10-22/wire-parity.json
.cache/powershell/pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/qev-http-selected-2026-10-22/http-matrix.json \
  -OutputDir benchmarks/new-http-run -Runs 3 -WarmupRuns 1 -Affinity 255
```

Use `concurrency-matrix.json` for the fixed-work test and adapt local executable
paths when moving to another machine. Use fresh output paths. The prior binary
was built from `9967619`; the current
server was built from `6cbfdf9`. Compact results, all sample latencies, commands,
binary/checkpoint/input hashes, source revisions and test logs are in
[`results/qev-http-selected-2026-10-22`](results/qev-http-selected-2026-10-22/summary.json).
Raw profiles remain under ignored `benchmarks/qev-http-selected-2026-10-22`.
