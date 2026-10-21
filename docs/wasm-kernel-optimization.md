# Local WASM Q4 kernel optimization

This continues the [direct-FMA optimization](wasm-fma-optimization.md) for the
same calibrated H128/Q4-G32-DOT4 checkpoint. The remaining native/browser gap
was not an established WebAssembly limit. Profiling found additional avoidable
work in the browser's Q4 matrix kernels. All changes and artifacts remain local.

## Execution changes

The direct-FMA diagnostic measured 1,729 ms of prefill on the long request,
including 1,064 ms in Q4 matrix kernels (about 62%). Q4 kernel wait time was
only about 17 ms, so worker synchronization was not the main cost in that
profile. After optimization, the same diagnostic measured about 586 ms in
Q4 kernels (47% of a 1,247 ms instrumented forward). Preparation time is
recorded separately; nested operation and stage
timings must not be added as independent costs.

The selected implementation changes three details:

- It converts eight FP16 block scales using SIMD instead of eight scalar
  conversion calls. A short path handles ordinary normalized scales; a full
  path preserves zeros, subnormals, infinities, signs and NaN payload bits.
- It finishes token tiles for one weight-row tile before moving to the next
  row, improving the opportunity to reuse those weights in small CPU caches.
- The guarded direct-FMA/bounded-DOT4 build processes eight prompt tokens together,
  followed by four-token and single-token tails. Other SIMD builds retain
  four-token tiles. Each output keeps the original block and FMA order.

No weights, checkpoint layout, calibration, H128 transforms, activation
quantization, probability temperatures or native inference kernels change.
There is no persistent expanded copy of the weight scales. Native benchmark
tooling adds `--session-warmup all` so both engines warm all three shapes;
the earlier `first` behavior remains its default.

Single-run probes also tried two-token tiles, a SIMD attention implementation
that retained the scalar reduction order, and a temporary per-worker scale
cache. They did not establish a benefit over the selected implementation and
were discarded. The early eight-token probe also included the discarded
attention implementation; its timing alone cannot isolate the tile-size
effect. Probe timings are retained as diagnostics, not final performance
statistics.

## Native versus browser performance

| Engine | Short routing, 74 tokens | Snake, 77 tokens | Long routing, 579 tokens |
| --- | ---: | ---: | ---: |
| Native AVX-512/VNNI, fresh | **37.95 ms** | **38.34 ms** | **248.68 ms** |
| WASM before these kernel changes | 223.42 ms | 224.18 ms | 1,741.87 ms |
| WASM SIMD scale conversion and eight-token tiles, fresh | **157.42 ms** | **157.59 ms** | **1,237.24 ms** |
| Current WASM/native latency ratio | 4.15× | 4.11× | 4.98× |

The new browser implementation takes **29.0–29.7% less time** than the preceding
WASM version on these inputs, corresponding to **1.41–1.42×** serial throughput.
The fresh native/browser comparison puts the remaining gap at approximately
**4.1× for the short requests and 5.0× for the long request**.

The fresh native and WASM rows use three measured process runs after one
process warmup, plus one in-session warmup of each shape. Both run serial B1
requests with fresh recurrent/KV state, 32-token chunks, eight engine threads,
context 2048 and process affinity mask 255 (CPUs 0–7). Hardware is the AMD
Ryzen 9 9955HX3D. Browser is Chromium 156.0.8078.4, built with Emscripten
6.0.13. Native uses the repository's AVX-512/VNNI backend. Model loading,
tokenization, state preparation and main-thread return are excluded from the
compute timings. No decode forwards or generated tokens are measured.

The previous-FMA row reuses the immediately preceding three-run series,
whose runtime was archived before these changes. It uses the same frozen
inputs, thread count, affinity and compute scope. It is not a simultaneous
fresh baseline. The current native/browser ratio uses only the fresh series.
Full worker latencies and all samples are retained separately.

This remains an implementation comparison on one CPU and browser, not a
universal WASM overhead factor. The native engine has wider AVX-512/VNNI
kernels; the browser executes SIMD128 instructions. Generated instructions,
register allocation, arithmetic conversion and cache behavior also matter.
These measurements do not establish the minimum possible browser/native gap
or promise that the native latency can be reached by further tuning.

## Quality and validation

The 18-request, 24-question local parity suite still evaluates 2,549 tokens.
All raw option logits and answers are bit-identical to the earlier direct-FMA
WASM bundle. Prompt IDs and usage match the native reference exactly. No
selected choice changes. Maximum absolute native probability difference is
still 0.0040 (0.40 percentage points), with maximum raw logit difference
0.05436945. This is backend parity against native H128/Q4-G32-DOT4, not a new
full evaluation-set or BF16 accuracy result.

The new SIMD half-conversion test checks every one of the 65,536 FP16 bit
patterns in mixed vectors against the scalar conversion, including signed
zeros, subnormals, infinities and NaN payloads. Existing FMA, H128, signed
integer DOT4, stride and tail checks remain enabled. All seven WASM variants passed these
checks and the DeltaNet tests; Q4 token counts include 9, 12, 15, 16 and 31 to
exercise combinations of eight-token, four-token and single-token tails.
All 17 native CTests passed. Real Chromium UI checks passed for inference,
cancellation, cache reuse after unloading, malformed editor JSON, responsive
layout, localhost-only resources and fallback when direct FMA is unavailable.
The maximum measured WASM heap remains 575,209,472 bytes (548.56 MiB), excluding
additional browser memory.

## Reproduction and evidence

Build and serve with [the local demo instructions](wasm-local.md), then run:

```sh
.cache/powershell/pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/qev-wasm-kernels-2026-10-21/matrix.json \
  -OutputDir benchmarks/qev-wasm-kernels-repeat \
  -Runs 3 -WarmupRuns 1 -Affinity 255
```

The native benchmark target must be rebuilt to accept `--session-warmup all`.
For numerical parity, use `benchmark-wasm.mjs` with the frozen request/native
reference files, `--mode fma-threads --threads 8 --checks true`, and the same
sequential runner. Diagnostic operation profiling is disabled in final speed
measurements.

Compact results, per-run samples and reproduction metadata are retained in
[performance.csv](results/qev-wasm-kernels-2026-10-21/performance.csv),
[quality.json](results/qev-wasm-kernels-2026-10-21/quality.json),
[candidates.json](results/qev-wasm-kernels-2026-10-21/candidates.json),
[profiling.json](results/qev-wasm-kernels-2026-10-21/profiling.json) and
[provenance.json](results/qev-wasm-kernels-2026-10-21/provenance.json).
Raw commands, profiles, outputs, source/binary/checkpoint hashes and UI
captures remain under ignored `benchmarks/qev-wasm-kernels-2026-10-21/`.
No build output or model weight is tracked or uploaded.
