# Local WASM FMA optimization

The initial browser port spent substantial time reproducing native FP32 fused
multiply/add through FP64 SIMD and exceptional-case fallbacks. This report
records a local optimization of the same calibrated H128/Q4-G32-DOT4 model.
It changes execution, not weights or calibration. Nothing has been uploaded.

## What changed

Optional operation/stage profiling is now available through
`qev_options(chunk, profile)` before model loading. It is disabled in normal
operation and in performance measurements. On the 579-token request, the
strict-FMA-emulation diagnostic attributed about 60% of prefill time to Q4
matrix operations and 22% to DeltaNet attention. Operation and stage records
overlap; they must not be summed as independent costs.

The new `fma` and `fma-threads` bundles use direct
`f32x4.relaxed_madd` in the isolated Q4 and DeltaNet kernels. They retain the
bounded integer DOT4 path. There is no global fast-math flag and no change to
the H128 transform, weight packing or activation quantization.

The [relaxed SIMD specification](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md)
permits multiply/add to be fused or unfused. The page therefore executes a
small arithmetic probe before choosing these bundles, and the worker repeats
the guard before loading one. The probe checks cancellation, overflow
cancellation, subnormal output and signed zero. Unsupported or unfused
implementations fall back to the earlier strict-emulation bundles. The probe
is a runtime compatibility check; the kernel and model checks below establish
the tested numerical behavior on this machine.

The demo now prefers guarded direct FMA and keeps the prior variants in its
runtime selector. Prefill chunks remain 32 tokens. Single-run candidate probes
at 64, 128 and 256 tokens did not improve the three-request timing enough to
replace that default. These probes are diagnostics, not three-run performance
results. Shorter chunks also preserve frequent cancellation opportunities.

## Performance

| Browser CPU engine | Short routing, 74 tokens | Snake, 77 tokens | Long routing, 579 tokens |
| --- | ---: | ---: | ---: |
| Custom H128/Q4-G32-DOT4, direct FMA | **223.42 ms** | **224.18 ms** | **1,741.87 ms** |
| llama.cpp Q8_0 | 318.44 ms | 327.79 ms | 2,456.81 ms |
| llama.cpp Q4_0 | 404.66 ms | 404.13 ms | 3,116.89 ms |
| Custom H128/Q4-G32-DOT4, strict FMA emulation | 417.70 ms | 422.71 ms | 3,214.28 ms |
| llama.cpp Q4_K_M | 447.53 ms | 446.41 ms | 3,389.75 ms |

Direct FMA reduces custom-engine latency by **45.8–47.0%** on these inputs,
corresponding to **1.85–1.89×** serial throughput. The optimized engine delivers
**1.79–1.81×** the llama.cpp Q4_0 throughput, **1.41–1.46×** Q8_0 throughput,
and **1.95–2.00×** Q4_K_M throughput in this test. Quality is checked separately
below; these quantization names do not imply equal accuracy.

Every row uses the browser CPU, eight engine threads and affinity mask 255
(CPUs 0–7), on an AMD Ryzen 9 9955HX3D. Browser: Chromium 156.0.8078.4;
compiler: Emscripten 6.0.13. Values are medians of three measured process runs
after one process warmup. Each engine also warms all three request shapes
once per process. All requests run serially with empty recurrent/KV state,
resident weights, no prefix reuse and zero decode forwards.

The reported scope is prefill plus final logit projection; it excludes model
loading, tokenization, state preparation and the return to the main thread.
Full worker times are retained separately. Our engine uses 32-token chunks;
llama.cpp uses automatic Flash Attention and 512-token microbatches. The same
unmodified upstream revision and GGUF files as the earlier comparison are
used. These three inputs do not establish a universal engine ranking.

The native AVX-512/VNNI advantage does not transfer automatically to SIMD128
WASM. Wider integer/floating-point vectors, generated machine code, memory
layout and thread scheduling can still affect the remaining native/browser
gap. This experiment isolates a concrete avoidable cost in the original port;
it does not attribute the entire remaining gap to WebAssembly itself.

## Quality and integration checks

The 18-request, **24-question** local parity suite covers 2,549 evaluated
tokens and compares against native H128/Q4-G32-DOT4. Direct FMA produced
**bit-identical raw option logits to the original strict-emulation WASM
bundle on all 24 questions**. Prompt IDs and usage match the native oracle,
and no selected choice changes. The maximum absolute probability difference
from native remains **0.0040 (0.40 percentage points)**; the maximum raw
logit difference remains 0.05436945. This is a backend parity suite, not the
complete qev evaluation set and not a new BF16 accuracy measurement.

In the Snake fixture, the browser probabilities remain `up=9.06%`,
`left=66.08%`, `right=24.86%`; native H128 gives 9.37%, 66.09%, 24.53%.
Both select `left`. The optimization adds no observed probability regression
on these fixtures.

All seven WASM bundles passed DOT4 signed-extreme/tail/stride tests and the
DeltaNet tests. SIMD FMA tests compare 40,000 random FP32 lanes against
`std::fma`, including exceptional values (NaN payloads are not compared).
All 17 native CTests passed after the changes. Guard tests reject mocked
unfused and unsupported implementations. The browser protocol checks cover
malformed input, invalid encoding, context overflow, cancellation and fresh
state after cancellation, plus two raw-JSON ordering fixtures.

Real Chromium UI checks passed for the new default runtime, routing and Snake
inference, responsive layout, cancellation, cache reuse after unloading,
malformed editor JSON and localhost-only resources. A mocked unsupported-FMA
browser selects the strict fallback and disables both direct-FMA choices.
The maximum observed custom WASM linear memory remains 575,209,472 bytes
(548.56 MiB); this excludes browser processes and other browser storage.

## Reproduction and evidence

Build and serve as described in [the local demo guide](wasm-local.md).
The normal build now emits seven bundles; use
`bash scripts/build-wasm.sh fma-threads` for only the fast threaded variant.
The build host must also provide fused arithmetic to pass that variant's
kernel tests. The standard bundles remain buildable without it.

```sh
node tests/wasm_capabilities_test.mjs
.cache/powershell/pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/qev-wasm-optimization-2026-10-21/matrix.json \
  -OutputDir benchmarks/qev-wasm-optimization-repeat \
  -Runs 3 -WarmupRuns 1 -Affinity 255
```

The matrix assumes the local Playwright installation and Chromium executable
recorded in it. For the complete parity test, use `benchmark-wasm.mjs` with
`--mode fma-threads --threads 8 --chunk 32`, the request/reference files in the
local demo guide, and the same sequential runner. The common comparison
driver accepts `--mode`, `--chunk` and diagnostic `--profile-ops true` for
custom-engine entries. Never enable diagnostic instrumentation for a final
speed comparison.

Compact results are retained in
[performance.csv](results/qev-wasm-optimization-2026-10-21/performance.csv),
[quality.json](results/qev-wasm-optimization-2026-10-21/quality.json),
[candidates.json](results/qev-wasm-optimization-2026-10-21/candidates.json) and
[provenance.json](results/qev-wasm-optimization-2026-10-21/provenance.json).
Raw profiles, commands, binary/checkpoint hashes, CPU affinity, numerical
outputs and UI captures remain under the ignored
`benchmarks/qev-wasm-optimization-2026-10-21/` directory. The original strict
baseline runtime was archived there before rebuilding. Build outputs and
weights remain outside Git.
