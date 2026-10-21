# Local browser decision engine

This is a local development build for qev-0.8b with the calibrated
**H128/Q4-G32-DOT4** checkpoint. It runs the repository's C++ inference engine
on the browser's CPU. No Hugging Face Space or inference service is created
by the build or serving scripts.

## Start the existing local build

From the repository root:

```sh
python3 scripts/serve-wasm.py
```

Open `http://127.0.0.1:8765`, load the model, and select an example or edit
the System One request JSON. The server binds to loopback and serves static
files and a fixed model-file allowlist. All model inference happens in a
browser worker. Stop the server with Ctrl+C.

The default model directory is `dist/qev-h128-public-final`. Override it with
`--model-dir PATH`; it must contain the existing published checkpoint named
`model.q35h` and its configuration/tokenizer files. The server verifies the
checkpoint SHA-256 before serving it:

```text
e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69
```

The checkpoint is 424,964,864 bytes. Including tokenizer/configuration files,
the initial local model transfer is 447,872,409 bytes. No weights are embedded
in JavaScript, WASM, or Git. The browser cache is controlled by the cache
checkbox; uncheck it to load directly from localhost without retaining
a model cache. "Unload engine" terminates the worker and its thread pool,
allowing another runtime to be selected. "Clear cache" removes only this demo's
OPFS directory and Cache API namespace.

## Build

The locally tested compiler is **Emscripten 6.0.13**, SDK toolchain
`eb132757325583b787fe9414b09afea9442b72ad`. Install it in a local SDK directory
or use the existing ignored `.cache/emsdk` installation. No global shell
configuration needs to change.

```sh
source .cache/emsdk/emsdk_env.sh
bash scripts/build-wasm.sh
```

This builds and tests five runtime bundles under ignored `dist/wasm`:

| Build argument | CPU execution | Requirements |
| --- | --- | --- |
| `scalar` | Portable C++, one thread, no compiler SIMD vectorization | WebAssembly |
| `simd` | SIMD128 DOT4, H128 and DeltaNet, one thread | WASM SIMD128 |
| `threads` | Same standard SIMD kernels, 1–8 engine threads | SIMD128 and shared memory |
| `relaxed` | Optimized bounded integer DOT4, one thread | SIMD128 and relaxed integer dot products |
| `relaxed-threads` | Optimized bounded integer DOT4, 1–8 engine threads | Relaxed integer dot products and shared memory |

Pass a build argument to rebuild only that variant. Build all variants before
using the default demo, which chooses the first supported runtime. The page
uses small `WebAssembly.validate` probes to detect SIMD and relaxed dot support.
Thread variants require `crossOriginIsolated`; the local server sets COOP/COEP.
Single-thread variants have separate binaries and do not require shared memory.

The build copies project/dependency licenses alongside the local web bundle.
Dependencies remain pinned in CMake. Builds and model files remain ignored.

## Architecture and numerical behavior

`web/app.js` sends a raw JSON request to `web/worker.js`. Raw strings preserve
the input order of numeric-looking question IDs and option names; parsing and
re-serializing JavaScript objects would reorder these keys. The worker also
accepts ordinary JavaScript objects, whose order follows JavaScript semantics.

The worker loads an Emscripten module and mounts readonly model Files/Blobs
through WORKERFS. With caching enabled, large files are streamed to OPFS;
Cache API storage is a fallback. This avoids placing another checkpoint-sized
copy in MEMFS. The C++ loader verifies each tensor checksum, loads owned packed
weights, and closes the file mount. Configuration/tokenizer files also receive
browser SHA-256 checks. The model's SHA-256 is verified by the local server;
the browser does not allocate an additional 425 MB ArrayBuffer to hash it.

`src/wasm/decision.cpp` exposes the single-owner C ABI declared in
`include/qwen35x/runtime/browser_decision.h`:

- `qev_load(directory, threads, context)` and `qev_unload()` manage one model.
- `qev_begin(json)` validates and prepares a System One decision.
- `qev_step()` executes one bounded prefill chunk, or finishes a question.
- `qev_cancel()` releases the current request's state.
- `qev_debug()` returns diagnostic prompt IDs and raw option logits.

Results and errors are JSON strings owned by the module until the next API
call. Calls must be serialized. Each question owns fresh recurrent/KV state;
there is no prefix reuse or simultaneous request batching. The engine uses
32-token prefill chunks. The worker yields through MessageChannel between
chunks so cancellation can be handled without nested timer delays. Model
loading can be aborted by unloading/terminating the worker.

Prompt formatting, tokenizer, option ordering, temperatures, and answer
formatting use the existing native implementation. Choice, noul/bool, score,
joint and per-question encoding are supported. One forced internal placeholder
allows the scheduler to finish the readout without a decode forward; the API
reports zero generated tokens.

The WASM kernels live in isolated translation units. Native x86 dispatch and
AVX translation units remain separate. SIMD modules are selected before
instantiation, and their internal backend dispatch preserves a scalar reference.

The optimized integer DOT4 path supplies unsigned weights in **0..15** as the
7-bit operand of the relaxed-dot instruction. Activations are signed int8.
Adjacent products cannot overflow int16, and the `8 * activation_sum`
correction restores the signed Q4 values. Consequently this instruction has
an exact integer result for our inputs; its implementation-dependent cases
are excluded. See the [WebAssembly relaxed SIMD specification](https://github.com/WebAssembly/relaxed-simd/blob/main/proposals/relaxed-simd/Overview.md).

Floating-point relaxed multiply/add is deliberately unused. FP32 FMA is
implemented through SIMD FP64 products/sums, with standard-library fallback
for double-rounding ties and subnormal results. DOT4 and DeltaNet use this
path to reduce drift from the native engine's fused arithmetic. H128 preserves
the portable butterfly order and produces bit-identical packed activations
in the kernel tests. Native and browser engines still have different reduction
and math-library implementations; compare probabilities as well as top choices.

## Reproduce correctness and performance checks

Create the native browser oracle with the opt-in native build target:

```sh
cmake -S . -B build-linux -DQWEN35_BUILD_WASM=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux --target qev_browser_reference systemone_protocol_test --parallel 4
build-linux/qev_browser_reference dist/qev-h128-public-final configs/qev-wasm-check.jsonl > .cache/qev-wasm-reference.jsonl
build-linux/systemone_protocol_test dist/qev-h128-public-final < configs/qev-wasm-protocol.jsonl > .cache/qev-wasm-protocol-reference.jsonl
```

`scripts/benchmark-wasm.mjs` drives a real Chromium browser through Playwright.
It accepts `--playwright-module PATH`, optional `--browser PATH`, `--mode`,
`--threads`, `--requests-file`, and the runner-supplied `--profile-json`.
Use `--references-file` to compare prompt IDs, raw logits, answer probabilities,
usage and selected options. `--checks true` exercises malformed input, context
overflow, cancellation and fresh state after cancellation. Add `--protocol-file`
and `--protocol-reference` for the raw JSON order fixtures. The numerical gate
is at most 0.005 absolute probability drift and no changed selected options;
prompt token IDs must match exactly. Failures preserve numerical profiles.

For performance, supply this script as a matrix entry to
`scripts/benchmark-inference-seq.ps1` with Node as the executable. Keep its
default three measured runs and one process warmup. Each browser instance also
warms the first request once after model readiness; loading/download time is
excluded. The frozen performance input is `configs/qev-wasm-bench.jsonl`:
short support routing, Snake, and long support routing. Every request runs
serially, with fresh state and zero decode forwards. This measures B1 decision
latency and input-token throughput, not generated tokens per second.

Profiles retain per-request outputs, token IDs, option logits, worker timing,
WASM heap size, browser version, model/bundle/source hashes and available Linux
process affinity/RSS diagnostics. The sequential runner preserves commands,
input hashes and run order. Model load is excluded from decision measurements.

`tests/wasm_browser_ui.mjs` checks real UI inference, desktop/mobile layout,
cancellation, unloading, reuse of the browser model cache and malformed editor
JSON. Its request audit requires every resource to come from localhost.

The local parity suite is **not** a rerun of the complete qev evaluation set.
The native model's published benchmark accuracy is not automatically a measured
WASM accuracy. Mobile browsers and other browser engines need separate
performance/compatibility measurements before publication. WASM heap size is
not total browser RAM; browser processes, cached Files/Blobs and runtime metadata
consume additional memory. The module currently caps linear memory at 2 GiB.

## Local results, 2026-10-21

Measured on an AMD Ryzen 9 9955HX3D in Chromium 156.0.8078.4 with
Emscripten 6.0.13. All cases used CPU affinity 0–7 (mask 255), the same
checkpoint and frozen requests. These are median resident-model B1 latencies
from three measured process runs after one process warmup. Each process also
warmed the first request once. Requests had fresh state and ran serially.
The three input lengths were 74, 77 and 579 evaluated tokens respectively.

| Engine | Threads | Short routing | Snake | Long routing |
| --- | ---: | ---: | ---: | ---: |
| Native AVX-512/VNNI | 8 | 38.27 ms | 36.98 ms | 243.41 ms |
| WASM bounded integer dot | 8 | 426.16 ms | 443.91 ms | 3,208.43 ms |
| WASM bounded integer dot | 4 | 772.14 ms | 784.40 ms | 5,976.67 ms |
| WASM bounded integer dot | 1 | 2,680.58 ms | 2,767.13 ms | 20,840.74 ms |
| WASM standard SIMD128 | 1 | 3,947.46 ms | 4,073.98 ms | 31,032.96 ms |

The current eight-thread browser port takes **11.14×, 12.00× and 13.18×**
the native time on these inputs. Equivalently, serial request throughput is
91.0–92.4% lower. This measures this implementation against the native
AVX-512/VNNI engine on this CPU; it is not a universal WASM overhead factor.
The browser port uses SIMD128 and software FP32 FMA emulation. Their individual
contributions to the gap have not been isolated by profiling.

The bounded integer-dot kernel reduces single-thread latency by about 32–33%
versus standard SIMD128. Going from four to eight threads reduces latency by
43–46% on this CPU. The demo therefore defaults to eight threads on machines
reporting at least 16 logical CPUs, four for at least eight, two for at least
four, and one otherwise. This heuristic has not been benchmarked on other CPUs.
The portable scalar bundle passed a separate 35-token functional fixture;
the full scalar timing suite exceeded its per-request timeout and is excluded
from the comparison.

The native comparison uses `qev_browser_reference` with eight threads and the
same 32-token chunked C ABI used by the browser. The measured region includes
input preparation, execution and parsing result/debug JSON. The browser also
yields between chunks; returning the result to the page and model loading are
excluded. The native and browser performance matrices ran sequentially, with
no overlapping inference measurements. Their commands are preserved in
[provenance.json](results/qev-wasm-local-2026-10-21/provenance.json).

For a native matrix entry, use `build-linux/qev_browser_reference` as the
executable and these arguments:

```json
["dist/qev-h128-public-final", "configs/qev-wasm-bench.jsonl",
 "--threads", "8", "--weights", "dist/qev-h128-public-final/model.q35h"]
```

The sequential runner supplies `--profile-json`. Use the same three measured
runs, one warmup and affinity 255 for both engines.

Quality is measured separately from speed. The optimized eight-thread browser
bundle passed the 18-request, **24-question** local suite (2,549 evaluated
tokens). Prompt IDs and usage matched the native engine exactly. No selected
choice changed, and the largest absolute probability difference was **0.0040
(0.40 percentage points)**. The largest raw option-logit difference was
0.05437. This comparison is against native H128/Q4-G32-DOT4, not BF16.

| Snake option | Native H128/Q4-G32-DOT4 | Browser H128/Q4-G32-DOT4 |
| --- | ---: | ---: |
| up | 9.37% | 9.06% |
| left | 66.09% | 66.08% |
| right | 24.53% | 24.86% |

Both engines selected `left`. The maximum observed WASM heap was 575,209,472
bytes (548.56 MiB), excluding additional browser memory. All 17 native CTest
checks passed, as did kernel/DeltaNet tests for all five WASM bundles. Real UI
checks passed for inference, cancellation, cache reuse after unload, malformed
JSON, localhost-only resource loading and responsive layout. A mobile viewport
check does not establish mobile-browser compatibility.

Machine-readable results are in
[performance.csv](results/qev-wasm-local-2026-10-21/performance.csv),
[quality.json](results/qev-wasm-local-2026-10-21/quality.json) and
[provenance.json](results/qev-wasm-local-2026-10-21/provenance.json).
Raw profiles, outputs, commands, hashes and screenshots remain locally in
ignored `benchmarks/qev-wasm-local/`; neither the implementation nor these
WASM artifacts have been published.

For the direct CPU/WASM comparison against llama.cpp Q8_0, Q4_0 and Q4_K_M,
plus Transformers.js / ONNX Runtime, see
[browser engine comparison](wasm-browser-comparison.md). Those measurements
use per-shape warmups and retain both compute and worker timing scopes.
