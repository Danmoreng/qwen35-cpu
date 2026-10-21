# qev native System One API and CPU serving

The native HTTP server implements the text-only `POST /v1/systemone` protocol
used by qev-0.8b and current llama.cpp. The evaluated release artifact is
`model-cal-1024-block128.q35h`, SHA256
`e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69`.
It uses H128/Q4-G32-DOT4 and the independently calibrated 1,024-request recipe
described in the [quality study](qev-calibration-2026-10-21.md).

## What a decision computes

A request is rendered as a plain state-first prompt ending in `Answer: (`.
The model processes its input tokens, applies its final vocabulary projection,
and the server reads the logits corresponding to the option letters. It computes
an option-only softmax, applies the publisher's per-type temperature with a
1e-9 floor, renormalizes, and formats the System One answer. Reading raw recurrent
states as probabilities would be incorrect: the vocabulary projection and
probability transformation remain necessary.

There is no autoregressive continuation. The implementation uses one internal
forced placeholder output to obtain its synchronous raw-logits callback, but
performs **zero decode forwards** and reports `usage.output_tokens: 0`.
It currently computes the full vocabulary head rather than a specialized
option-row projection. Decode tokens/s therefore does not describe decision
performance. Measure decision latency, completed decisions/s, input tokens/s,
and queueing under a stated load.

Multiple questions use one independent request state per evaluated row. Joint
mode evaluates the causal prefix through each answer slot; per-question mode
repeats the state with each isolated question. Both are compatible with the
publisher's token layout. The default is joint; release instructions select
per-question mode to match the independent quality evaluation. Neither mode
uses decision prefix caching in this initial implementation.

## Parallel HTTP requests and prefill

HTTP workers can accept multiple requests while one owner drives the existing
bounded CpuEngine. Each prefill chunk uses the engine's configured thread pool
and batched operations over that prompt's tokens. The scheduler time-slices
prefill chunks between resident requests. It does **not** combine tokens from
different requests into one cross-request prefill batch.

The engine's decode batching only applies when requests need subsequent token
forwards. Decision requests finish at the initial readout, so that optimization
does not apply. More outstanding requests may keep the server busy but can also
increase queueing without increasing compute throughput. Parallel HTTP arrival,
parallel execution within one prefill and cross-request prefill batching are
different mechanisms. The measurements below separate throughput from latency.

## Verification

- All **17 CTest cases** pass, including protocol formulas, option ordering,
  score/noul readout, tie handling, JSON escaping and array annotation.
- **40 prompt/answer fixtures** agree with decider-ai 1.9.0, including both
  encodings, 255-option questions, multiple question types, nested/Unicode states,
  combining marks, indentation and punctuation.
- All **15,154 frozen evaluation question prompts** produce exactly the same
  native token IDs as the HF-prepared quality inputs. This validates prompt
  preparation, without fitting or reselecting any final candidate.
- Real HTTP probabilities on the original 19 smoke questions match the offline
  calibrated checkpoint's probabilities after four-place rounding (maximum delta
  zero). A joint multi-question HTTP result also agrees with independent native
  evaluation of its exact prefix rows.
- Concurrent repeat requests preserve answers. Existing greedy/sampled text
  completions, mixed settings, shared prefixes and CLI/API parity still pass.
  Invalid requests, authentication and mixed completion/decision traffic are
  exercised by the HTTP checks.

The full prompt audit exposed pre-existing native-tokenizer differences that
were hidden by the study's explicit HF token IDs: whitespace-regex backtracking,
incorrectly treating `#` merge lines as comments, approximate Unicode categories
and absent NFC normalization. The tokenizer now implements the two supported
Qwen regex variants and statically links pinned utf8proc 2.10.0 for Unicode/NFC.
There is no additional shared-library runtime dependency. Inference weights and
quantization math are unchanged.

## Performance protocol

Ryzen 9 9955HX3D, eight physical V-cache cores (affinity mask 255), GCC 16.2.1
Release, native AVX-512/VNNI dispatch. llama.cpp uses CPU only at revision
`9b56ccf00d755b0bf780140f6352a67db1ac0378`. Every comparison runs sequentially
through `scripts/benchmark-inference-seq.ps1`, with one warmup and three measured
runs. No quality evaluator or build runs concurrently.

Each sample starts a fresh server, waits for readiness, and times its first
decision request or concurrent group. Model loading/readiness are excluded;
local HTTP, tokenization, scheduling, inference and JSON output are included.
The same JSON fixtures are sent to native and llama.cpp. Request state is a
pre-rendered string and option keys are sorted, removing object-order ambiguity.
The one-request fixtures have 74 and 579 input tokens on both APIs. These are
B1 first-request observations, not a warmed steady-state throughput claim.

Concurrency tests use only the native Q4 server, the same eight inference
threads, and 1/4/8/16 residents. Each request receives a distinct single-letter
state prefix; the B1 control uses the same variant. This yields 76 or 581 input
tokens per request at every concurrency, without identical-state cache credit.
Client thread-pool dispatch and queueing are included in group wall time.
Aggregate decisions/s is completed requests divided by group wall time;
request latency is measured separately for each HTTP call. No concurrent native
throughput is compared with the B1 baselines of another engine.

## Single-request HTTP latency

| Server / checkpoint | 74 tokens, median ms [min, max] | 579 tokens, median ms [min, max] |
| --- | ---: | ---: |
| Native H128/Q4-G32-DOT4, calibrated | 60.99 [50.66, 62.25] | 249.23 [221.57, 250.46] |
| llama.cpp CPU BF16 | 88.21 [79.48, 88.84] | 549.55 [483.58, 578.41] |
| llama.cpp CPU Q8_0 | 100.24 [93.74, 107.36] | 756.60 [693.91, 771.09] |

## Native Q4 concurrency

Separate workload: distinct state tags add two input tokens, so the control
and all concurrent requests contain 76 or 581 input tokens. Throughput is the
aggregate across a group; latency is per request. Values are medians across
three measured groups (latency uses each group's mean).

| Concurrent requests | 76-token decisions/s | Mean request latency ms | 581-token decisions/s | Mean request latency ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 19.11 | 52.33 | 4.41 | 226.95 |
| 4 | 20.61 | 169.31 | 4.63 | 809.98 |
| 8 | 21.95 | 315.40 | 4.60 | 1622.96 |
| 16 | 24.19 | 559.36 | 4.59 | 3238.25 |

For the short fixture, concurrency 16 increases observed throughput by 26.6%,
while mean request latency rises from 52 ms to 559 ms. For the longer fixture,
throughput changes by only 4.2% (4.41 to 4.59 decisions/s), while mean latency
rises from 227 ms to 3,238 ms. With only three groups, the small long-prompt
throughput difference should not be treated as a robust improvement. These
measurements support bounded queueing and latency-aware admission; the current
server does not get a decode-batching speedup on decision requests. The cause
of the modest throughput changes is not isolated by this experiment.


## Reproduction and evidence

```sh
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j 8
ctest --test-dir build-linux --output-on-failure
.cache/qev-venv/bin/python tests/systemone_tokenizer_parity.py \
  --prepared benchmarks/qev-calibration-2026-10-21/eval \
  --protocol build-linux/systemone_protocol_test --model-dir models/qev-0.8b \
  --output benchmarks/qev-api-2026-10-21/full-tokenizer-parity-final.json
.cache/qev-venv/bin/python tests/systemone_server_smoke.py \
  --server build-linux/qwen35_cpu_server --protocol build-linux/systemone_protocol_test \
  --native-study build-linux/qev-native-study --model-dir models/qev-0.8b \
  --weights models/qev-0.8b/model-cal-1024-block128.q35h \
  --study-logits benchmarks/qev-calibration-2026-10-21/smoke-1024-block128.tsv
.cache/powershell/pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix benchmarks/qev-api-2026-10-21/http-final-matrix.json \
  -OutputDir benchmarks/qev-api-2026-10-21/http-final -Runs 3 -WarmupRuns 1 -Affinity 255
```

Use fresh output paths when reproducing. The optional native study target is
excluded from the default build; build it explicitly for the joint CPU oracle.
Python validation requires the study's pinned tokenizers/decider dependencies;
the HTTP server itself requires neither. The llama-server comparison is built
from the pinned upstream source with CUDA disabled and server enabled.

Compact CSV/JSON summaries, fixtures, commands, binary/checkpoint hashes,
source revisions and test logs are in `docs/results/qev-api-2026-10-21`.
Raw profiles, server logs, original and repeated measurements remain locally
under the ignored `benchmarks/qev-api-2026-10-21` directory.
