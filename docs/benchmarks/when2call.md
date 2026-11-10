# qev on NVIDIA When2Call

This study compares the same qev-0.8b fine-tune in H128/Q4-G32-DOT4, Q8_0 and
BF16 through the Decision API. The dataset is NVIDIA's complete **3,652-item
When2Call MCQ test split**, without exclusions. Model artifacts, prompts,
temperatures, answer ordering and scoring are fixed; no fitting occurs.

Results were collected locally on November 10, 2026:

| Checkpoint | Complete-test backend | Correct / 3,652 | Accuracy | Complete pass wall time |
| --- | --- | ---: | ---: | ---: |
| H128/Q4-G32-DOT4 | Native CPU, Ryzen 5 7600 | 1,651 | 45.21% | 38m 41s |
| Q8_0 | llama.cpp Vulkan, Intel Arc Pro B70 | 1,651 | 45.21% | 7m 49s |
| BF16 | llama.cpp Vulkan, Intel Arc Pro B70 | 1,647 | 45.10% | 7m 34s |

The CPU Q8 run was stopped after 1,317 tasks when GPU acceleration was ready.
At the time of the user's duration question, the projected complete CPU Q8
pass was about 90 minutes. The 32-item CPU pilot projected about 57 minutes for
BF16: BF16 was faster than Q8 in that pilot, despite its larger checkpoint.
Those are historical estimates, not completed CPU benchmark durations.
The complete-pass times above include startup and evidence writes and are not
controlled, repeated speed comparisons between formats or hardware.

Q4 and Q8 have equal total accuracy but different predictions on 199 items:
Q4 alone is correct on 74, Q8 alone on 74. Against BF16, Q4 alone is correct
on 77 items and BF16 alone on 73, a net difference of four tasks (0.11 percentage
points). This does not establish a quantization quality improvement. Native
explicit API choices give Q4 1,650 correct (45.18%); its prespecified returned-
probability argmax score is 1,651 because of one rounded tie.

The much lower score than the earlier JevBench run exposes a task limitation:
qev calls tools too readily when information or a suitable tool is missing.
This matches the [publisher's declared abstention weakness](https://huggingface.co/labqoat/qev-0.8b/blob/9a095f35fe0e27a888cfed9eb1517718dc9dc59e/README.md). Higher weight
precision does not resolve it in this protocol.

| Checkpoint | Tool-call recall | Request-information recall | Cannot-answer recall | Three-class macro-F1 | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| H128/Q4-G32-DOT4 | 94.75% | 26.74% | 10.81% | 0.3885 | 0.7507 |
| Q8_0 | 94.29% | 30.04% | 8.57% | 0.3875 | 0.7697 |
| BF16 | 94.29% | 29.94% | 8.34% | 0.3858 | 0.7709 |

The 45% accuracy exceeds the 35.46% majority baseline, but none of these Brier
scores improves on uniform four-candidate probabilities (0.75). Accuracy alone
does not establish a useful confidence calibration or reliable abstention.

## CPU and GPU agreement

The same 128-item panel gives **zero changed decisions** between CPU and Vulkan
for both Q8 and BF16. Each backend is correct on 56/128 panel items. The maximum
single-option probability difference is 3.94 percentage points for Q8 and 1.27
points for BF16. This is an arithmetic agreement check, not a second full-test
accuracy estimate.

The retained 1,317-item Q8 CPU prefix gives a larger check: **14 changed
decisions (1.06%)**, with four CPU-only correct answers and five GPU-only correct
answers. Five other changes switch between wrong candidates. CPU is correct on
115 of those items and GPU on 116; maximum option-probability difference is 4.24
percentage points. This source-ordered prefix is not representative full-test
accuracy and is excluded from the complete-test denominator.

GPU acceleration therefore preserves the broad result while occasionally
changing close decisions. It is not bit-identical CPU arithmetic. Q4 native CPU
versus Q8/BF16 Vulkan comparisons describe checkpoint/runtime pairs; they cannot
isolate bit width from every backend arithmetic difference. A complete CPU/GPU
agreement claim is not made from a subset check.

Compact evidence contains the [comparison](../results/when2call-2026-11-10/comparison.csv),
[all 3,652 per-item predictions and probabilities](../results/when2call-2026-11-10/predictions.csv),
[per-category metrics](../results/when2call-2026-11-10/per-category.csv),
[CPU validation](../results/when2call-2026-11-10/validation.json) and
[commands, artifact hashes and source revisions](../results/when2call-2026-11-10/manifest.json).
All 10,956 complete-test responses, 256 CPU-panel responses and 1,317 interrupted
CPU responses were independently checked against their raw response hashes and
rescored. No complete-test response was invalid, truncated or retried.

## Task and scoring

[When2Call](https://github.com/NVIDIA/When2Call) presents a user question,
available tool definitions and four candidate assistant responses. The decision
is whether to call a tool, request missing information, decline because the
available tools cannot answer, or answer directly.

The actual MCQ test has the following gold counts:

| Response category | Gold items |
| --- | ---: |
| Tool call | 1,295 |
| Request information | 1,062 |
| Cannot answer | 1,295 |
| Direct answer | 0 |

Every item still has four candidates, including the direct-answer distractor.
Consequently the test cannot measure direct-answer recall. Direct predictions
are retained as errors, rather than removed. Uniform guessing has 25% expected
accuracy; the majority-category baseline is 1,295/3,652, about 35.46%.

Each request contains only the supplied tools, user question and exact candidate
texts. It does **not** contain `correct_answer`, `target_tool`, `orig_tools`,
`orig_question`, `held_out_param`, source identifiers or any other generation
metadata. NVIDIA's generic tool instructions and tool JSON strings are retained
in the state, with a fixed System One choice question. Candidates have opaque
`option_1` to `option_4` names. Their order is the stable SHA-256 ordering of
`qev-when2call-2026-11-10-v1:uuid:category`, chosen before model outputs. Correct
positions are 901, 941, 904 and 906, respectively.

Accuracy is the share of all 3,652 items whose highest returned option
probability selects the gold category. Exact probability ties choose the
lexicographically first option key. The engine's own `choice` is also retained
to expose any differences caused by output rounding. Failed/invalid responses
count as incorrect. Brier is the sum of squared errors over all four candidate
probabilities, averaged over valid responses; lower is better. ECE uses ten
equal-width confidence bins.
Uniform probabilities over the four candidates give a Brier baseline of 0.75.

The native API rounds probabilities to four decimal places, whereas llama.cpp
returns more digits. An unrounded native winner can therefore become tied in
the returned distribution. The secondary `api_choice_correct` count retains
the explicit API decisions; it does not replace the prespecified probability
argmax score. The comparison remains one of these checkpoint/runtime pairs,
rather than bit width alone.

Category precision, recall and confusion counts are reported separately.
Macro-F1 over the three categories with gold support is distinguished from
fixed-four-category macro-F1, which includes the unsupported direct category.
The no-tools hallucination rate follows NVIDIA's definition: among gold
`cannot_answer` items with no available tools, the fraction choosing `tool_call`.

This is a **candidate-choice adaptation for a decision model**. NVIDIA's
original lm-evaluation-harness setup instead scores the likelihood of each
complete response, with both raw and length-normalized variants. Our score is
not claimed to reproduce those likelihood metrics or the Artificial Analysis
filtered When2Call subset. The full original NVIDIA MCQ test remains the
denominator.

## Frozen inputs and execution protocol

- Dataset: [`nvidia/When2Call`](https://huggingface.co/datasets/nvidia/When2Call/tree/0582f7749df63a96fdc3070932e83e72396ace53),
  revision `0582f7749df63a96fdc3070932e83e72396ace53`,
  `test/when2call_test_mcq.jsonl`, 20,424,385 bytes, SHA-256
  `8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c`.
  Data license: CC-BY-4.0. The separate 300-item LLM-judge split is not included.
- NVIDIA code: `ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc`.
  The generic tool format and source scoring definitions are archived unchanged.
- Model artifacts and engines are the same pinned files as the
  [JevBench precision comparison](jevbench-v1.md#repeated-cpu-latency-and-q8bf16-references):
  Q4 uses native CPU inference. Q8_0 and BF16 use external llama.cpp at
  `7d7f0287530bc8981690f361d8c1627dbae1c99a`, built with Vulkan for complete
  quality runs and CPU-only for a separate agreement check. Both paths load the
  exact same publisher GGUFs; the native product remains CPU-only.
- AMD Ryzen 5 7600, six inference threads pinned to logical CPUs 0–5 (six
  physical cores), `amd-pstate-epp`, `powersave` governor. Raw identities retain
  executable/model/library hashes and commands. GPU quality runs use the Intel
  Arc Pro B70 (Vulkan0), with all model layers offloaded. The existing GPU service
  stays running. These runs measure quality, not isolated GPU performance.
- One resident HTTP request, prefix caching disabled, `per_question`, fixed
  choice temperature 1.7, FP16 KV, 8,192-token context. No generated output
  tokens, retries, prompt optimization or truncation. llama.cpp uses batch
  2,048, microbatch 256 and Flash Attention.
- Native input lengths: 246–5,526 tokens, median 897.5; **3,658,064 tokens**
  for the complete test. The entire test fits without shortening any tool
  definition, user question or response candidate.
- Exact input-token equality is independently checked on all 3,652 items:
  the native protocol encoder and both actual GGUF vocabularies through
  `llama_tokenize` (the server's `add_special=false`, `parse_special=true`
  settings) produce identical sequences. The offline vocabulary audit performs
  no model inference. Actual HTTP input-token counts must also match each
  prepared row throughout every run.

The complete quality pass runs once per checkpoint/runtime pair. After the Q4
CPU pass completed, the user requested GPU acceleration for Q8 and BF16. The
already running Q8 CPU pass is retained as partial validation evidence; its
results do not contribute to the complete-test accuracy denominator. Neither
prompts nor weights changed. The original plan for repeated CPU latency runs is
retained in raw evidence but superseded by this explicit change of scope.

A separate CPU check uses the same **128 input-only selected items** for Q8 and
BF16, ordered by SHA-256 of `timing:uuid`, with one pass per model. Those CPU
probabilities and predictions are compared with the matching full GPU results.
The panel has 126,062 input tokens, median 828 and maximum 5,455.
The interrupted Q8 CPU prefix provides an additional comparison. These are
backend agreement checks, not new unique test cases or performance measurements.

All scored series use the unmodified `scripts/benchmark-inference-seq.ps1`,
sequentially with affinity 63 and `-Runs 1 -WarmupRuns 0`. GPU startup smoke tests
are excluded from published accuracy. The preserved original CPU driver and the
amended GPU-capable driver are separately hashed. The amendment changes only
backend selection and identity recording; input construction and scoring remain
frozen. Native/GGUF input token sequences still match on all 3,652 tasks.

Reported elapsed time is the wall time for one complete quality pass, including
startup, HTTP requests and evidence writes. It is **not** a repeated median
latency or a controlled CPU/GPU speed comparison. When2Call's median complete-test
input is about 898 tokens, versus 115 for the earlier JevBench run; latency across
those workloads is not an engine regression comparison. No new comparative CPU
latency table is claimed by this study.

## Reproduction and validation

The stdlib-only Python 3.11+ [driver](../../scripts/when2call.py) requires the pinned native
qev package, publisher Q8/BF16 GGUFs, `build-jevbench/qwen35_cpu_server`,
`build-jevbench/systemone_protocol_test`, and the pinned CPU `llama-server`. Full GPU reference runs additionally need
a separate stock Vulkan build at `.cache/llama-qev-vulkan/bin/llama-server`,
with the same llama.cpp revision and Intel B70 visible as `Vulkan0`.
The preparation command validates the complete dataset SHA-256 and exports
requests with exact native token counts:

```sh
python scripts/when2call.py prepare \
  --dataset benchmarks/when2call-2026-11-10/upstream/test-when2call_test_mcq.jsonl \
  --prepared benchmarks/when2call-new/prepared
```

Use the recorded [Q4 CPU](../results/when2call-2026-11-10/quality-cpu-matrix.json),
[Q8/BF16 GPU](../results/when2call-2026-11-10/quality-gpu-matrix.json) and
[CPU-validation](../results/when2call-2026-11-10/cpu-validation-matrix.json) matrices, adjusting
the prepared-input path if needed. Use fresh output directories. Each uses
`-Runs 1 -WarmupRuns 0 -Affinity 63`. `--backend vulkan` selects the external
GPU binary for Q8/BF16; the default remains CPU. `--limit 128` selects the frozen
CPU agreement panel. `quality-sources.json` identifies the three complete passes.
The interrupted CPU run is never treated as a complete quality pass.
Then independently recompute results from raw HTTP responses:

```sh
python scripts/summarize-when2call.py \
  --root benchmarks/when2call-2026-11-10 \
  --output docs/results/when2call-2026-11-10
python tests/when2call_test.py
```

Raw tasks, HTTP responses, logs, profiles, source snapshots and identities stay
under ignored `benchmarks/when2call-2026-11-10/`. Compact evidence belongs under
`docs/results/when2call-2026-11-10/`. The driver bounds each run and stops before
22:40 Europe/Berlin, preserving partial evidence on failure and terminating its
server. The dataset is public: the original fine-tune's unpublished training
corpus still prevents an independent exclusion of training overlap. This run
neither uses test labels for model fitting nor establishes contamination-free
generalization.
