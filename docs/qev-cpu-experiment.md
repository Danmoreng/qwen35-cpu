# qev-0.8b CPU feasibility and calibration plan

The subsequent [calibration study](qev-calibration-2026-10-21.md) updates
llama.cpp, evaluates the author's Jev Skills suite and compares calibrated Q4
and selective Q8. The initial measurements below remain historical feasibility
results; they are not measurements of the newer calibrated artifacts.

## Result

The text-only [labqoat/qev-0.8b](https://huggingface.co/labqoat/qev-0.8b)
fine-tune runs locally in BF16 and Q8_0 through the CPU-only llama.cpp comparison
harness, and in **H128/Q4-G32-DOT4** through the existing native engine. The native
packer converted all 230 required tensors without an architecture or loader change.
The custom checkpoint uses **unweighted MSE16, without calibration**. It does not
inherit the original chat checkpoint's calibration or its quality scores.

qev uses Qwen3.5-0.8B-Base's text architecture, with decision-specific fine-tuned
weights. Its output is a distribution over option-letter tokens at an answer slot.
No generated answer text is needed. Images were not tested.

## First local latency measurements

AMD Ryzen 9 9955HX3D, eight physical cores (CPU affinity 0-7, mask 255) sharing
96 MiB L3, eight threads, CPU-only execution, FP16 KV, B=1. Measurements use
`scripts/benchmark-inference-seq.ps1`, one warmup process run and three measured
process runs, alternating case order. Values below are median milliseconds.

| Engine / checkpoint | 74 input tokens | 579 input tokens | Checkpoint file MB |
| --- | ---: | ---: | ---: |
| Native H128/Q4-G32-DOT4, uncalibrated | 49.56 | 218.56 | 425.0 |
| llama.cpp BF16 | 79.06 | 585.38 | 1,516.7 |
| llama.cpp Q8_0 | 95.26 | 786.24 | 811.8 |

These measure the first prompt forward after each fresh model/context load,
including full-vocabulary output projection. Loading, tokenization, probability
postprocessing and logit-dump I/O are excluded. There are zero decode forwards and
no prefix-cache hits. A warmup process warms the OS cache; this is not a benchmark
of repeated requests in a resident server. The Python adapter starts a process
for every question, so its end-to-end latency also includes loading.

The long BF16 samples range from 503.36 to 594.08 ms; long Q8_0 ranges from
707.28 to 793.75 ms. These are initial measurements on this machine, not a claim
about all workloads or optimized qev runtimes. Q8_0 does not automatically make
prefill faster than this CPU's BF16 path. File MB is decimal disk size, not RAM
usage. The BF16 and Q8 GGUFs contain the text model; the downloaded safetensors
also contain the unused vision tower.

## Initial quality checks

The [smoke requests](../configs/qev-smoke.jsonl) contain 15 requests / 19 questions
covering routing, support, injection, sentiment, scores, rules, a German request
and one game-move decision. Every question uses its own prompt (`per_question`).
The label tokens and type temperatures match the pinned upstream configuration.

| Checkpoint / scoring backend | Argmax matches hand expectations | Mean option KL to BF16, nats | Largest option probability change |
| --- | ---: | ---: | ---: |
| BF16 / llama.cpp | 19/19 | 0 | 0 |
| Q8_0 / llama.cpp | 19/19 | 0.00003663 | 1.00 percentage points |
| H128/Q4-G32-DOT4 / native | 19/19 | 0.01004140 | 26.68 percentage points |

For scores, the check uses the most probable level, not the rounded expected
score. KL is computed over each question's tempered option distribution and
averaged across questions. These are hand-authored sanity checks, not a held-out
benchmark or an estimate of real-world accuracy/calibration. BF16 here means
BF16 GGUF on llama.cpp, not the publisher's CUDA reference.

The game example exposes a large probability change despite unchanged argmax:
BF16 assigns the safe left move 70.52% and the dead-end right move 20.72%; the
uncalibrated custom checkpoint assigns 43.84% and 42.79%. Preserving only top-1
answers would miss this regression. The 19 smoke questions must remain development
examples after inspecting these results.

## Reproduction

Install the experiment's Python-only prompt dependencies in an isolated environment:

```sh
uv venv --python 3.12 .cache/qev-venv
uv pip install --python .cache/qev-venv/bin/python tokenizers==0.23.3 decider-ai==1.9.0
```

Download `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
`merges.txt`, `qev_config.json` and `model.safetensors` from model revision
`6334efd2d556cad22a3040049efb2887a60e7fb2` into `models/qev-0.8b`.
Download `qev-0.8b-bf16.gguf` and `qev-0.8b-Q8_0.gguf` from the
[GGUF repository](https://huggingface.co/labqoat/qev-0.8b-GGUF) at revision
`79caef6979f61c4ee4fa6e894424789ce66042db` into that directory.
All downloaded assets in this experiment match the respective upstream SHA256SUMS.

```sh
build-linux/qwen35_cpu_pack --hf-model-dir models/qev-0.8b \
  --output models/qev-0.8b/model-mse16.q35h

taskset -c 0-7 .cache/qev-venv/bin/python scripts/qev-cpu.py \
  --model-dir models/qev-0.8b --requests configs/qev-smoke.jsonl \
  --output benchmarks/qev-native-new --backend native \
  --executable build-linux/qwen35_cpu --weights models/qev-0.8b/model-mse16.q35h
```

For BF16 or Q8_0, select `--backend llama`,
`--executable build-llama-linux/bin/llama-fixed-cpu-bench` and the matching GGUF.
Use a fresh output directory for each invocation. `--prepare-only` writes prompt
tokens without loading weights. Supply existing upstream config/tokenizer files:
the adapter verifies all 255 label IDs and passes identical explicit input tokens
to both engines, avoiding differences in native versus HF pre-tokenization.

The llama comparison harness now accepts one output at B=1, so prefill computes
the decision logits without a subsequent decode step. Its first logits were
verified bit-identical to the existing two-output path. All 19 adapter prompt
encodings and probability readouts match the pinned upstream `qev.py`; all 16
existing CTest checks pass. No System One HTTP endpoint was added.

For timing, prepare token files under the paths in the
[saved matrix](results/qev-cpu-2026-10-21/speed-matrix.json), including the
[long request](results/qev-cpu-2026-10-21/long.jsonl), then run that matrix through
`scripts/benchmark-inference-seq.ps1` with `-Runs 3 -WarmupRuns 1 -Affinity 255`
and a fresh output directory. Run performance measurements sequentially.

## Scaling the calibration study

Quantization calibration fits weight scales/codes using BF16 projection inputs.
It needs representative input requests, without gold labels. Probability
calibration fits the three output temperatures against labelled decisions. The
model card's held-out calibration split refers to the latter. Keep these separate.

Use a sample of the original training inputs if available, or independently
generated/domain inputs in the same `state` / `questions` layout. Preserve the
actual runtime encoding, prompt-length distribution, option counts and languages.
No answer letter is inserted into an answer slot. Include easy and ambiguous
decisions, negation, missing information and plausible distractors. Split related
records, paraphrases, reordered options, coherence twins and common source
documents as groups before sampling.

The publicly listed Labqoat dataset found on 2026-10-21 is
[Jev Skills](https://huggingface.co/datasets/labqoat/jev-skills): 10,168 evaluation
items. The complete 2.2M-request training mix was not located in the model card
or the author's public dataset listing. Reserve Jev Skills, Benchmark Heaven and
JevBench for evaluation; similarly named datasets by other authors are not the
original qev-0.8b training release. This is a discovery result, not proof that no
training-data download exists elsewhere.

Recommended experimental budget (a proposal, not an established optimum):

| Purpose | Starting size | Use |
| --- | --- | --- |
| Quantization calibration | 256, then 1,024 independent requests | Capture BF16 moments; compare nested subsets; aim initially for roughly 0.25-1M actual prompt tokens at the larger size |
| Development evaluation | 1,000 labelled decisions | Select calibration size, sampling mix and fitting recipe |
| Optional temperature fitting | 3,000 labelled decisions, about 1,000 per output type | Refit temperatures only after choosing the quantized checkpoint |
| Final evaluation | 5,000-10,000 labelled decisions or the full Jev Skills suite | Freeze until recipe selection is complete; report per-skill results |

If 1,024 calibration requests still improve development results, test 4,096 and
at least two sampling seeds. Requests can contain several correlated questions:
record independent state/group counts, question counts and tokens separately.
Do not pad short requests or concatenate unrelated recurrent states merely to
hit a token quota. More calibration data is not guaranteed to improve the result.
[Calibration-data research](https://arxiv.org/abs/2311.09755) supports checking
the sampling distribution rather than assuming a generic corpus is sufficient.

Keep upstream temperatures fixed during the initial quantization comparison.
Report task accuracy, option KL/total variation to BF16, Brier/NLL against gold,
top-1 flips by BF16 margin, score error, and coherence on grouped transforms.
Report ECE with bin definitions and uncertainty; use paired/group bootstrap
intervals for differences. Matching BF16 probabilities does not establish that
BF16 itself is well calibrated. A temperature adjustment cannot restore changed
option rankings.

For CPU throughput, keep the BF16 teacher resident and accumulate diagonal and
block-128 second moments in memory, writing compact statistics. Reuse those
statistics for fitting variants. Preserve H128 basis handling (identity for the
tied embedding/head), weighted MSE16, damping and block error compensation from
the existing recipe. Do not reuse activation statistics from the chat model.
The existing capture/fitting tools provide the primitives; a resident CPU study
driver and grouped data preparation remain to be implemented.

Cache option logits instead of full-vocabulary dumps for the large study:
10,000 slots × 255 × FP32 is about 10.2 MB per checkpoint, compared with about
9.93 GB for 248,320 logits at every slot. Stream/compact projection statistics
instead of retaining raw activation matrices for every request. The initial
subprocess adapter and full-vocabulary dump path are suitable for feasibility
checks but should be replaced before a large sweep.

## Storage budget for the calibration study

On 2026-10-21 the local filesystem has about 22.2 GiB free. The existing qev
weights already occupy about 4.2 GiB and are included in used space. The study
can run within the remaining capacity if captures and evaluation output are
bounded. A planning allowance of 5-8 GiB additional disk space covers a modest
sweep; it is an estimate, not a measured peak. Keep at least 10-15 GiB free and
check capacity before each stage. Larger sweeps should add checkpoints and
statistics incrementally rather than multiplying every variant at once.

The safetensors header contains 187 relevant projection matrices with 277,504
input columns in total. The resulting sizes are:

- FP64 block-128 second-moment sums: 271 MiB per aggregate calibration set.
- Fitted FP32 moments plus inverse-Cholesky factors: another 271 MiB per set.
- FP64 diagonal sums: 2.12 MiB per set.
- One existing-style raw capture: up to about 270 MiB for 256 trunk samples
  per projection and one output-head sample. Retaining 1,024 such captures
  would require about 270 GiB.
- One additional standard custom checkpoint: about 405.3 MiB.
- 10,000 answer slots of 255 FP32 option logits: about 9.73 MiB.

Accumulate block and diagonal moments before reclaiming generated raw captures;
retain the input corpus, selection, counts, hashes and checked aggregate files.
Diagonal-only compaction cannot reconstruct covariance later. Bound raw captures
to one or two requests at a time, account for the temporary aggregate file used
for atomic replacement, and preserve the covariance information needed by the
error-compensated fit. This retains the statistics used by the current fitting
objective without keeping all activation matrices. The existing resident driver
is CUDA-specific and needs a CPU adaptation before running this qev study.

## Evidence

[Performance CSV](results/qev-cpu-2026-10-21/performance.csv) ·
[Quality summary](results/qev-cpu-2026-10-21/quality.json) ·
[Manifest](results/qev-cpu-2026-10-21/manifest.json) ·
[Upstream source verification](results/qev-cpu-2026-10-21/llama-source-verification.json).

Raw profiles, commands, logit dumps, source snapshots and conversion logs remain
in ignored `benchmarks/qev-2026-10-21/`; weights remain in ignored
`models/qev-0.8b/`. The manifest records hashes of binaries, shared libraries,
weights, tokenizer and evidence. All 3,540 files in the upstream llama.cpp source
archive were verified against pinned revision
`73a43d1f69345aee8bb186ef4b3172cef892f2e5`. The CMake-generated build version can
show the parent repository's revision because this source is an archive without
`.git`; use the verified upstream revision above for provenance.
