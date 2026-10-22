# qev-0.8b decision calibration study

This study evaluates the decision fine-tune with the native CPU engine. It keeps
**H128/Q4-G32-DOT4** as the standard representation and tests selective Q8 as an
explicit size/quality tradeoff. This is a quality study; resident evaluation
throughput is not a performance benchmark. The earlier feasibility measurements
are documented in [the CPU experiment](https://github.com/Danmoreng/qwen35-cpu/blob/3680efb33bf6975ce6722d64d20c641cb7490b06/docs/qev-cpu-experiment.md).

## References and data separation

The model is `labqoat/qev-0.8b` at revision
`6334efd2d556cad22a3040049efb2887a60e7fb2`. BF16 and Q8_0 GGUFs come from
`labqoat/qev-0.8b-GGUF` at `79caef6979f61c4ee4fa6e894424789ce66042db`.
All assets match their publisher's SHA256SUMS. The downloaded safetensors also
contain vision weights; this CPU experiment converts only the supported text
model.
An additional direct audit verifies all 187 projection matrices byte for byte
between HF and BF16 GGUF. All 133 auxiliary text tensors agree under the expected
Qwen norm/SSM conversions: 130 exactly, with only three negative-exponential SSM
parameters differing by at most one FP32 ULP (maximum absolute delta 1.19e-7).

llama.cpp was updated to master revision
`9b56ccf00d755b0bf780140f6352a67db1ac0378` (2026-10-21) before these results.
The earlier interrupted run and the old CPU speed measurements use an older
revision and do not contribute logits to this study.

The author-published **Jev Skills** suite is pinned at
`fda80b2a75fca3fa3e5d3f6378acc2a6c04d4032`: 10,168 requests, 15,154 questions
and 5,219,303 input tokens, with a longest question prompt of 8,603 tokens.
These are Jev Skills results, not Benchmark Heaven scores from the model card.
No prompt truncation or exclusions were needed at the 32,768-token context cap.

A stable SHA256 split was frozen before candidate fitting/results. It keeps
questions belonging to a request, game question/state variants, and paired
complete/missing-fact examples together:

| Panel | Requests | Questions | Input tokens | Purpose |
| --- | ---: | ---: | ---: | --- |
| Development | 1,035 | 1,553 | 464,693 | Choose fitting settings and precision |
| Final | 9,133 | 13,601 | 4,754,610 | Evaluate frozen selected candidates |

The original 19-question smoke set is development material. Its Snake example
can illustrate probability drift but cannot establish general quality.

## Independent calibration inputs

The full original 2.2M-request training release was not located publicly. We
therefore prepare inputs from independently pinned public **training** splits,
plus local program-generated tasks. We never fit weights on Jev Skills prompts
or labels. An exact-state and shared 16-word-shingle screen rejects overlaps
with evaluation states; it does not prove semantic independence or remove any
possible overlap already present during the author's model training.

| Input family | Requests in 1,024 mix | Source / license recorded in manifest |
| --- | ---: | --- |
| Customer support, 4–77 options | 128 | Banking77 train / CC-BY-4.0 |
| Passage-grounded yes/no | 256 | BoolQ train / CC-BY-SA-3.0 |
| Paraphrase decisions | 128 | PAWS labeled-final train / Apache-2.0 |
| Prompt-injection detection | 128 | deepset prompt-injections train / Apache-2.0 |
| Response preferences | 128 | UltraFeedback train_prefs / MIT |
| Policy threshold counts | 128 | Local program / CC0 |
| Coordinate choices | 128 | Local program / CC0 |

The 1,024-request mix contains **252,138 tokens**. Its nested 256-request prefix
contains **63,800 tokens** with the same family proportions. Inputs are capped
at 2,048 tokens. Some local policy questions use German. Actual calibration uses
only input activations; source labels are provenance, not fitting targets.
This is quantization calibration, separate from output-temperature fitting.
The publisher's fixed temperatures remain choice 1.7, noul 0.9 and score 1.25.

## Execution and fitting

All engines receive identical explicit HF token IDs and the publisher's 255
option-letter IDs. Each question is encoded independently (`per_question`),
without a generated answer. Each evaluator resets KV and recurrent state between
questions. Reported KL is BF16-to-candidate KL over the actual tempered option
distribution, averaged across questions, in nats. Accuracy is the upstream
scorer's macro mean across scored skills; `rules/missing_facts` uses its separate
AUROC metric. Argmax disagreement with BF16 is not an additional-error count.

BF16 and Q8 references use the external llama.cpp CUDA teacher. Native candidate
execution uses CPU only. On 225 stratified questions, including the five longest,
CPU versus CUDA BF16 mean option KL is 0.00002347, maximum option probability
difference 1.73 percentage points, with two near-tie argmax changes. This is a
validated practical teacher, not bit-identical CPU arithmetic. Enabling activation
capture preserves option logits bit for bit on four validation requests.

The vendor Q8 GGUF is not a native-engine baseline: its tokenizer metadata is
`qwen2`, and the native GGUF reader rejects it under its stricter `qwen35`
profile. That guard was retained. Native qev candidates use the HF-to-`.q35h`
conversion and explicit HF input IDs. This does not affect llama.cpp's BF16/Q8
references or the native selective-Q8 `.q35h` path.

One resident teacher captures inputs to all 187 quantized projections. Trunk
activations are transformed into the exact signed H128 basis; the tied head
uses identity. FP64 diagonal and block-128 second moments are accumulated, with
at most 256 sampled input positions per projection per request. The head capture
uses the decision answer slot. A 64-request aggregate checkpoint supports a
verified resume; the collector resumed once from request 192. Duplicate diagnostic
logit rows from the interrupted collection do not enter fitting or evaluation.

Each raw capture is hashed, compacted and removed from a task-owned RAM directory.
Aggregate moments and source/selection hashes remain on disk. Both diagonal
importance and covariance fitting use the existing offline fitter. Covariance
omits cross-block correlations and uses BF16 teacher inputs throughout; it is not
sequential layer recalibration or full-matrix GPTQ.

The search compares the 256-request diagonal and block fits, then 1,024-request
block fits with damping 0.001, 0.01 and 0.1. Development results exposed a tradeoff:
damping 0.1 improves KL but reduces macro accuracy relative to 0.01. The selection
policy was therefore adapted on development data to minimize option KL subject
to macro accuracy no more than 0.5 percentage points below the development BF16
point estimate. This is a practical guard, not a statistical equivalence margin.
Q8 head, gates and combined ablations are tested on both the unconstrained KL
winner and the accuracy-guarded pure-Q4 recipe. Brier and all unsuccessful
candidates remain reported. Both final candidates are frozen before their final
evaluation; final results do not tune any recipe.

## Selective precision

The standard checkpoint quantizes all large projection matrices and the tied
input-embedding/output matrix to Q4. Norms, convolution coefficients and small
recurrent scalar parameters retain FP32. Input and output embeddings share one
248,320 × 1,024 matrix; promoting it changes both lookup and output projection.
The standard artifact has 97 fused Q4 tensors with 751,894,528 parameters and
133 FP32 tensors with 498,496 parameters. Thus approximately 99.93% of its text
parameters are Q4; retaining small FP32 tensors costs 1,993,984 bytes. The tied
matrix accounts for about one third of all text parameters.

`--q8-head` promotes that complete matrix to identity-basis Q8_0. `--q8-gates`
promotes the 36 small recurrent B/A matrices, grouped into 18 runtime tensors.
Unpromoted tensors still use calibrated H128/Q4-G32-DOT4. The converter now allows
these existing selective Q8 modes alongside covariance fitting: Q8 tensors bypass
the Q4 fit, while all remaining Q4 tensors retain it. Artifact audits compare the
unpromoted payload bytes, including split/fused gate groups, against the pure-Q4
parent with the same fitting settings. The selected mixed and pure candidates
can have different damping; they are chosen separately on development data.
There is no native GPU backend or new runtime tensor format.

The Q8 gate experiment changes precision and basis together: the standard gates
use H128, whereas the existing Q8 gate path uses identity. It does not isolate
bit width alone. Native activation quantization also differs from the external
BF16/Q8 teacher's CUDA arithmetic.

## Verification

All 16 existing CTest cases pass. Seven malformed calibration cases are rejected
without publishing partial checkpoints. Repacking the earlier 256-request block
candidate with the extended converter produces a byte-identical checkpoint,
SHA256 `e7eb20e1020059d5f6a4899aa54326c6c96f03bda369c08d23d6e6d14614b6aa`.
Selective precision audits check unpromoted payloads against each candidate's
matching pure-Q4 parent. Raw logits and checkpoint/source/binary hashes remain
available; aggregate fitting statistics are retained after deleting raw captures.

## Development search

These are development results on 1,553 questions; they must not be compared as
if they were measured on the final panel. All native rows use
H128/Q4-G32-DOT4, with selective Q8 additions explicitly identified. File sizes
include metadata and use decimal MB. Rows are ordered by tensor size descending,
then option KL ascending (same ordering as file size for these artifacts).

| Calibration / selective precision | File MB | Macro accuracy | Mean option KL |
| --- | ---: | ---: | ---: |
| BF16, llama.cpp CUDA | 1,516.75 | 69.11% | 0 |
| Q8_0, llama.cpp CUDA | 811.85 | 69.40% | 0.000540 |
| 1024 inputs, block-128 covariance, damping 0.1 + Q8 head and gates | 552.40RTX PRO 5500  | 68.97% | 0.022652 |
| 1024 inputs, block-128 covariance, damping 0.01 + Q8 head and gates | 552.40 | 69.74% | 0.023752 |
| 1024 inputs, block-128 covariance, damping 0.1 + Q8 head | 552.10 | 69.33% | 0.022279 |
| 1024 inputs, block-128 covariance, damping 0.01 + Q8 head | 552.10 | 69.15% | 0.024078 |
| 1024 inputs, block-128 covariance, damping 0.1 + Q8 gates | 425.26 | 68.52% | 0.023610 |
| 1024 inputs, block-128 covariance, damping 0.01 + Q8 gates | 425.26 | 68.90% | 0.025090 |
| 1024 inputs, block-128 covariance, damping 0.1 | 424.96 | 68.35% | 0.023377 |
| 1024 inputs, block-128 covariance, damping 0.01 | 424.96 | 69.30% | 0.025433 |
| 256 inputs, block-128 covariance, damping 0.01 | 424.96 | 69.56% | 0.026113 |
| 1024 inputs, block-128 covariance, damping 0.001 | 424.96 | 68.85% | 0.027782 |
| 256 inputs, diagonal, damping — | 424.96 | 67.97% | 0.038317 |
| Uncalibrated H128/Q4-G32-DOT4 | 424.96 | 67.53% | 0.044945 |

With the same 1,024-input, damping-0.01 parent, promoting the tied head reduces
KL by 5.3%; promoting both head and gates reduces it by 6.6%. At damping 0.1,
the head gives a 4.7% KL reduction. Head promotion adds 127,139,840 file bytes
(29.9%). Gate promotion adds only 297,408 file bytes, but its benefit varies
with fitting settings. The best mixed candidate and best pure candidate have
different damping, so their total difference is not a clean head-only effect.
At 256 inputs, covariance fitting reduces development KL by 31.9% compared
with diagonal importance alone. Increasing the block-fit calibration mix from
256 to 1,024 inputs reduces KL by a further 2.6% at damping 0.01; accuracy does
not increase monotonically. This run supports a diverse 256-input pilot followed
by a 1,024-input confirmation, rather than assuming more calibration tokens
always improve every metric. These observations belong to development data.

## Frozen final results

The independent final panel contains 13,601 questions. BF16 and Q8 use llama.cpp
CUDA for scoring; all H128/Q4-G32-DOT4 candidates use the native CPU engine.
Tensor payload sizes below use decimal MB and exclude metadata.

| Representation / calibration | Tensor MB | Macro accuracy | Macro Brier | Mean option KL to BF16 | Mean TV to BF16 | Argmax disagreements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16, llama.cpp CUDA | 1,505.78 | 68.92% | 0.4097 | 0 | 0 | 0 |
| Q8_0, llama.cpp CUDA | 800.88 | 68.87% | 0.4095 | 0.000496 | 0.79% | 159 |
| H128/Q4-G32-DOT4 + Q8 head, 1,024 inputs, block-128 covariance | 552.07 | 68.87% | 0.4121 | 0.020501 | 5.34% | 1,138 |
| H128/Q4-G32-DOT4, 1,024 inputs, block-128 covariance | 424.93 | 68.74% | 0.4125 | 0.023169 | 5.57% | 1,074 |
| H128/Q4-G32-DOT4, uncalibrated | 424.93 | 67.61% | 0.4195 | 0.040394 | 7.52% | 1,573 |

The selected pure-Q4 artifact gains **1.13 percentage points** of macro accuracy
and reduces mean option KL by **42.6%** relative to the uncalibrated artifact at
the same **424,964,864-byte file size**. Argmax disagreements with BF16 fall by
31.7%. Its accuracy point estimate is 0.18 percentage points below BF16 and
0.13 below Q8; this does not establish equivalent probability predictions.
The paired cluster bootstrap (1,000 resamples, grouping related requests)
places the Q4-minus-Q8 accuracy difference in a 95% interval of
**−0.68 to +0.45 percentage points**. Against BF16 the interval is
−0.73 to +0.38 points. Both include zero and permit a meaningful accuracy loss;
they are not evidence of equivalence.
Against the uncalibrated artifact, the pure-Q4 accuracy-gain interval is
+0.45 to +1.79 points, and its macro Brier improvement interval excludes zero.

The selected mixed recipe reaches the same rounded accuracy as Q8, but its
option KL remains about 41 times Q8's. Compared with the selected pure-Q4
recipe, it reduces KL by 11.5%, reduces mean TV by 4.2%, and has **more** argmax
disagreements with BF16 (1,138 versus 1,074). The file grows to 552,104,704 bytes.
Relative to uncalibrated Q4, it gains 1.26 accuracy points (95% interval +0.60
to +1.84) and reduces KL by 49.2%. Because the final recipes differ in damping,
these final differences cannot be attributed solely to the Q8 head. Its
Q8-relative accuracy interval is −0.58 to +0.57 points.

### Probability drift

For each final question, take the largest absolute change among its option
probabilities relative to BF16. These differences are in **percentage points**:

| Statistic | Q8_0, llama.cpp CUDA | Calibrated H128/Q4-G32-DOT4, native CPU | Selected Q4 + Q8 head, native CPU |
| --- | ---: | ---: | ---: |
| Median | 0.40 | 3.02 | 2.96 |
| 90th percentile | 1.85 | 12.63 | 12.57 |
| 95th percentile | 2.59 | 17.77 | 17.50 |
| 99th percentile | 4.98 | 33.24 | 30.75 |
| Maximum | 19.06 | 84.24 | 79.12 |
| Questions exceeding 10 points | 15 / 13,601 | 2,123 / 13,601 | 2,039 / 13,601 |
| Questions exceeding 20 points | 0 / 13,601 | 527 / 13,601 | 488 / 13,601 |

The similar aggregate accuracy hides different individual decisions: Q4 changes
the BF16 argmax on 7.90% of final questions, versus 1.17% for Q8. Some changes
correct BF16 errors while others introduce errors. Rare large deviations remain:
on final browser question `row-001703`, one option rises from 11.78% with BF16 to
96.02% with Q4, while the BF16 favorite falls from 86.03% to 3.16%. Accuracy alone
cannot validate confidence thresholds used by an application.

The original Snake smoke case is development material. Re-evaluating its exact
same prompt with current references and the selected pure-Q4 recipe gives:

| Direction | BF16, llama.cpp CUDA | Q8_0, llama.cpp CUDA | Calibrated H128/Q4-G32-DOT4 | Uncalibrated H128/Q4-G32-DOT4 |
| --- | ---: | ---: | ---: | ---: |
| Up | 8.66% | 8.27% | 9.37% | 13.37% |
| Left | 70.81% | 70.74% | 66.09% | 43.84% |
| Right | 20.53% | 20.99% | 24.53% | 42.79% |

For Left, calibration closes about 82.5% of the original BF16 probability gap:
26.97 points becomes 4.72 points. The selected mixed-precision recipe gives
62.77% for Left on this inspected case; its selection uses the development
suite rather than this single example.

## Matched CPU latency

All time measurements use the current llama.cpp revision and run **sequentially**
through `scripts/benchmark-inference-seq.ps1`: one warmup and three measured
runs per case, eight threads pinned to physical CPUs 0–7 (the 96 MiB V-cache CCD)
on the Ryzen 9 9955HX3D. Each case is B1 with identical explicit input tokens,
the full vocabulary head and one predicted output, hence **zero decode
forwards**. Model loading, tokenization and application probability postprocessing
are excluded; these are first-prompt-forward latencies, not end-to-end API times.
No resident quality workers run concurrently with this comparison.

| Representation / engine | 74 input tokens, median ms [min, max] | 579 input tokens, median ms [min, max] |
| --- | ---: | ---: |
| H128/Q4-G32-DOT4, uncalibrated / native CPU | 49.93 [47.66, 54.49] | 220.71 [219.90, 221.45] |
| H128/Q4-G32-DOT4, calibrated / native CPU | 50.07 [48.09, 56.72] | 222.88 [220.37, 224.13] |
| Calibrated Q4 + Q8 head / native CPU | 51.48 [50.92, 52.50] | 227.77 [225.26, 239.10] |
| BF16 / llama.cpp CPU | 81.11 [78.83, 81.87] | 546.49 [534.94, 550.23] |
| Q8_0 / llama.cpp CPU | 95.43 [95.18, 100.56] | 761.58 [755.73, 768.52] |

Calibration preserves the Q4 runtime format. Its latency difference from
uncalibrated Q4 is small in these three-run samples; this is not a formal
performance-equivalence test. The Q8 head adds modest latency on these prompts.
The measured BF16/Q8 ordering reflects this llama.cpp build and these CPU
workloads, and cannot be inferred from their quantization labels alone.
Raw profiles, exact commands, hashes and min/max observations are retained in
`benchmarks/qev-calibration-2026-10-21/release-evidence/performance-raw.tar.gz`.

## Reproduction

Raw logits, command lines, intermediate fitting moments and run logs remain
under `benchmarks/qev-calibration-2026-10-21`; checkpoints remain under
`models/qev-0.8b`. These directories are ignored. Compact results, hashes, exact
calibration source-row selections and pinned source revisions are saved under
`docs/results/qev-calibration-2026-10-21`. The historical source archive is retained locally
under `benchmarks/qev-calibration-2026-10-21/release-evidence`.

Use Python 3.12 with the pinned packages from `python-requirements.txt` in the
results directory. Core steps, using fresh output paths, are:

```sh
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux --target qwen35_cpu_pack qev-native-study -j 8

cmake -S tools/llama-comparison -B build-qev-llama-current -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR="$PWD/.cache/llama-current"
cmake --build build-qev-llama-current --target qev-llama-study -j 8

# Separate offline GPU reference/capture tool; not a native engine backend.
cmake -S tools/llama-calibration-cuda -B build-qev-cuda-current -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_SOURCE_DIR="$PWD/.cache/llama-current" \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-qev-cuda-current --target qev-teacher-cuda -j 8

R=benchmarks/qev-calibration-2026-10-21
PY=.cache/qev-venv/bin/python
$PY scripts/qev-study.py prepare --model-dir models/qev-0.8b \
  --upstream "$R/upstream" --out "$R/eval" --max-context 32768
$PY scripts/split-qev-study.py --prepared "$R/eval" --out "$R/panels"
$PY scripts/prepare-qev-calibration.py --sources "$R/train-sources" \
  --eval-items "$R/eval/items.json" --model-dir models/qev-0.8b \
  --out "$R/calibration-final"
$PY scripts/collect-qev-calibration.py --prepared "$R/calibration-final" \
  --teacher build-qev-cuda-current/bin/qev-teacher-cuda \
  --weights models/qev-0.8b/qev-0.8b-bf16.gguf --out "$R/capture"

$PY scripts/fit-block-covariance.py "$R/capture/n1024/covariance" \
  --out "$R/capture/n1024/fitted-covariance" --damping 0.01
build-linux/qwen35_cpu_pack --hf-model-dir models/qev-0.8b \
  --output models/qev-0.8b/model-cal-1024-block128.q35h \
  --importance-dir "$R/capture/n1024/importance" \
  --covariance-dir "$R/capture/n1024/fitted-covariance"
# Repeat with a new output filename and --q8-head and/or --q8-gates for ablations.

taskset -c 0-7 build-linux/qev-native-study --model-dir models/qev-0.8b \
  --weights models/qev-0.8b/model-cal-1024-block128.q35h \
  --jobs-file "$R/panels/development/jobs.tsv" --letters-file "$R/letters.txt" \
  --output "$R/1024-block128-development.tsv" --threads 8 --max-context 32768
$PY scripts/qev-study.py score --prepared "$R/panels/development" \
  --upstream "$R/upstream" --logits "$R/1024-block128-development.tsv" \
  --reference "$R/current-bf16-gpu.tsv" --out "$R/1024-block128-development-score.json"
```

Generate BF16/Q8 references with the same jobs and letters using
`qev-teacher-cuda --weights ... --jobs-file ... --letters-file ... --output ...
--threads 4 --max-context 32768`. `qev-llama-study` accepts the same flags for a
CPU reference. The recorded command log and archived coordinators specify the
complete candidate ordering, frozen selection, balanced final-job partitioning
and paired cluster-bootstrap analysis. Native final workers run on physical CPU
affinities 0–7 and 8–15 with eight threads each; related variants remain grouped
for confidence intervals. Any subsequent performance comparison must use
`scripts/benchmark-inference-seq.ps1` sequentially, with one warmup and three
measured runs, rather than timings from these resident quality workers.
