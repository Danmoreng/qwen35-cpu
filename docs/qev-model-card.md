---
license: apache-2.0
base_model: labqoat/qev-0.8b
base_model_relation: quantized
tags:
- cpu
- quantized
- qwen3.5
- decision-model
- custom-format
---
# qev-0.8b H128/Q4-G32-DOT4

Calibrated, text-only **425 MB** quantization of
[labqoat/qev-0.8b](https://huggingface.co/labqoat/qev-0.8b), a decision fine-tune
of Qwen3.5-0.8B-Base. Run it on CPU with the native
[qwen35-cpu server](@ENGINE_URL@/tree/@ENGINE_REVISION@) and its
**`POST /v1/systemone`** API. No Python, PyTorch or CUDA is needed by the server.
The `.q35h` artifact is a custom format; use the matching native engine.

## Download and serve

Build the engine revision supporting this model, then download this repository
into a model directory (the `hf` CLI is only a download tool):

```sh
git clone https://github.com/Danmoreng/qwen35-cpu.git
cd qwen35-cpu
git checkout @ENGINE_REVISION@
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target qwen35_cpu_server -j 8
hf download danmoreng/qev-0.8b-H128-Q4-G32-DOT4 --local-dir models/qev
build/qwen35_cpu_server --model-dir models/qev --threads 8 \
  --decision-encoding per_question
```

```sh
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' -d '{
    "state": {"ticket": "My card was charged twice. Please refund the duplicate."},
    "encoding": "per_question",
    "questions": {
      "department": {
        "type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "tech": "bugs", "sales": null}
      },
      "urgent": {"type": "noul", "instructions": "Is this urgent?"}
    }
  }'
```

The response contains `model`, `answers`, `usage` and `latency_ms`.
Choice answers include `choice`, `probabilities`, `confidence`, `x_p_max` and
`certainty`. Noul answers contain the probability of true in `noul`; score
answers contain an expected level index, legend and probability distribution.
`usage.output_tokens` is zero. The server supports string/object/array states,
up to 32 questions per request, 255 choice options, and 2–10 score levels.
Images are unsupported. Context overflow is rejected instead of truncating.
See the [API documentation](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/server.md).

`per_question` matches the quality evaluation and keeps each question independent
of the other questions. The compatibility default is `joint`; either encoding
can be selected in the request or as the server default. In this implementation,
joint questions evaluate separate causal prefixes. The server can restore a
registered exact token prefix across decision requests, including the full
attention KV and the GatedDeltaNet state. Each rendered row starts with the
state, followed by the question and options. Changing the ticket in the state
typically rules out reuse of a later fixed rubric. The response reports the
number of restored tokens in `cache.read_tokens`. The native server projects only the required option-letter logits for this
checkpoint, with no autoregressive continuation. `--decision-head full` restores
the full-vocabulary reference; other supported head formats use it automatically.
The adaptive prefill chunk default is 64 tokens for short prompts and 128 for
longer prompts. `--prefill-chunk 128` selects a fixed chunk explicitly. The [Ryzen decision tuning measurements](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/decision-tuning-9955hx3d.md) compare settings for short prompts and concurrent routing.

## Format and calibration

- File: `model.q35h`, **424,964,864 bytes**; tensor payload 424,934,656 bytes.
- SHA256: `@SHA256@`.
- Format: **H128/Q4-G32-DOT4**, signed 128-channel transform and 32-weight
  scale groups, activation-weighted MSE16 with block-128 covariance compensation.
- All large matrices, including the tied embedding/output matrix, use Q4.
  Small norms and recurrent parameters retain FP32; KV uses FP16.
- Calibration: **1,024 independent training/program-generated requests,
  252,138 tokens**, damping 0.01. Jev Skills evaluation inputs/labels were not
  used for fitting. Settings were selected on a separate development panel.
- Publisher temperatures remain choice **1.7**, noul **0.9**, score **1.25**.
  Weight calibration does not refit these output temperatures.
- Source model revision: `6334efd2d556cad22a3040049efb2887a60e7fb2`.

`quantization.json` records source/checkpoint hashes, calibration sources and
licenses, fitting parameters and independent evaluation provenance.

## Quality

Author-published **Jev Skills**, grouped independent final panel: **13,601
questions / 9,133 requests**, per-question encoding, unchanged temperatures.
Macro accuracy is the upstream scorer's average across scored skills; this is
not the model card's Benchmark Heaven score. BF16/Q8 scoring uses **llama.cpp
CUDA** at revision `9b56ccf00d755b0bf780140f6352a67db1ac0378`; our Q4 uses
the **native CPU engine** in the frozen study snapshot. The serving revision
includes subsequent upstream CPU optimizations; its 19 smoke questions still
match the original rounded probabilities exactly. A separate 225-question CPU/CUDA reference check is
included in the study. A subsequent 256-question regression spanning all 22
skills (up to 8,603 tokens) finds bit-identical FP32 option logits for the
selected readout and fixed chunk 128 relative to full readout/adaptive chunks.
The full 13,601-question quality score below was not rerun for this release.
Sizes below are tensor payload, decimal MB.

| Representation / scoring backend | Tensor MB | Macro accuracy | Mean option KL to BF16 |
| --- | ---: | ---: | ---: |
| BF16 / llama.cpp CUDA | 1,505.78 | 68.92% | 0 |
| Q8_0 / llama.cpp CUDA | 800.88 | 68.87% | 0.000496 |
| **H128/Q4-G32-DOT4, calibrated / native CPU** | **424.93** | **68.74%** | **0.023169** |
| H128/Q4-G32-DOT4, uncalibrated / native CPU | 424.93 | 67.61% | 0.040394 |

Calibration gains **1.13 accuracy percentage points** and reduces mean option
KL by **42.6%** without increasing size. The paired 95% interval for the
accuracy gain is +0.45 to +1.79 points. The Q4-minus-Q8 accuracy interval is
−0.68 to +0.45 points; similar point estimates do not establish equivalence.

**Probability fidelity remains materially below Q8.** Taking the largest option
probability difference per question relative to BF16, the median is **3.02
percentage points**, the 95th percentile **17.77**, and the maximum **84.24**.
Q8's corresponding median/95th percentile are 0.40/2.59 points. Q4 changes the
BF16 argmax on 7.90% of questions, versus 1.17% for Q8. Evaluate confidence
thresholds on your own workload. The source model's probabilities are also
estimates, not guarantees about real-world outcomes.

## CPU performance

On the **Ryzen 9 9955HX3D**, the calibrated checkpoint answered a 74-token ticket in **40.29 ms** and a 579-token ticket in **180.07 ms**. Both were the first request after a native server became ready. Across 64 German routing requests (178–216 tokens) sent serially to a resident server, median latency was **57.50 ms**, down from **63.08 ms** before the two prefill optimizations.

The November 20, 2026 measurements used eight physical cores, the adaptive chunk size, one warmup and three measured passes. Timings include localhost HTTP, tokenization and option scoring. Model loading is excluded. All 66 inputs produced identical returned answers and probabilities across the three engine builds. [Paired timings and methodology](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/laptop-refresh-9955hx3d.md#qev-08b-decision-latency) · [Earlier llama.cpp and concurrency comparisons](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/decision-http.md)

## Source and limitations

Original model by **labqoat**, based on **Qwen3.5-0.8B-Base** by Qwen. Changes:
text-only extraction, independent activation calibration and H128/Q4-G32-DOT4
packing. Vision weights are omitted. The publisher declares Apache-2.0 for the
model; see the [upstream model card and training-data terms](https://huggingface.co/labqoat/qev-0.8b#training-data-and-licenses).
The native protocol port includes Apache-2.0 code adapted from decider-ai 1.9.0;
the engine's notices and dependency licenses are included with its source.

Full methods, unsuccessful candidates, per-skill results, probability tails,
source revisions and reproducible commands:
[quality study](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/decision-quality.md),
[current HTTP study](@ENGINE_URL@/blob/@ENGINE_REVISION@/docs/benchmarks/decision-http.md).
