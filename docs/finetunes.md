# Qwen3.5-0.8B fine-tunes on CPU

## Import and evaluate

The importer accepts a Hugging Face repository ID or a local safetensors
directory. A PEFT LoRA adapter is merged into its original base in FP32 before
quantization. The native artifact stores the text backbone with a tied output
head. Separate trained heads, resized vocabularies, and other architectures
require engine changes. Use a Qwen byte-level BPE tokenizer. Use the prompt
format used in training, including delimiters and whitespace.

Build the Release targets `qwen35_cpu_pack` and `qwen35_cpu_server`. For a local
full checkpoint, conversion alone uses Python 3.11+ and the packer. Hub access
requires `huggingface_hub`. Evaluation and adapter import also use CPU PyTorch,
Transformers (tested with 5.5.4), Accelerate, safetensors and PEFT. The command
below downloads a public document reranker at a fixed revision, converts it,
starts a loopback server and scores the original and native models on the same
labeled examples. Use an unused output directory. In a fresh Python environment,
install a CPU wheel of PyTorch and the reference tools first.

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "transformers==5.5.4" peft accelerate safetensors
```

```sh
python scripts/import-finetune.py infgrad/Prism-Qwen3.5-Reranker-0.8B \
  --revision c60729783cb2f0662c0b054a02964402a211e3f4 \
  --packer build/qwen35_cpu_pack --output models/prism \
  --classifier-profile configs/classifiers/prism.json \
  --eval-panel tests/data/finetunes/prism.json \
  --server-binary build/qwen35_cpu_server --threads 4
```

For an adapter, pass its path or repo ID as the source. Use `--base` to override
the base in `adapter_config.json` and `--base-revision` to pin its revision.
Supply `--importance-dir` and optionally `--covariance-dir` to use compatible
calibration files produced for this fine-tune. Without calibration the packer
uses unweighted MSE16. Keep evaluation examples separate from calibration and
training. The importer stages the package next to its destination and publishes
it after the requested commands succeed. `import.json` records the model,
source and output hashes. `quality.json` records task scores and per-case
predictions. Running an evaluation produces a report without enforcing a
universal acceptable quality threshold.

## Task panels

A panel contains a `task`, a Python `str.format` prompt template, and `cases`
with `fields` and `expected`. Templates are rendered once and used in both
engines. Reference inference uses the source checkpoint's stored precision.
The evaluator checks that native and HF prompt token counts agree. For an
existing package run `scripts/evaluate-finetune.py --reference HF_DIR --server
http://127.0.0.1:8080 --panel PANEL --output scores.json`.

| Task | Expected output | Measurements |
| --- | --- | --- |
| classification | One label from the panel's label-to-token mapping | Accuracy, macro-F1, reference agreement, probability drift. With a positive label and query groups, mean reciprocal rank. |
| asr | Corrected transcript | Word error rate and counts of inserted, deleted and substituted whitespace-delimited words. Case and punctuation are retained. |
| extraction | JSON object | Fraction of expected fields with correct values, and fraction of outputs parsed as objects. Extra keys are ignored. |

ASR and extraction use greedy generation with the same token limit. Set
`max_new_tokens` in the panel to suit the task. The bundled Prism and QEV
panels are small, authored test sets. The QEV panel targets the released
calibrated package and its System One decision protocol.

## Single-token classifiers

Put a `classifier.json` alongside the model or pass it to the importer.
For Prism it contains `{"labels":{"irrelevant":"no","relevant":"yes"}}`.
Each value must encode to exactly one distinct token. Whitespace is significant.
The model determines which tokens represent the classes.

```sh
build/qwen35_cpu_server --model-dir models/prism --threads 4
```

In a second terminal, use the first query/document pair from the test panel.

```sh
python - <<'PYTHON' > formatted-prism-request.json
import json
from pathlib import Path
panel = json.loads(Path("tests/data/finetunes/prism.json").read_text())
print(json.dumps({"prompt": panel["prompt_template"].format(**panel["cases"][0]["fields"])}))
PYTHON
curl http://127.0.0.1:8080/v1/classify -H 'Content-Type: application/json' \
  --data-binary @formatted-prism-request.json
```

The request contains one string field, `prompt`, with the fully rendered input.
The response contains the winning `label`, a `probabilities` object normalized
over configured labels, and token usage. For Prism the relevant score equals
the sigmoid of the yes/no logit difference. This endpoint performs one prefill,
returns zero generated tokens and supports the server's prefix cache. The DOT4
head evaluates the selected embedding rows. `--decision-head full` forces the
full projection for comparison. Multi-token candidates need sequence scoring.

## Shared prefill for short decisions

`--prefill-requests N` (1–8, default 1) lets the scheduler combine already
available selected-readout requests with at most 128 uncached tokens remaining
each. It adds no deliberate waiting period. It batches the MLP projections
across request chunks. Attention, convolution, DeltaNet and prefix publication
stay private to each sequence. Full prefix hits and full-head requests use
their usual paths. The setting controls a different operation from
`--batch-width`, which controls offline job concurrency. `GET /v1/cache`
exposes `shared_prefill_ticks` and `shared_prefill_requests`. A request that
participates in two grouped chunks counts twice.

See the [fine-tune validation and CPU measurements](benchmarks/finetune-serving.md).
