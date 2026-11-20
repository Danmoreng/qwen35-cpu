# BA Research and Labqoat Qev at 0.8B

November 20, 2026. We compared two independently trained checkpoints bearing the Qev name. All three text suites replay the frozen requests from the [four-model comparison](decision-models-gpu.md) and the [German routing study](decision-model-followup.md). Labqoat led all three suites at matched BF16 and remains the stronger 0.8B reference for this engine. Both BF16 reference configurations ran locally on the RTX 5080 Laptop GPU. The Q8_0 row reuses the earlier measured Labqoat deployment configuration.

| Checkpoint and setting | German routing | Benchmark Heaven | When2Call |
| --- | --- | --- | --- |
| Labqoat BF16 | 28/64 (43.8%) | 180/231 (77.9%) | 1,648/3,652 (45.1%) |
| Labqoat Q8_0 | 25/64 (39.1%) | 179/231 (77.5%) | 1,652/3,652 (45.2%) |
| BA Research BF16, up to 8,192 tokens | 18/64 (28.1%) | 140/231 (60.6%) | 1,264/3,652 (34.6%) |
| BA Research BF16, released 2,048 tokens | 18/64 (28.1%) | 132/231 (57.1%) | 1,221/3,652 (33.4%) |

## What changes with the context limit?

BA's checkpoint sets a 2,048-token maximum for each candidate prompt. Every request first used that exact configuration. When the upstream loader rejected a longer prompt before inference, we retried it with the same weights and an 8,192-token maximum. The released-limit row counts a rejected input as a miss. The next table compares only the same inputs that fit the released limit for BA. The longer-context run is an inference experiment.

| Suite | Inside BA limit | BA correct | Labqoat BF16 correct | Labqoat Q8_0 correct |
| --- | --- | --- | --- | --- |
| german | 64 | 18 | 28 | 25 |
| heaven | 195 | 132 | 159 | 159 |
| when2call | 3538 | 1221 | 1603 | 1607 |

## Which decisions differ?

| German action | Cases | BA correct | Labqoat BF16 correct | Labqoat Q8_0 correct |
| --- | --- | --- | --- | --- |
| call | 16 | 1 | 6 | 6 |
| ask | 16 | 4 | 12 | 10 |
| unsupported | 16 | 1 | 6 | 5 |
| direct | 16 | 12 | 4 | 4 |

| When2Call action | Cases | BA correct, up to 8,192 | Labqoat BF16 correct | Labqoat Q8_0 correct |
| --- | --- | --- | --- | --- |
| tool_call | 1295 | 1143 | 1221 | 1220 |
| request_for_info | 1062 | 45 | 319 | 322 |
| cannot_answer | 1295 | 76 | 108 | 110 |

BA recognized 12 of 16 directly answerable German questions, versus four for Labqoat BF16. Labqoat recognized 12 of 16 necessary clarifications, versus four for BA. Each German action is correct exactly 16 times. On this panel BA chose the direct-answer action 46 times. The German panel consists of newly authored scenarios, and its sample size limits conclusions about production traffic. When2Call tests a choice among four complete responses. Direct answers appear as distractors in this split. All accuracy values use the same ground truth and tie rules for both checkpoints.

## Model and scoring details

- **BA Research.** The authors supply a rank-8 LoRA adapter on Qwen/Qwen3.5-0.8B, a separate learned decision head and a calibration temperature of 0.93348. We used their pinned Open-Jev loader with BF16 backbone inference and the checkpoint's head. It renders a prompt per candidate, scores it from the last hidden state and normalizes the candidate scores with the supplied temperature. Model file hashes matched the publisher's checksums. [Model card](https://huggingface.co/ba-research/Qev1.0-Qwen3.5-0.8B/blob/e55f3373173b7196854adb2e9e7496514c41f702/README.md) · [Open-Jev readout](https://github.com/Zefan-Cai/Open-Jev/blob/bd4118882f733574a3250a4b65fe4d884130c08b/jev/model.py)
- **Labqoat.** The publisher describes a full fine-tune of the text layers of Qwen3.5-0.8B-Base. Its text path scores option-letter logits. We ran the publisher's joint text GGUF through the same llama.cpp CUDA decision endpoint used previously. The BF16 and Q8_0 files each contain 187 weight tensors in the named precision and 133 F32 tensors. The prompt rendering and readout are part of each model's native interface. [Labqoat model and readout](https://huggingface.co/labqoat/qev-0.8b-GGUF/blob/9550cc0d3f0e8f651cced8469ddfb89acc58c26d/README.md)
- **Reproduction.** The German inputs and keys were fixed before this comparison. Benchmark Heaven uses its official TypeSafe adapter and all 231 public v1 text items. When2Call uses all 3,652 public MCQs with the repository's existing prompt builder, deterministic option order and probability argmax. Published counts include any failure in the denominator. [Predictions with context eligibility](../results/qev-08-comparison-2026-11-20/predictions.csv) · [Scores, coverage and paired results](../results/qev-08-comparison-2026-11-20/scores.json) · [Model, runtime and input revisions](../results/qev-08-comparison-2026-11-20/provenance.json)
