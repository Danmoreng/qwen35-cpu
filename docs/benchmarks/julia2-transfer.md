# Does Qev's advantage over Julia-2 transfer beyond Jev Skills?

Qev's advantage survives a change of publisher, a screen for common benchmark items and a German test generated after both checkpoints were released. The gap varies by task. Both models route new tickets by an explicit branch code reliably, while Qev handles numerical thresholds and requests for clarification more accurately under the tested prompts.

| Evaluation | Julia-2, original precision | Labqoat Qev-0.8B BF16 |
| --- | ---: | ---: |
| Jev Skills frozen final, 13,601 questions, macro accuracy | 56.94% | 68.92% |
| NVIDIA When2Call MCQ, all 3,652 questions | 1,213 (33.21%) | 1,647 (45.10%) |
| When2Call after removing identified common BFCL sources, 3,091 questions | 1,029 (33.29%) | 1,392 (45.03%) |
| Frozen German routing, 64 cases in 16 domains | 15 (23.44%) | 28 (43.75%) |
| Newly generated German tickets, 128 cases in 64 pairs | 107 (83.59%) | 120 (93.75%) |

Compare models within each row. Jev Skills averages skill scores, while the other rows give equal weight to each item. The When2Call rows overlap. The existing H128 Qev implementation scored 1,651/3,652 (45.21%) on the complete NVIDIA panel. [Jev Skills comparison](julia2-jev-skills.md) · [When2Call protocol](when2call.md) · [All counts and paired intervals](../results/julia2-transfer-2026-11-22/scores.json)

## What can explain the Jev Skills result?

Labqoat reports a dedicated training phase for games, rules and spatial tasks, and describes using benchmark scores to select checkpoints and weight mixtures. Jev Skills includes the publisher's own JevSpatial and Rules-7 tasks. Labqoat also discloses shared text between training rows and indirect prompt-injection items in JevExtra, and shared sources for its other skills. Its explicit 16-token overlap statement concerns Benchmark Heaven and the separately named JevBench. The available evidence cannot identify which, if any, of the 13,601 final questions occurred with their gold answers during Qev training. Our split froze Q4 quantization choices and has no bearing on the original fine-tune's exposure. [Qev publisher disclosure](https://huggingface.co/labqoat/qev-0.8b/blob/main/README.md#benchmarks) · [Jev Skills dataset](https://huggingface.co/datasets/labqoat/jev-skills/tree/fda80b2a75fca3fa3e5d3f6378acc2a6c04d4032) · [Prior provenance audit](jevbench-v1.md#precision-and-training-provenance-audit)

Julia's publisher describes training for stable decisions under option reordering, negation and distractors. The checkpoints differ in their data, objectives and readout. The comparisons measure their released behavior and cannot isolate a causal effect of any one training decision. [Julia-2 training provenance](https://huggingface.co/SupersonicLabs/Julia-2/blob/b3e610b47d140cc4fcbd00d7b58b86184a13bcff/provenance.json)

## NVIDIA When2Call and shared sources

We rendered the identical frozen states, instructions, options and candidate ordering used in the previous Qev study with Julia's own tokenizer and prompt. The full test contains 1,295 gold tool calls, 1,062 requests for information and 1,295 unanswerable cases. Qev BF16 identified 94.29%, 29.94% and 8.34% respectively. Julia identified 85.48%, 6.59% and 2.78%. A constant choice of the most frequent response category scores 35.46%. This decision-model adaptation asks the model to choose among four complete candidates. Both tested models struggle to identify when the available tools cannot answer. [Original NVIDIA benchmark](https://github.com/NVIDIA/When2Call) · [Pinned NVIDIA data](https://huggingface.co/datasets/nvidia/When2Call/tree/0582f7749df63a96fdc3070932e83e72396ace53)

NVIDIA When2Call and Jev Skills both draw on Berkeley Function Calling Leaderboard (BFCL) data. We compared all 571 BFCL items in the entire Jev Skills release with each NVIDIA item's BFCL source ID, user question and original user question where available. Text matching ignored case and whitespace. We removed every variant belonging to a matched source group. This excluded 561 NVIDIA rows from 200 groups. On the remaining 3,091 rows Qev BF16 still leads by 11.74 points (95% paired source-group bootstrap interval 10.42–13.06). This screen detects known cross-benchmark reuse. Semantically similar prompts and overlap with either model's training data can remain.

## German support decisions

On the existing 64-case panel, Qev answered 12/16 clarification cases and Julia 0/16. For direct answers Julia led 13/16 to 4/16. These four-way routing questions share a fixed action set and span 16 fictional service domains. Both checkpoints were run anew for this comparison. [Frozen German inputs and answer key](decision-model-followup.md#german-routing)

For a control on memorization, a fresh random seed generated 64 German ticket pairs after both releases. Within a pair, the ticket, rules, team labels and option order stay fixed. Only one fact changes and the correct team switches. The generator fixes the answer key before inference. Rule and option order are shuffled between pairs. Both members must be correct to earn a pair-level hit.

| Both members correct | Julia-2 | Qev BF16 |
| --- | ---: | ---: |
| Match branch code to team | 32/32 | 32/32 |
| Cross a stated waiting-time threshold | 11/32 | 24/32 |
| Total | 43/64 | 56/64 |

The difference across the 128 individual tickets is 10.16 points (95% paired bootstrap interval 4.69–15.63). This tests two synthetic rule templates. Newly generated instances reduce the possibility of memorizing those exact examples; learned familiarity with the rule families remains relevant. For an operational decision, a blinded, human-labeled sample of recent private support tickets would give a better estimate of real routing errors. [Frozen generator and seed](../results/julia2-transfer-2026-11-22/german.py) · [Predictions](../results/julia2-transfer-2026-11-22/predictions.jsonl.gz)

Julia used the validated RTX 5080 Laptop GPU evaluator with F32 GGUF text weights expanded from BF16 and the original FP32 head. The German Qev run used its published BF16 GGUF on the same CUDA device. When2Call reuses the archived Qev Vulkan BF16 and native CPU H128 results. All 3,844 distinct Julia requests and 192 new Qev requests completed without truncation. [Commands](../results/julia2-transfer-2026-11-22/reproduce.md) · [Artifact hashes](../results/julia2-transfer-2026-11-22/manifest.json)
