# Julia-2 and Qev on the complete Jev Skills text suite

Qev remains the better choice for the engine's text decision endpoint. On the frozen final panel, shipped Qev H128 leads unquantized Julia-2 by **11.80 percentage points** in macro accuracy. Qev wins 19 of the 21 skills that contribute to the score, including tool selection, browser actions, grounding and prompt-injection detection. Julia wins on admission rules and patent phrases. The [earlier native probe](../julia-2-port.md) establishes that a Julia text port is feasible. These quality results give little reason to prioritize it for general text routing.

The [transfer comparison](julia2-transfer.md) tests whether the ranking holds on NVIDIA When2Call and newly generated German tickets and examines shared sources and checkpoint selection.

## Results

| Model | Final macro accuracy ↑ | Final macro Brier ↓ | Entire suite macro accuracy ↑ |
| --- | ---: | ---: | ---: |
| Julia-2, original BF16 text weights and FP32 head | 56.94% | 0.56455 | 57.06% |
| Labqoat Qev-0.8B, BF16 | 68.92% | 0.40971 | 68.94% |
| Labqoat Qev-0.8B, H128/Q4-G32-DOT4 | 68.74% | 0.41249 | 68.78% |

The primary comparison uses the same 9,133 requests and 13,601 questions as the [Qev calibration study](decision-quality.md#frozen-final-results). The last column also includes all 1,553 development questions, for **15,154 questions across 10,168 requests**. Qev's quantization settings were selected using the development panel. The final panel stayed separate throughout selection.

The upstream scorer gives every scored skill equal weight. On the final panel, 13,051 labeled questions contribute to this 21-skill average. The suite reports detection of missing facts separately. Brier is the mean per-skill squared error of the option probabilities. A paired bootstrap over 8,463 groups puts Qev H128's final accuracy lead at **10.80 to 12.95 points** (95% interval, 1,000 resamples). Qev BF16's lead is 10.93 to 13.05 points. These intervals describe variation within this suite. [Paired estimates](../results/julia2-jev-skills-2026-11-22/paired.json)

| Final category | Julia-2 | Qev BF16 | Qev H128 |
| --- | ---: | ---: | ---: |
| Agents | 61.73% | 74.18% | 72.99% |
| Judging | 60.18% | 71.91% | 71.62% |
| Rules | 56.22% | 63.50% | 63.66% |
| Safety | 69.57% | 80.62% | 81.39% |
| Games and spatial reasoning | 51.07% | 67.90% | 67.56% |

For tool selection, Julia scores **80.23%** against **85.55%** for Qev H128. For browser actions the scores are **72.99% vs 88.50%**, and for prompt-injection detection **71.70% vs 88.68%**. Julia's two wins are **93.24% vs 89.58%** on admission rules and **39.72% vs 34.65%** on patent phrases. These are text decisions on the Qev publisher's suite. Image and audio decisions would need their own evaluation. [All final skills and separate missing-fact metrics](../results/julia2-jev-skills-2026-11-22/scores.json) · [Entire suite](../results/julia2-jev-skills-2026-11-22/complete-suite.json)

## Execution and checks

- Downloaded the original Julia-2 checkpoint and expanded its BF16 text weights to F32 GGUF. On an RTX 5080 Laptop GPU, llama.cpp offloaded all 26 layers and used F32 keys and values. An offline evaluator applied Julia's own FP32 255-option head to the final normalized hidden state. The final panel took 4 min 44 s, and the development panel took 28 s, including model loading in each run. [Pinned checkpoint](https://huggingface.co/SupersonicLabs/Julia-2/tree/b3e610b47d140cc4fcbd00d7b58b86184a13bcff)
- Used the publisher's tokenizer, prompt and option ordering. Every question ran with fresh state. All 5,292,277 input tokens across both panels were evaluated, with a longest prompt of 8,607 tokens and a context limit of 16,384. The two models saw the same states, questions and choices through their own prompt formats. Qev's saved GPU BF16 and native H128 outputs were rescored using its published temperatures and the identical upstream scorer. Both reproduced the original final scores. [Pinned suite](https://huggingface.co/datasets/labqoat/jev-skills/tree/fda80b2a75fca3fa3e5d3f6378acc2a6c04d4032)
- Matched the publisher runtime's FP32 reference choice on **30/30** cross-checks, with a maximum option-probability difference of 0.0601 percentage points. Separately, the evaluator matched **299/300** stored publisher fixture choices, above the publisher's 99% threshold. The differing binary decision was near even in both implementations. The largest probability difference across those 300 fixtures was 1.03 points. [Validation](../results/julia2-jev-skills-2026-11-22/validation.json)
- Published per-question option probabilities for both panels, along with the evaluator, reproduction commands and hashes. [Final predictions](../results/julia2-jev-skills-2026-11-22/julia-predictions.jsonl.gz) · [Development predictions](../results/julia2-jev-skills-2026-11-22/development-predictions.jsonl.gz) · [Reproduction](../results/julia2-jev-skills-2026-11-22/reproduce.md) · [Manifest](../results/julia2-jev-skills-2026-11-22/manifest.json)
