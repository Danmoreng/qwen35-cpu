# Decision models on a local RTX 5080 Laptop GPU

On November 20, 2026, we compared three Qwen-based 0.8B decision models and Liquid's d1-2B on the same three text suites. Qev made the most correct decisions on the 231-item Benchmark Heaven set. Liquid led on base-question accuracy, consistency and tool routing.

| Model and weights | Benchmark Heaven correct | Brier ↓ | JevBench base accuracy | JevBench-mini coherence / 100 | When2Call correct |
| --- | --- | --- | --- | --- | --- |
| Qev-0.8B Q8_0 | 179/231 (77.5%) | 0.329 | 84.9% | 65.1 | 1,652/3,652 (45.2%) |
| Liquid d1-2B Q8_0 | 177/231 (76.6%) | 0.285 | 88.4% | 73.8 | 3,295/3,652 (90.2%) |
| Mapika Decider-0.8B BF16 | 153/231 (66.2%) | 0.485 | 82.4% | 70.8 | 1,745/3,652 (47.8%) |
| EvalEngine Decision-0.8B Q8_0 | 148/231 (64.1%) | 0.438 | 81.5% | 70.1 | 1,604/3,652 (43.9%) |

Each primary run attempted all 231 Benchmark Heaven decisions, 1,200 JevBench-mini consistency tests across 240 cases, and 3,652 When2Call decisions. The base-accuracy column uses the same 1,938 labeled questions for every model. EvalEngine's example command uses a 2,048-token context. That setting rejected 36 Benchmark Heaven items and 258 When2Call items. A second complete run at 16,384 tokens answered 231/231 and 3,652/3,652. The tables use the larger context. Both runs are retained.

## Where the models differ

| Model | Call a tool | Ask for information | Cannot answer |
| --- | --- | --- | --- |
| Qev-0.8B Q8_0 | 1,220/1,295 (94.2%) | 322/1,062 (30.3%) | 110/1,295 (8.5%) |
| Liquid d1-2B Q8_0 | 1,179/1,295 (91.0%) | 885/1,062 (83.3%) | 1,231/1,295 (95.1%) |
| Mapika Decider-0.8B BF16 | 1,124/1,295 (86.8%) | 449/1,062 (42.3%) | 172/1,295 (13.3%) |
| EvalEngine Decision-0.8B Q8_0 | 1,217/1,295 (94.0%) | 267/1,062 (25.1%) | 120/1,295 (9.3%) |

On the hard Benchmark Heaven tier, Qev solved 64/111 and Liquid solved 61/111. Both solved all 48 easy items. Across the full set they got 157 items right in common. Qev alone got 22 right and Liquid alone got 20 right. Liquid's lower Brier score indicates that its probabilities were closer to the observed outcomes on this set. When2Call rewards both taking action and recognizing a missing prerequisite. The category breakdown shows that the largest gap comes from recognizing unanswerable requests.

| Model | Representation | Batch isolation | Probability laws | Logic | Choice | Overall 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| Qev-0.8B Q8_0 | 75.3 | 70.8 | 34.6 | 74.2 | 70.8 | 62.1–68.1 |
| Liquid d1-2B Q8_0 | 81.9 | 100.0 | 42.6 | 73.8 | 70.8 | 71.4–76.4 |
| Mapika Decider-0.8B BF16 | 74.4 | 100.0 | 39.7 | 62.1 | 77.8 | 68.4–73.3 |
| EvalEngine Decision-0.8B Q8_0 | 73.3 | 100.0 | 32.1 | 71.2 | 73.6 | 67.3–72.6 |

A model that processes questions independently gets batch isolation by construction. Qev used its joint text encoding here. The publisher also provides a per-question encoding, so the batch result applies specifically to joint serving. [Qev encoding and serving options](https://huggingface.co/labqoat/qev-0.8b-GGUF/blob/9550cc0d3f0e8f651cced8469ddfb89acc58c26d/README.md)

For tool routing, d1-2B is the strongest candidate for a follow-up on application-specific traffic. Qev remains the 0.8B reference for general decisions. The two-item gap on the 231-item set gives little reason by itself to choose between those two models.

## Which JevBench?

Benchmark Heaven's downloadable English text set contains 72 original, 48 easy and 111 hard decisions. The 231 decisions include related variants. The v1.4 official evaluation added 308 sealed decisions. The later v1.7.6 release describes a 1,500-item self-hosted evaluation with 300 common public anchors and 1,200 sealed decisions. The public inputs verified for this run are the earlier 231. The official data card distributes system-level results. Benchmark Heaven also runs image and audio tracks. [Public text fixtures](https://github.com/fstandhartinger/jevbench/tree/609a664e81f46e03731600bd7bd9f1b7a61257c5/datasets/public) · [v1.4 methodology](https://github.com/fstandhartinger/jevbench/blob/609a664e81f46e03731600bd7bd9f1b7a61257c5/docs/METHOD-v1.4.md) · [v1.7.6 draw methodology](https://benchmarkheaven.com/jev-models/v1.7.6) · [Data card](https://benchmarkheaven.com/jev-models/data)

The independent JevBench project supplies 240 synthetic cases in 12 decision domains. Its mini suite tests 50 relations on 24 cases each. Its full suite applies all 50 relations to all 240 cases, yielding 12,000 tests on the same base cases. This comparison ran the frozen mini suite and scored the 1,938 base questions with known targets. Coherence measures whether probabilities obey relations such as invariance under renaming. A uniform distribution scores highly on several relations, so read coherence alongside base-question accuracy. The repository's earlier Jev Skills evaluation offers a third text corpus with 13,601 questions on the final split. [Independent JevBench](https://github.com/JevBench/jevbench/tree/e18733694623aa93058e279c3534f8b8e2edefa3) · [Jev Skills in this repository](decision-quality.md)

## Evaluation protocol

- **Benchmark Heaven.** All models received the same ordered public tasks through the upstream TypeSafe adapter. The upstream scorer produced top-choice accuracy and mean multiclass Brier score. Brier measures the squared error of the full probability distribution, with lower values better.
- **JevBench-mini.** The official CLI ran the frozen suite at seed 0, one trial and fixed tolerance 0.05. Its scoring code computes the overall coherence score and a case-bootstrap interval. Its leaderboard accuracy script excludes unknown targets and splits credit for tied maxima. The evidence includes the dimensions, subgroups and coverage for each model.
- **When2Call.** All 3,652 NVIDIA MCQs used this repository's frozen request builder, deterministic option shuffle and four candidate replies. The score uses the highest returned option probability with ties broken by option key. An invalid response counts as a miss. The three gold categories are calling a tool, requesting more information and reporting that the request cannot be answered. A direct reply is also offered as a distractor. The 3,423-item filtered cohort is recorded separately. [Request construction and scoring](../../scripts/when2call.py) · [NVIDIA dataset](https://huggingface.co/datasets/nvidia/When2Call/tree/0582f7749df63a96fdc3070932e83e72396ace53)
- **Inference.** One model served local loopback requests at a time. Qev used the publisher's joint text GGUF. Liquid used its decision GGUF. Both ran on the stock llama.cpp CUDA decision endpoint with 16,384 context tokens. EvalEngine used its published no-thinking system prompt, serialized each question and its ordered options as JSON, and scored every option letter from the final-position logits before softmax. Mapika used the publisher's BF16 PyTorch scorer on CUDA, state first, with independent questions, isolated score levels, temperature 1.03 and CUDA graphs disabled. The serving choices affect results for multi-question consistency tests.
- **Interpretation.** These scores describe the named model artifacts and serving configurations. Three models used Q8_0 weights and Mapika used BF16. The inputs are public and independent training-data overlap checks were unavailable. GPU quality results alone do not establish CPU latency or the quality of a future four-bit conversion.

[Scores and coverage](../results/decision-gpu-2026-11-20/scores.json) · [Model, dataset and runtime revisions](../results/decision-gpu-2026-11-20/provenance.json) · [Benchmark Heaven predictions](../results/decision-gpu-2026-11-20/heaven-predictions.csv) · [When2Call predictions](../results/decision-gpu-2026-11-20/when2call-predictions.csv)

### Repeating the comparison

Use the revisions and file hashes in the provenance manifest. For the two decision GGUFs, start the pinned CUDA build of `llama-server` with the model file, `--host 127.0.0.1 -ngl 999 -np 1 -c 16384 -b 8192 -ub 8192 --no-warmup`. Point the official CLI for each benchmark at the same local server. Benchmark Heaven expects the server root for its `typesafe` adapter. JevBench-mini expects the full `/v1/systemone` URL. They are separate Python projects sharing the module name `jevbench`, so run each from its own checkout.

For EvalEngine, use the GGUF card's system prompt, disable thinking, and serialize the insertion-ordered object `{"state": ..., "question": ..., "options": [...]}` with Python's `json.dumps(..., ensure_ascii=False)` and its default spacing. Each option has a letter label, key and description. Preserve the request order for choices, use zero-based level numbers for scores, and use “No” then “Yes” for a binary question. Read the next-token logits for all supplied letters, apply a softmax over those letters alone, and convert ordinal probabilities to the expected level. Clear the model state between questions. Mapika uses `Decider(path, device="cuda", use_graphs=False).system_one(state, questions)` from its pinned snapshot. [EvalEngine readout and template](https://huggingface.co/evalengine/decision-0.8b-gguf/blob/ce993792ad283053004bf0230b1709ff7351bb24/README.md) · [Mapika inference](https://huggingface.co/Mapika/decider-0.8b/blob/a0a01d6f8135298f400a8c856b355793012ae971/decider/infer.py)
