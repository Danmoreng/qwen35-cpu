# Q8 tied embedding/output experiment, 2026-09-06

On the regression suite, Q8 improves PPL by **2.38%** and KL by **17.04%**.
The cost is **15.65-25.98% lower decode throughput** and **127.14 MB** more
checkpoint storage; prefill is nearly unchanged. Held-out PPL is essentially
unchanged, and Q4_K_M still has better quality. The Q4 standard is retained.

This experiment promotes only the shared embedding/output matrix to Q8_0.
All other quantized matrices, including the recurrent B/A gates, retain the
current calibrated MSE16 H128/Q4-G32-DOT4 recipe. This isolates the large-matrix
change from the [earlier gate experiment](q8-experiments-2026-09-06.md).
It is an explicitly requested measurement, not an automatic standard promotion.

## Storage and execution

The model has one tied `model.language_model.embed_tokens.weight` matrix with
248,320 rows and 1,024 columns, used both for token lookup and output logits.
It is stored once. Q8_0 uses 32 signed bytes plus one FP16 scale per group of
32 weights, in the identity basis. The converter reads the original BF16
weights directly and uses absmax/127 quantization. The existing Q4 calibration
inputs are reused for the other matrices; no larger calibration corpus is used.

| Storage | Calibrated Q4 standard | Q8 head |
| --- | ---: | ---: |
| Tensor payload, bytes | 424,934,656 | 552,074,496 |
| Whole artifact, bytes | 424,964,864 | 552,104,704 |

The increase is **127,139,840 bytes**, or 127.14 MB / 121.25 MiB / 29.92% of
the standard file. The audit confirms that every other tensor's bytes remain
identical. Runtime resident memory is measured separately in `performance.csv`;
it also includes scale caches, activations and request state.

The loader and Q8 dispatch added with the gate experiment already support this
artifact, including a full-logit greedy fallback. No inference source changes
were needed for this experiment. Both candidates use the same frozen executable.
The results measure this Q8 implementation, including its kernel and activation
preparation costs; they do not isolate memory bandwidth or establish the best
possible Q8 speed.

## Quality and speed

| Quality suite / checkpoint | PPL | Mean KL to BF16 |
| --- | ---: | ---: |
| regression / Calibrated Q4 | 16.040874 | 0.09645640 |
| regression / Q8 head | 15.658958 | 0.08002328 |
| regression / llama.cpp Unsloth Q4_K_M | 14.752902 | 0.03469311 |
| heldout / Calibrated Q4 | 4.851430 | 0.07183477 |
| heldout / Q8 head | 4.853280 | 0.06249722 |
| heldout / llama.cpp Unsloth Q4_K_M | 4.659449 | 0.02516250 |

**Regression:** PPL improves by 2.38% and KL by 17.04%. The paired
95% document-bootstrap PPL-ratio interval is 0.9713-0.9809, below 1.
**Held-out:** PPL rises by 0.038% (essentially unchanged); KL improves by
13.00%. Its PPL-ratio interval, 0.9938-1.0119, includes no change. Both KL
difference intervals are below zero. The larger matrix gives a clear KL
benefit on these suites but does not bring this recipe to Q4_K_M quality.

| Workload B / input / output | Q4 prefill tok/s | Q8 prefill tok/s | Change | Q4 decode tok/s | Q8 decode tok/s | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 512 / 128 | 2185.16 | 2164.35 | -0.95% | 120.56 | 89.24 | -25.98% |
| 1 / 4096 / 2 | 1917.47 | 1919.27 | +0.09% | n/a | n/a | n/a |
| 4 / 512 / 128 | 2195.94 | 2177.83 | -0.82% | 318.86 | 253.52 | -20.49% |
| 16 / 512 / 128 | 2196.22 | 2168.51 | -1.26% | 600.46 | 498.18 | -17.03% |
| 1 / 8192 / 128 | 1647.78 | 1649.75 | +0.12% | 68.84 | 58.07 | -15.65% |

**Prefill stays within about 1.3% of baseline. Sustained decode loses
15.65-25.98% throughput**, depending on workload. For one request with 512
input tokens, this is 120.56 to 89.24 tok/s; 25.98% less throughput means
35.09% more decode time per token. Median peak resident memory grows by
about 158.9-159.2 MB. The extra FP32 Q8 scale cache explains approximately
31.8 MB beyond the 127.1 MB weight increase.

The measured tradeoff does not meet the existing 3% speed screen, and
held-out PPL does not improve. The calibrated Q4 standard is retained.
This experiment answers the requested tradeoff question; it does not reject
all possible selective-precision recipes or optimized Q8 implementations.
The combined Q8-gates-plus-head variant has not been measured.


Quality uses the same BF16 teacher, prompts, target tokens and scoring masks as
the published standard: 8,192 scored tokens in 16 WikiText-2 article windows and
a separate six-document, 1,024-token screening suite. These are bounded subsets,
not full-corpus WikiText perplexity or a broad capability certification. Lower
PPL and KL are better. `results.json` includes paired document-bootstrap
intervals (2,000 resamples, seed 1234) against the standard.

Speed runs use `scripts/benchmark-inference-seq.ps1`, sequentially, with one
warmup and three measured repetitions per case, on a Ryzen 9 9955HX3D. Both
candidates use eight threads, affinity 21845 (`0x5555`), FP16 KV, identical
forced output tokens and full-vocabulary logits. No quality evaluation or
other heavy task runs during timing. Load, tokenization and HTTP are excluded;
there is no prefix-cache credit. Batch throughput is compared only at the same
batch size. The 4,096-input/two-output case measures prefill; its one decode
forward is excluded from sustained decode conclusions. Measurements do not
cover the Q4-specific greedy-head shortcut; Q8 currently falls back to full
logits for greedy generation.

## Relation to Q4_K_M

The shared idea is to allocate more precision to selected matrices, but this
is not the same recipe as Q4_K_M. The actual pinned Unsloth Q4_K_M reference
contains 98 Q4_K, 36 Q5_K, 17 Q6_K, 36 Q8_0 and 133 F32 tensors. Its tied
`token_embd.weight` is **Q6_K**, with a 208,588,800-byte payload. There is no
separate output matrix. Its total tensor payload is 521,555,200 bytes.

Our candidate uses Q4 and Q8 for quantized matrices, plus the unchanged small
F32 tensors. Its payload is 30,519,296 bytes larger than this Q4_K_M reference.
The quality evaluation also computes PPL and KL for that pinned reference on
identical positions. Comparing precision labels alone would miss the different
H128 transform, fitting method, group layouts and selected tensor precisions.

## Validation and reproduction

The candidate passes free greedy German `2 + 2` generation (`4`), the separate
prefix/answer logit test, and real-model scheduler/paged-state and prefix-cache
checks. The unchanged inference implementation previously passed all 15 local
tests and Windows/Linux CI at commit `134092e`.

```powershell
benchmarks/q8-tools/qwen35_cpu_pack.exe --hf-model-dir models/qwen3.5-0.8b --output models/qwen3.5-0.8b/model-calibrated-q8-head.q35h --quantizer mse16 --importance-dir benchmarks/plan-calibration-fit --q8-head
python scripts/audit-q8-artifact.py models/qwen3.5-0.8b/model-calibrated-q8-head.q35h --out benchmarks/q8-head-audit.json
python scripts/prepare-q8-benchmarks.py --candidate models/qwen3.5-0.8b/model-calibrated-q8-head.q35h --label q8-head --out benchmarks/q8-head-speed-inputs
pwsh -File scripts/run-q8-head-experiment.ps1
```

Freeze the built tools under `benchmarks/q8-tools` before running. Use fresh
output directories for repetitions. The existing calibration and reference-logit
caches are described in the [original implementation report](implementation-plan-results-2026-09-06.md).

Candidate SHA-256:
`5417ecc7e681b0c5bca5b6868bf30e056c221ba2dfa421c9f84693fad2fb1b4a`.

[Results and uncertainty](results/q8-experiments-2026-09-06/q8-head/results.json) Â·
[Speed and resident memory](results/q8-experiments-2026-09-06/q8-head/performance.csv) Â·
[Raw evidence](results/q8-experiments-2026-09-06/q8-head/raw-results.zip) Â·
[Checksums](results/q8-experiments-2026-09-06/q8-head/SHA256SUMS).

Experimental weights remain local. The published calibrated Q4 default and
the README's llama.cpp/Unsloth comparison continue to describe that standard.
