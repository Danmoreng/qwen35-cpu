# CPU quantization comparison — September 6, 2026

This comparison measures fixed-token native engine calls on an AMD Ryzen 9
9955HX3D. It is not an HTTP load test or a claim about every x86 processor.
Quality and speed are separate measurements; a four-bit label does not establish
equal accuracy or equal tensor storage.

## Quality results

Lower perplexity and KL are better. The BF16 teacher's perplexity is **14.3856**.

| Candidate | Perplexity | Mean KL, nats | p95 KL, nats | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| H128/Q4-G32-DOT4 | 18.5833 | 0.17484 | 0.49173 | 78.44% |
| Q4_0 `--pure` | 18.0259 | 0.14452 | 0.37970 | 80.29% |
| Unsloth Q4_0 | 15.5809 | 0.06839 | 0.19024 | 86.73% |
| Unsloth Q4_K_M | 14.7529 | 0.03469 | 0.09855 | 90.39% |
| Unsloth IQ4_XS | 15.1614 | 0.05054 | 0.14247 | 88.15% |

**This evaluation does not show a quality advantage for H128.** Even against
the equal-payload pure Q4_0 control, H128 has approximately 3.1% higher perplexity
and 21.0% higher mean KL. The larger downloaded recipes perform better still.
This is a result for the tested English-prose subset and implementations, not
a universal ranking of quantization methods.

[Aggregate quality CSV](results/2026-09-06/quality-summary.csv) ·
[Per-window means](results/2026-09-06/quality-windows.csv).

## Frozen inputs

- Engine runtime: `0992f9864d2faf47f58a7d88d4b533688067e093` (v0.1.1).
  The comparison adds measurement fields and tools, without changing inference kernels.
- llama.cpp: `73a43d1f69345aee8bb186ef4b3172cef892f2e5`, freshly fetched for this run.
- H128 model: [public artifact](https://huggingface.co/danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4/tree/cc7df08da7ef7ac15db62e80b4eda85e19a143da),
  revision `cc7df08da7ef7ac15db62e80b4eda85e19a143da`.
- BF16, Q4_0, Q4_K_M and IQ4_XS: [Unsloth GGUFs](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0),
  revision `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
- Q4_0 `--pure`: generated locally by the pinned llama quantizer from that BF16 GGUF,
  with `llama-quantize --pure INPUT OUTPUT Q4_0 12` and no importance matrix.

All 187 BF16 tensor payloads match the parent project's conversion of the source
used for H128. Of all 320 tensors, 313 match exactly; seven derived FP32 `ssm_a`
tensors differ by at most 5.97e-8. The comparison therefore includes a tiny
conversion-rounding difference as well as quantization and engine arithmetic.
The original source revision was not recorded; its safetensors SHA-256 is retained
in the H128 model's `quantization.json` and the result metadata.

| Artifact | Tensor payload, decimal MB | Bits/parameter | File, decimal MB |
| --- | ---: | ---: | ---: |
| H128/Q4-G32-DOT4 | 424.935 | 4.518 | 424.965 |
| Q4_0 `--pure` | 424.935 | 4.518 | 435.896 |
| Unsloth Q4_0 | 496.193 | 5.276 | 507.155 |
| Unsloth Q4_K_M | 521.555 | 5.546 | 532.517 |
| Unsloth IQ4_XS | 481.644 | 5.121 | 492.606 |
| BF16 teacher | 1,505.783 | 16.011 | 1,516.745 |

The GGUF files embed tokenizer metadata; `.q35h` does not. The smaller H128 file
relative to pure Q4_0 is therefore **not a reduction in tensor storage**.
All variants represent 752,393,024 text parameters, with tied output embeddings
and no vision/MTP tensors. The published Unsloth recipes retain Q5/Q6/Q8 tensors;
their full tensor-type counts are recorded in `inputs.json` inside the raw archive.
Their metadata identifies `unsloth_calibration_Qwen3.5-0.8B.txt`, 80 chunks and 186
importance entries. Its contents and disjointness from the test set were not
verified. These are downloaded public recipes, not independently reproduced
calibrations with a guaranteed disjoint calibration corpus.

## Quality protocol

Dataset: [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3),
`wikitext-2-raw-v1` test split, pinned revision
`b08601e04326c79dfdd32d625aee71d232d685c3`.
This is **a deterministic English-prose subset, not full WikiText perplexity**.

The preparation script identifies top-level article boundaries, keeps articles
with at least 768 tokens and selects 16 evenly spaced eligible articles. Each
window starts with 256 context tokens and scores the following 512 targets:
**8,192 scored tokens**, with fresh state at every article. No prompt is selected
based on a candidate's scores. Exact windows, hashes and token IDs are retained.

The original HF tokenizer supplies all IDs, with no BOS insertion or chat
template. One native-tokenizer mismatch was found on a multiple-whitespace span
in the selected Zrinski Battalion article. Feeding identical external IDs avoids
mixing tokenizer segmentation differences into this quantization comparison.
The mismatch remains a separate tokenizer limitation; these results do not
validate end-to-end text tokenization equivalence.

Every engine teacher-forces identical target tokens, without sampling, penalties
or EOS truncation. The teacher is llama.cpp BF16, with FP16 K/V and FP32 recurrent
state, as used for candidates. Quality runs use eight threads and a 1024-token
context limit. Raw full-vocabulary logits cover all 248,320 tokens at each scored
position. Logit capture is excluded from every speed run.

The scorer computes stable float64 log-softmax, target negative log likelihood
and `KL(p_BF16 || p_candidate)` in nats. Perplexity is `exp(mean NLL)` across all
8,192 positions, **not an average of window perplexities**. Mean/p95 KL and top-1
agreement use the same positions. This measures the combined checkpoint and
inference arithmetic difference from the common teacher, not quantization error
isolated from implementation differences.

## Speed protocol

Windows, Ryzen 9 9955HX3D (16 cores / 32 logical processors), approximately 64 GiB
installed RAM, Balanced power scheme. MSVC 19.44 Release builds. llama.cpp uses
CPU-only native AVX-512/VNNI, OpenMP, flash attention, `n_batch=2048` and
`n_ubatch=512`. The wrapper explicitly probes/enables VNNI on MSVC because upstream
native detection otherwise omitted it on this host. No upstream kernels were
modified. Our portable runtime selects AVX-512-VNNI and default 128-token prefill
chunks. Both use FP16 K/V and full vocabulary-head computation.

All timed runs use `scripts/benchmark-inference-seq.ps1`, one process at a time,
one discarded warmup and three measured runs. Case order reverses on alternating
passes. Tables use the median; CSV also records minimum and maximum. Model load,
initial context allocation, tokenization, logits export and file I/O are outside
the compared forward timings. Our raw profile additionally retains prefill wall
time including request allocation; the comparison uses its separately recorded
`prefill_forward_tokens_per_second`, matching llama's context allocation boundary.

- Single request: equal 8 and 12 threads, mask `0xffff` on the V-Cache CCD.
  P=512/1024/2048/4096 with N=2 for prefill; P=512 with N=128/256/512/1024 for decode.
- Static independent batches: B=4/16, P=512, N=128; both engines at 8 threads with
  mask `0xffff`, and at 16 threads with mask `0x55555555` (one logical processor per
  physical core across both CCDs). These are matched configurations, not an
  exhaustive thread/affinity autotuning sweep.
- Every request prefills separately, then all active requests decode together.
  Each sequence has private state and a distinct final prompt token. There is
  no prefix cache, shared-prefix credit, continuous admission or HTTP overhead.
- Prompt IDs come from the checked-in speed fixture, repeated cyclically for
  longer lengths; output IDs continue from the same cyclic fixture. These inputs
  are suitable for fixed-work speed comparison, not representative perplexity.
- Prefill includes the final full vocabulary head. The first output is predicted
  during prefill, so N output tokens require **N-1 decode forwards per request**.
  Aggregate decode throughput is `B*(N-1)/decode_seconds`; it is never divided by
  an unrelated single-request llama.cpp baseline to advertise batched speedup.

The adapter's batched final logits are checked against 16 separate single-sequence
runs over a short forced continuation, including every vocabulary entry. Small
floating-point differences between matrix shapes are allowed; state routing must
agree. This is a harness sanity check, not an exhaustive llama.cpp validation.
On this build the B=4 and B=16 checks were bit-identical to their independent
single-sequence references (maximum absolute difference and RMSE both zero).

## Reproduction

From the repository root, build the engine and pin/build llama.cpp:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git .cache/llama.cpp
git -C .cache/llama.cpp checkout 73a43d1f69345aee8bb186ef4b3172cef892f2e5
./scripts/build.ps1
./scripts/build-llama-comparison.ps1
```

Download the pinned H128 package using the README's download command into
`models/hf-download-test`; its tokenizer JSON and tokenizer configuration are
byte-identical to the original HF files used for this evaluation. Python preparation/scoring additionally needs
`huggingface_hub`, `transformers`, `pyarrow` and `numpy`. The release server does not.

```powershell
python scripts/prepare-comparison-data.py
./build-llama/bin/llama-quantize.exe --pure models/llama-comparison/Qwen3.5-0.8B-BF16.gguf models/llama-comparison/Qwen3.5-0.8B-Q4_0-pure.gguf Q4_0 12
python scripts/record-comparison-inputs.py
python scripts/evaluate-comparison-quality.py
python scripts/validate-llama-batching.py
python scripts/prepare-comparison-matrix.py
./scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/comparison-2026-09-06/single-ccd-matrix.json -OutputDir benchmarks/comparison-2026-09-06/speed-single-ccd -Affinity 65535
./scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/comparison-2026-09-06/batch-ccd-matrix.json -OutputDir benchmarks/comparison-2026-09-06/speed-batch-ccd -Affinity 65535
./scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/comparison-2026-09-06/batch-physical-matrix.json -OutputDir benchmarks/comparison-2026-09-06/speed-batch-physical -Affinity 1431655765
python scripts/summarize-comparison.py
```

Use affinity masks appropriate to the target CPU. Record its machine configuration
as `machine.json` next to the matrices before summarizing. Each speed output
directory must be new. The runner records source revision, executable hashes,
commands and affinity; `record-comparison-inputs.py` additionally hashes the model
files, tokenizer, speed fixture and llama DLLs. Raw logits are deleted after
per-position metrics are retained, keeping temporary storage bounded.

WikiText data derives from Wikipedia and is attributed to the WikiText dataset
authors and Wikipedia contributors. The pinned dataset card lists **CC BY-SA 3.0
and GFDL**; retained token-derived evaluation data remains subject to those data
licenses, not the engine's MIT license. Full source articles and model weights
are not included in the results archive.
