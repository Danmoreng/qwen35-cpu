# Qwen3.5-2B on the i7-8750H and GTX 1070 Max-Q

## Offline teacher collection

The CPU and CUDA builds of the same resident collector used the same 2B BF16
text GGUF, six physical CPU cores (mask 63), and the same two 1,024-token
documents. Each process loaded the teacher once. The sequential benchmark
alternated case order across one warmup and three measured runs. CUDA 12.4
targeted SM 6.1 and cuBLAS used FP32 arithmetic on the BF16 source weights.

| Collector | Capture time for both documents, median | Entire process, median |
|---|---:|---:|
| i7-8750H | 30.25 s | 30.94 s |
| GTX 1070 Max-Q | 4.12 s | 5.50 s |

The 7.34× ratio covers teacher evaluation and writing raw samples to local disk.
Device-wide VRAM peaked at 6,171 MiB with a 3 MiB idle reading (200 ms sampling).
Both collectors produced 187 named projection inputs per document with matching
shapes and sample counts. The lowest per-projection cosine was 0.9999798.
Final-token greedy choices agreed on both documents. CPU-to-GPU KL divergence
was 0.00004972 on the German document and 0.00000875 on the English document.

CPU statistics fitting and final Q4 packing run separately. See the [per-run
profiles, input hashes, commands and parity results](../results/qwen35-2b-i7/teacher-pilot.json)
and the [conversion guide](../quantization.md#qwen35-2b).

## Native inference

The 2B artifact contains 230 text tensors and occupies 1,060,442,880 bytes.
The original 0.8B artifact and the new 2B artifact were measured on six physical
cores with AVX2, a single request, the same 512-token prompt, 64 forced output
tokens and full logits. The table reports medians from three measured runs
after one warmup. Prefill measures the forward pass and peak RSS covers the
inference process.

| Model, H128/Q4-G32-DOT4 | Prefill tok/s | Decode tok/s | Peak RSS MiB |
|---|---:|---:|---:|
| Qwen3.5-0.8B | 463.94 | 63.34 | 468.3 |
| Qwen3.5-2B | 206.96 | 28.06 | 1081.6 |

## Quantization quality

The 2B-specific fit used 256 documents and 262,144 tokens across five domains.
The resident CUDA teacher collected all documents in a single process. A local
CPU fitting pass shared the FP64 H128 transform between the diagonal and block
statistics. On one complete pilot document, all 187 diagonal arrays and all
187 block-moment arrays matched the reference implementation bit for bit.
The standard weighted MSE16 packer used block-128 error compensation with
damping 0.01. Collection and statistics preparation took 585.47 seconds, including periodic
verified saves. CPU per-tensor packing times summed to 11.94 minutes. The calibrated artifact SHA-256 is
`1084c3a2751d3813a2aa686337f717e56291e6a1a79ff8be1b6d1665d22892a3`.

The ten reserved documents contained two windows per domain. Each had a
512-token prompt and 128 scored positions. All candidates used the same forced
tokens and the same full-vocabulary CPU BF16 reference from llama.cpp.
Perplexity is the exponential of mean target NLL across all 1,280 positions.

| 2B model and scoring backend | Perplexity | Mean KL from BF16 |
|---|---:|---:|
| BF16, llama.cpp CPU | 8.69191 | 0 |
| H128/Q4-G32-DOT4 unweighted MSE16, native CPU | 9.41504 | 0.086713 |
| H128/Q4-G32-DOT4 calibrated MSE16, native CPU | 9.06662 | 0.048752 |

The 0.8B four-position regression retained identical logits. For both 2B
artifacts, batch sizes 4, 8, 12, 16, 20 and 32 reproduced the single-request
logits bit for bit. Native and Hugging Face tokenizer output also agreed on
the first 1,024 tokens of one document in each domain.
Per-window quality, corpus revisions, binary and checkpoint hashes, exact
benchmark arguments and individual speed runs are in the
[model results](../results/qwen35-2b-i7/model-results.json).
