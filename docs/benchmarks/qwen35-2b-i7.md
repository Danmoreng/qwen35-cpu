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

## Matched ik_llama.cpp comparison on November 18, 2026

On the i7-8750H the native engine decoded 11.2% faster at 0.8B and
15.4% faster at 2B than ik_llama.cpp IQ4_K_R4. Each pair has identical
on-disk tensor payload. Complete file sizes include format metadata and the
GGUF tokenizer. MB is decimal.

| Qwen3.5 | Engine and quant | Tensor MB | File MB | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|
| 0.8B | Native H128/Q4-G32-DOT4 | 424.93 | 424.96 | 463.42 | 63.65 |
| 0.8B | ik_llama.cpp IQ4_K_R4 | 424.93 | 435.90 | 303.10 | 57.24 |
| 2B | Native H128/Q4-G32-DOT4 | 1,060.41 | 1,060.44 | 203.45 | 28.13 |
| 2B | ik_llama.cpp IQ4_K_R4 | 1,060.41 | 1,071.37 | 119.46 | 24.38 |

Both calibrated native checkpoints were remeasured alongside IQ4_K_R4 files
freshly quantized with the same IK build used for inference. The 0.8B source
was the archived Unsloth BF16 GGUF. The 2B source was the original Qwen
checkpoint, exported as a 24-layer text-only BF16 GGUF with the optional MTP
head excluded. IK used its pure IQ4_K_R4 recipe without an importance matrix
and enabled runtime tensor repacking. The CPU-only Release build used GCC
15.2.0 and host-native instruction selection. [Tested IK revision](https://github.com/ikawrakow/ik_llama.cpp/commit/093e74cb0590eaf462a04516ba1bab8c6b6637bc).

Each model size ran as its own two-engine matrix with alternating case order.
All rates are medians of three measured runs after one warmup. Each process
used six physical cores (mask 0x3f), one request, an 8,192-token context limit,
FP16 KV, the same 512 prompt tokens, and the same 64 forced output tokens.
Prefill includes the final vocabulary projection, using the native forward
timer and IK's synchronized decode call. Generation counts the remaining 63
decode forwards. Load time and load-time repacking are excluded.

The [run samples](../results/qwen35-2b-i7/ik-2026-11-18/samples.csv) and
[reproduction manifest](../results/qwen35-2b-i7/ik-2026-11-18/manifest.json)
record the timing boundaries, arguments, input and binary hashes, source
revisions and tensor inventories.

## BF16 fidelity of the size-matched quants

Each model size used one shared BF16 reference in llama.cpp on the CPU. The
test covered ten reserved documents across German, English, Python code, dialog
and math, two per domain. For each document, all three systems received the
same 512 prompt tokens and 128 forced continuations. Each score covers
1,280 positions. KL measures the mean divergence from the BF16 distribution
over all 248,320 vocabulary entries (nats per position, lower is closer).
Perplexity measures the probability assigned to the observed continuation.
Top-1 agreement counts positions where the quant and BF16 chose the same token.

The BF16 target perplexities were 12.1266 (0.8B) and 8.6919 (2B).

| Qwen3.5 | Checkpoint and scoring backend | Mean KL ↓ | Target perplexity ↓ | Top-1 agreement ↑ |
|---|---|---:|---:|---:|
| 2B | H128/Q4-G32-DOT4, native CPU | 0.048752 | 9.0666 | 88.91% |
| 2B | IQ4_K_R4, ik_llama.cpp CPU | 0.049760 | 9.1614 | 88.59% |
| 0.8B | H128/Q4-G32-DOT4, native CPU | 0.064336 | 12.7184 | 87.42% |
| 0.8B | IQ4_K_R4, ik_llama.cpp CPU | 0.077614 | 12.9446 | 85.55% |

| Qwen3.5 | H128 vs. IK change in mean KL | H128 wins by document (KL) | Paired KL difference, 95% interval | H128 vs. IK change in perplexity, 95% interval |
|---|---:|---:|---:|---:|
| 0.8B | -17.1% | 9/10 | -0.01328 [-0.01706, -0.00950] | -1.75% [-2.81%, -0.67%] |
| 2B | -2.0% | 7/10 | -0.00101 [-0.00524, +0.00323] | -1.03% [-2.48%, +0.43%] |

At 0.8B, both paired intervals favor the calibrated native checkpoint. At 2B,
its pooled lead is small and both intervals cross zero. IK was closer to BF16
on average on the two 2B dialog windows, while H128 had lower mean KL in each
of the other four domains.

The intervals resample matched documents within each domain, preserving the
five-domain mix. Two documents per domain and the fixed selection limit transfer
to other workloads. Scores include the arithmetic of each deployed CPU backend.
H128 used 256-document activation calibration and blockwise error compensation.
The measured IK quants were created without an external importance matrix.

As a calibration control, the unweighted 2B H128 artifact scored 0.086713
mean KL and 9.4150 perplexity on the same reference and targets.
This control reuses archived full-vocabulary native logits for the same windows.
All BF16, calibrated H128 and IQ4_K_R4 logits in the main comparison were freshly computed.

The [per-window measurements](../results/qwen35-2b-i7/ik-quality-2026-11-18/windows.csv)
and [quality manifest](../results/qwen35-2b-i7/ik-quality-2026-11-18/quality.json)
include domain scores, paired intervals, token hashes, source revisions and binary identities.
