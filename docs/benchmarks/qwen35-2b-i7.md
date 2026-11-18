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
