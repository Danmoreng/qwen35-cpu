# Native Qwen3.5-4B MTP on Ryzen 5 7600

The official Qwen3.5-4B snapshot provides 15 BF16 tensors for a trained
MTP block. The native experiment packs the block into an optional sidecar.
All timing cases use the same native 4B H128/Q4-G32-DOT4 target, including
the tied token embeddings and full-vocabulary output projection. The MTP
matrix formats are BF16, identity-basis Q8_0 and H128/Q4-G32-DOT4 with
unweighted MSE16. The packer stores normalization weights in F32.
[Pinned Qwen tensor inventory](https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/model.safetensors.index.json).

| MTP matrix format | Sidecar bytes | Sidecar MiB |
| --- | ---: | ---: |
| BF16 | 241,227,456 | 230.1 |
| Q8_0 | 128,177,856 | 122.2 |
| H128/Q4-G32-DOT4 | 67,884,736 | 64.7 |

BF16 uses BF16-stored matrix weights and F32 activations and accumulation.
The BF16 CPU kernel expands weights in registers and uses AVX2/FMA on this
system. Q8_0 and H128 use the engine's existing integer activation and
projection paths.

## Forward check

A PyTorch calculation used the official BF16 tensors with F32 arithmetic,
matched supplied embedding and hidden-state rows, and rounded K and V to
FP16. It reproduced the trained block's shift and concatenation, RoPE, gated
attention, SwiGLU and output RMS normalization using the upstream graph as a
reference. [Pinned llama.cpp Qwen3.5 implementation](https://github.com/ggml-org/llama.cpp/blob/73a43d1f69345aee8bb186ef4b3172cef892f2e5/src/models/qwen35.cpp#L485).

| Variant | Hidden-state RMSE from BF16 reference | Largest absolute difference | Lowest cosine over 7 positions |
| --- | ---: | ---: | ---: |
| BF16 | 0.0000283 | 0.000188 | 0.99999988 |
| Q8_0 | 0.05885 | 0.27370 | 0.99978018 |
| H128/Q4-G32-DOT4 | 0.50203 | 2.93628 | 0.98451871 |

The final probe row matched bit for bit between sequential and batched
MTP KV preparation for each precision. The probe uses seven fixed synthetic
inputs to check the block; acceptance and output parity use ordinary model
generation in the benchmarks.

## Resident request latency

Times are medians in milliseconds. Each MTP cell uses the fastest measured
depth for that precision and task. The authored email and transcript favored
two candidates. Prose and the 512-token continuation favored one.

| Workload | Ordinary decode | N-gram | Input lookup | MTP BF16 | MTP Q8_0 | MTP H128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Free prose, greedy | 5,832 | 5,834 | 5,862 | 5,971 | 5,827 | 5,709 |
| Free prose, sampled | 6,173 | 6,160 | 6,171 | 6,564 | 6,470 | 6,300 |
| English email edit, greedy | 3,412 | 1,739 | 1,506 | 2,396 | 2,328 | 2,277 |
| German transcript, greedy | 3,403 | 1,789 | 1,777 | 2,492 | 2,440 | 2,329 |
| 512-token English continuation, greedy | 3,507 | 3,318 | 3,287 | 3,504 | 3,542 | 3,519 |

H128 had the lowest MTP latency in four of the five cases. On greedy free
prose, every precision accepted 44 of 82 proposals at depth one. Median
proposal generation took 10.46 ms/token for BF16, 8.76 ms/token for Q8_0
and 7.40 ms/token for H128. H128 cut total latency by 2.1% versus ordinary
decode and moved from 22.46 to 22.99 decode tok/s. With seeded sampling,
all three accepted 47 of 79 at depth one and H128 took 2.1% more total time
than ordinary decode. With a 512-token prompt, BF16 and Q8_0 accepted 15
of 24 proposals and H128 accepted 14 of 25.

For the email and transcript at depth two, all three precisions accepted
45 of 46 proposals. H128 saved 33.2% and 31.6% versus ordinary decode.
Input lookup had the lowest latency on both. On free greedy prose, H128
at depths two and three took 6,395 and 6,624 ms, respectively. The optional
MTP mode defaults to one candidate based on these free-generation results.
Every timing case and run returned the exact ordinary-decode output tokens
for its workload and sampling settings.

## Method

The CPU engine and benchmark used GCC 16.2.1 Release with six physical
Ryzen 5 7600 cores pinned to CPUs 0–5. The target used native AVX-512/VNNI
dispatch, FP16 KV and one request. The sequential harness alternated case
order with one warmup and three measured runs. Resident latency sums prefill
and decode time inside the inference process and includes MTP KV preparation,
proposal generation, verifier work and cache repair. Load time and HTTP
transport are outside the resident timer.

Each workload uses an EOS stop and a 128-token output cap. Greedy decoding
uses temperature zero. The sampled prose case uses temperature 0.65,
top-k 40, top-p 0.9, repetition penalty 1.05 and seed 83. All modes use the
same target sampler and each MTP proposal uses an argmax.

Generate the authored input tokens with `scripts/prepare-draft-benchmark.py`.
The pinned 512-token prompt comes from the [CPU comparison fixtures](../results/cpu-refresh-2026-10-22/prompt-512.csv).
[Manifest](../results/qwen35-4b-mtp-2026-11-19/manifest.json) and
[per-run samples](../results/qwen35-4b-mtp-2026-11-19/samples.csv)
contain the command matrices, hashes, timings and token parity.
Save the manifest's `.matrix` as a
standalone file and pass it to `scripts/benchmark-inference-seq.ps1` with
`-Runs 3 -WarmupRuns 1 -Affinity 63`.
