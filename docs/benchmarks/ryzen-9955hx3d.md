# Qwen3.5-0.8B on the Ryzen 9 9955HX3D

On November 19, 2026, all five configurations ran on the same laptop in one
sequential series. The native engine used the published 0.8B H128/Q4-G32-DOT4
checkpoint. llama.cpp Q4_0 and ik_llama.cpp IQ4_K_R4 match its tensor size.
The smaller IK IQ4_KS_R4 and larger Unsloth Q4_K_M show the range on either
side of that size. The [README](../../README.md#qwen35-08b-performance)
summarizes the results. The [CSV](../results/readme-2026-11-19/performance.csv)
includes the median and range of each case.

## What was timed

The Ryzen 9 9955HX3D ran on AC power with the max-power platform profile,
Arch Linux kernel 7.3.4 and eight threads pinned to physical cores 0–7 on
the 96 MiB L3 CCD. The native Release build used GCC 16.2.1 and AVX-512/VNNI.
Each case ran once to warm up and three times for measurement. Case order
reversed each round. Small differences should be read alongside the
[individual samples](../results/readme-2026-11-19/samples.csv).

| Workload | Measurement |
| --- | --- |
| Single request, 512 input and 128 output tokens | Prompt processing and 127 decode forwards. The first output token comes from prefill. |
| Single request, 4,096 input and 2 output tokens | Prompt processing over the longer input. |
| 16 private requests, each with 512 input and 128 output tokens | Aggregate decode throughput across all 2,032 decode forwards. |

The engines used fixed token IDs, FP16 KV and full-vocabulary logits. Model
loading, tokenization, HTTP and prefix reuse are outside the timed interval.
Native prefill uses the model-forward timer. The comparator timer covers
synchronized `llama_decode`. These workloads exercise ordinary target-model
decode with fixed continuations. For prompts that can benefit from drafting
or history reuse, see the separate [4B agent study](qwen35-4b-pi-agent.md).

## Quantization quality

The native CLI rescored all 16 WikiText-2 test windows with 256 prompt tokens
and 512 scored tokens per window. Each full-vocabulary logit capture matched
the SHA-256 of the corresponding October native capture, including the target
IDs. Computing perplexity as the exponential of the mean target-token negative
log-likelihood gave 15.79649. The identical distributions also preserve the
original 0.060100 mean KL from the frozen BF16 teacher to the native model.
The comparator scores in the README come from the October run on the same
checkpoints and comparator builds.

To repeat the native check, write each window's `prompt_tokens` and
`target_tokens` as comma-separated IDs from the [input fixture](../results/cpu-refresh-2026-10-22/quality-inputs.json).
Run the native CLI with the arguments in the [verification record](../results/readme-2026-11-19/native-quality.json)
and compare each logit dump's SHA-256 with the recorded window hash.

## Same-day native control

A second series ran the archived October README executable and the November
build in alternating order on this laptop. Both used the same H128 artifact,
affinity, token files and full-logit settings as the five-way comparison.
Values are medians in tokens/s from three runs after one warmup per case.
[Per-run rates and executable hashes](../results/readme-2026-11-19/native-control.json)

| Workload | October build today | November build today | Change |
| --- | ---: | ---: | ---: |
| Decode B=1 | 109.9 | 108.4 | -1.4% |
| Decode B=16 (aggregate) | 573.4 | 576.4 | +0.5% |
| Prefill 512 | 3,052.3 | 3,026.3 | -0.9% |
| Prefill 4,096 | 2,674.6 | 2,661.7 | -0.5% |

The archived executable itself reached 109.9 tokens/s at B=1 today, compared
with 123.0 in the [October run](native-prefill.md#final-paired-measurement).
The two builds measured today are within 1.4% across these workloads. The
cross-date difference is substantially larger. Its cause was not isolated.

## Reproduction

- Native source was [`8b0d642`](https://github.com/Danmoreng/qwen35-cpu/commit/8b0d6426aa4bf3a2fbc4f648d29eeb0b4b2fadb6).
- The retained comparator builds used [llama.cpp b11815](https://github.com/ggml-org/llama.cpp/commit/7170cad4d02f7907517613931840cbb2367399a3) and [ik_llama.cpp `98aca86`](https://github.com/ikawrakow/ik_llama.cpp/commit/98aca8692ae334e6e23e4d39ac43ce0d21029c35). IK repacked weights at load. Only its batch-16 executable used the [graph-capacity patch](ik-batch16.md#reproduction).
- The [manifest](../results/readme-2026-11-19/manifest.json) records the host settings, executable and library hashes, input hashes and raw profile location. Model hashes and comparator executables match the [October series](cpu.md).

Build the native benchmark at the recorded revision. Build the comparator
targets using the [pinned build instructions](cpu.md#reproduction-and-evidence).
Run from the repository root with the models and binaries at the paths in
the [matrix](../results/readme-2026-11-19/matrix.json), or adapt those paths
to the local installation.

```sh
cmake -S . -B build-readme-refresh -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWEN35_BUILD_SERVER=OFF
cmake --build build-readme-refresh --target qwen35_cpu_bench qwen35_cpu
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 \
  -Matrix docs/results/readme-2026-11-19/matrix.json \
  -OutputDir benchmarks/ryzen-repeat -Affinity 255 -WarmupRuns 1 -Runs 3
```
