# Linux i7-8750H speed comparison

This is the Linux counterpart to the Windows Ryzen 9 9955HX3D speed table in
the README. It uses the same five checkpoint candidates, externally supplied
tokens, full-vocabulary logits and three published workloads. Quality was not
rerun: the README quality table remains a checkpoint-level result from the
common 8,192-token evaluation, not a CPU speed result.

## Speed contract

- Ubuntu 26.04.1 LTS, Linux 7.0.0-31-generic, GCC 15.2.0 Release builds.
- Intel Core i7-8750H, six physical cores / twelve logical processors.
- Six threads pinned to logical CPUs 0-5 (`0x3f`), one thread per physical core.
- Native engine forced to AVX2/FMA/F16C; llama.cpp built CPU-only with
  `GGML_NATIVE=ON`, OpenMP, flash attention, FP16 KV, `n_batch=2048` and
  `n_ubatch=512`.
- Maximum context 16,384, matching the Windows README comparison.
- One warmup and three measured runs through
  `scripts/benchmark-inference-seq.ps1`; case order reverses on alternating
  passes and no other performance workload runs concurrently.
- B1/P512/N128, B1/P4096/N2 and B16/P512/N128. N2 supplies only the long-prefill
  result. Decode counts N-1 actual forwards per request; B16 is aggregate
  throughput across sixteen private requests.

The native engine and all llama.cpp candidates were measured together in a new
series. Values in the README are medians; [performance.csv](results/readme-linux-i7-8750h-2026-09-07/performance.csv)
also retains min/max values. Raw profiles, commands, stdout/stderr and runner
metadata remain under the ignored local benchmark directory
`benchmarks/readme-linux-i7-8750h-20260907/`.

## Provenance

- Repository revision: `d46304c81fa543e0f3f12f41383618790cfbaf4d` with an
  empty tracked diff when the benchmark started.
- llama.cpp revision: `73a43d1f69345aee8bb186ef4b3172cef892f2e5`.
- H128 B+C 256 model revision:
  `f6cb5cf04a9094670f9041578a3f4e44b94a1395`.
- Unsloth GGUF revision:
  `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
- The pure Q4_0 control was generated locally from the pinned BF16 GGUF with
  the pinned `llama-quantize --pure ... Q4_0 12` command.

The compact [manifest](results/readme-linux-i7-8750h-2026-09-07/manifest.json)
records model, executable, shared-library, runner, matrix and raw-archive
hashes. Quantization labels are not quality equivalence claims, and the Windows
and Linux tables are separate complete series rather than cross-platform rows
spliced into one ranking.
