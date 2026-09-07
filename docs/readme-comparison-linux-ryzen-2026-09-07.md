# Linux/Ryzen: H128 B+C 256 versus llama.cpp

All five candidates are measured together in a fresh sequential Linux series:
the native H128/Q4-G32-DOT4 B+C 256 checkpoint, llama.cpp pure Q4_0, and Unsloth
Q4_0, Q4_K_M and IQ4_XS. The README uses this complete series, rather than
combining the earlier native-only desktop experiment with new comparator rows.
Quality was not rerun; the existing checkpoint-level quality results remain
separate from this speed comparison.

## Speed contract

- AMD Ryzen 9 9955HX3D, Arch Linux, kernel 7.2.2-arch1-1, GCC 16.2.1 Release.
- Eight threads on physical V-Cache cores 0–7 (`0xff`), maximum-performance
  power policy. A live llama.cpp affinity snapshot confirms all eight threads
  are constrained to that set.
- Codex automatically minimized for the entire series; XFCE compositing disabled;
  internal display unchanged at 2560×1600 / 240 Hz. The wrapper restores the
  original compositor and active-window state after completion or runner failure.
- Native AVX-512/VNNI. llama.cpp CPU-only, `GGML_NATIVE=ON`, OpenMP,
  AVX-512/VNNI verified in runtime logs, flash attention enabled, FP16 KV,
  `n_batch=2048`, `n_ubatch=512`, no GPU or KQV offload.
- Maximum context 16,384 per request; full-vocabulary logits and identical
  externally supplied tokens for all candidates.
- B1/P512/N128, B1/P4096/N2 and B16/P512/N128. The N2 workload supplies only
  long-prefill throughput. Decode counts 127 forwards per request; B16 is
  aggregate throughput across sixteen private requests.
- One warmup and three measured runs through `scripts/benchmark-inference-seq.ps1`,
  sequentially, with reversed case order on alternate passes. No other inference
  benchmark or compilation runs concurrently.
- Prefill includes the final vocabulary head. Loading, tokenization, HTTP,
  desktop changes and prefix-cache credit are excluded from the reported timings.

The [performance CSV](results/readme-linux-ryzen-2026-09-07/performance.csv)
contains medians and min/max for every workload. The README reports medians;
three runs describe this machine and workload, not a universal guarantee.
llama.cpp prefill varies more than decode in this series: for example, pure
Q4_0 P512 spans 1,102.19–1,349.45 tok/s, while its B1 decode spans
108.19–109.28 tok/s. Prefill ratios summarize the medians, not a tight bound.

## Revisions and evidence

- llama.cpp: `73a43d1f69345aee8bb186ef4b3172cef892f2e5`, matching the Windows
  comparison. Linux sources were extracted directly from that Git commit,
  without copying changes from the Windows working directory.
- Native engine binary is unchanged from the Linux investigation; no performance
  patch was required.
- H128 checkpoint revision: `f6cb5cf04a9094670f9041578a3f4e44b94a1395`.
- Unsloth revision: `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.
- All four GGUF files and prompt fixtures match the archived Windows input hashes.
  The pure Q4_0 control is the exact archived pure-quantized file.

The [manifest](results/readme-linux-ryzen-2026-09-07/manifest.json) records build
commands, revisions, binary/shared-library and checkpoint hashes, and the local
raw-archive hash. Raw profiles, commands, compile logs, matrices, desktop state,
affinity snapshot and the automatic minimization wrapper remain under ignored
`benchmarks/linux-ryzen-comparison-20260907/`. Weights, builds and ZIP archives
are not committed.

The local `run-minimized.py` wraps the existing sequential runner. It discovers
the Codex X11 window, records the initial state, minimizes it and disables the
compositor before starting measurements, then restores the state in a `finally`
block. Tool paths and the output directory are local to this experiment; verify
them before reuse. The runner refuses to overwrite an existing result directory.

See the [desktop investigation](linux-desktop-performance-root-cause-2026-09-07.md)
for why consistent rendering conditions matter on this laptop. Quantization
labels are not quality equivalence claims; compare the README's quality and
payload results separately from these CPU speeds.
