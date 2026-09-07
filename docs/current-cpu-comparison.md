# Current CPU comparison

The README compares the current calibrated H128/Q4-G32-DOT4 engine with four mainline llama.cpp
GGUFs and six ik_llama.cpp GGUFs. The native build includes the AVX-512/VNNI
DOT4 prefill kernel. The main sequential series covers every B1/prefill candidate
and native/mainline B16. A separate sequential B16 supplement covers all six IK
quants alongside fresh native and mainline Q4_0 controls under the same conditions.
The README uses the fresh controls; other scores retain their main-series values.
No optimization-screen timings are used.

## Performance contract

- Ryzen 9 9955HX3D, Arch Linux / kernel 7.2.2, GCC 16.2.1 Release.
- Eight threads pinned to physical V-Cache cores 0–7 (`0xff`), performance
  governor/EPP and max-power platform profile.
- Codex minimized and XFCE compositing disabled for the entire series, with the
  display unchanged at 2560×1600 / 240 Hz. The local wrapper restores the desktop.
- One warmup and three measured runs, alternating case order, through
  `scripts/benchmark-inference-seq.ps1`. No concurrent inference or compilation.
- Identical fixed tokens, full-vocabulary logits, FP16 KV, maximum context 16,384.
- B1/P512/N128 and B1/P4096/N2 for all eleven candidates; B16/P512/N128 for
  all candidates across the main series and B16 supplement. B16 counts sixteen private
  requests, with 127 decode forwards each. Long-prefill N2 decode is not used as
  the README decode score.
- CPU-only llama builds, flash attention, `n_batch=2048`, `n_ubatch=512`, no
  GPU/KQV offload. IK has runtime repacking enabled. Native prefill uses its
  default 128-token chunks; short batches retain the decode-oriented kernel.
- Loading, repacking, tokenization, HTTP, quality-logit capture and prefix-cache
  credit are excluded from speed measurements.

Medians and min/max are in the
[main-series CSV](results/current-cpu-2026-09-07/performance.csv) and
[B16 supplement](results/ik-batch16-2026-09-07/performance.csv). IK B16 uses a
local graph-capacity correction to the pinned fork; stock upstream aborts.
See the [patch, validation and reproduction](ik-batch16.md), including the
numerical differences between B8 and B16. IK B1/prefill and quality use stock IK.

## Quality and size

The README uses the common 16-window WikiText-2 English test subset, 8,192 scored
tokens, matching prompts/masks and the same BF16 teacher. It is not a full-corpus
or broad application-quality score. Lower perplexity and KL divergence are better.

The [native/mainline quality CSV](results/readme-bc256-2026-09-07/quality.csv)
and [IK quality CSV](results/ik-linux-2026-09-07/quality.csv) identify the scoring
backend. IK and mainline use identical files for the four shared GGUF formats;
backend outputs need not be bitwise identical. The three Unsloth quants have
mainline quality measurements, not separately measured IK scores.

The two IK-specific pure quants are generated directly from the archived BF16
model without an importance matrix or evaluation-set calibration. IQ4_K_R4 has
424,934,656 bytes of on-disk tensors; IQ4_KS_R4 has 401,437,952 bytes. Their full
GGUF files are 435,896,352 and 414,992,672 bytes. IK reports 567,966,976 and
540,110,336 bytes of loaded model tensors respectively, excluding KV and compute
buffers. Storage size is not total resident memory.

The native optimization preserves the checkpoint and arithmetic results. All 16
tests pass; 95,354,880 full-vocabulary logits match the baseline byte for byte
over 24 prompts / 384 output positions, including chunk boundaries and P4096.
See [implementation validation](prefill-avx512-2026-09-07.md).

## Reproduction and evidence

Mainline llama.cpp is pinned to `73a43d1f69345aee8bb186ef4b3172cef892f2e5`;
IK to `fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf`. Both source trees retain
their unmodified upstream implementation for the main series. IK B16 uses the
separate patched build described above. H128 uses revision
`f6cb5cf04a9094670f9041578a3f4e44b94a1395`; Unsloth uses
`6ab461498e2023f6e3c1baea90a8f0fe38ab64d0`.

The [main-series manifest](results/current-cpu-2026-09-07/manifest.json) and
[B16 manifest](results/ik-batch16-2026-09-07/manifest.json) preserve commands,
binary/library/checkpoint hashes, build configuration, source identities, CPU
affinity and desktop snapshots. Raw profiles and the automatic minimization
wrapper remain under ignored
`benchmarks/prefill-ik-investigation-20260907/current/`. Models, build outputs
and raw ZIP archives are not committed. The wrapper contains machine-specific
PowerShell and X11 paths; inspect it before reuse and select a fresh output folder.

Build the native engine with `cmake -S . -B build-prefill -G Ninja
-DCMAKE_BUILD_TYPE=Release` and `cmake --build build-prefill -j 8`.
The [IK report](ik-linux-comparison-2026-09-07.md) contains the comparator build
and quantization commands. Platform studies and optimization history are kept
in the detailed reports, separate from the README product summary.
