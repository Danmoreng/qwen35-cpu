# Current calibrated H128 versus llama.cpp

This comparison presents only the published calibrated MSE16 H128 artifact as
the native engine candidate. The other candidates are llama.cpp pure Q4_0 and
the Unsloth Q4_0, Q4_K_M and IQ4_XS mixed-precision files. Quantization labels
are not treated as equal storage budgets or equal quality.

## Speed measurement contract

The fresh series runs all five candidates together, sequentially, using
`scripts/benchmark-inference-seq.ps1`: one warmup and three measured runs per
case, alternating forward and reverse case order. No model-quality evaluation
or other inference workload runs alongside the speed measurements.

- AMD Ryzen 9 9955HX3D, Windows 11, eight threads and affinity `0x5555` for every
  candidate and batch size. This selects one logical processor per physical
  core on the VCache CCD.
- FP16 KV, maximum context 16,384, identical externally supplied prompt and
  output token IDs, full-vocabulary logits and no sampling or greedy-head shortcut.
- Single-request workloads: P512/N128 and P4096/N2. Batch workload: B16/P512/N128.
- Prefill is forward throughput including the final full vocabulary head.
  Load, tokenization and HTTP are excluded. Decode counts N−1 actual forwards
  per request. The N2 case has only one decode forward and is used for its
  prefill measurement, not a decode-speed claim.
- Batch 16 numbers are aggregate throughput across 16 requests with private
  state. No prefix-cache credit is applied. Batch speedups compare batch 16
  with batch 16, never with a single-request baseline.

The executable and llama.cpp DLLs are frozen before the series. Checkpoint,
executable, DLL and input hashes, exact commands, source hashes and raw profiles
are preserved. The llama.cpp revision is
`73a43d1f69345aee8bb186ef4b3172cef892f2e5`, CPU-only MSVC with AVX-512/VNNI.
The native artifact is downloaded from Hugging Face revision
`59f422b2d410fdaf4a9efc71ff12f278abc2a5d1`, SHA-256
`8c47dffa6cbc77e2af663a79012a6f847142d78479d9d7adcf28c5b300e2d83d`.

See the [README tables](../README.md#cpu-benchmarks-speed-and-quality) and
[speed CSV](results/readme-calibrated-2026-09-06/performance.csv) for the measured
medians and min/max ranges. Three samples characterize this run, not universal
speed guarantees or strong confidence intervals.

## Quality measurement contract

The quality table reuses the completed 8,192-position evaluations: 16 WikiText-2
test article windows, 256 prompt tokens and 512 scored tokens each. All candidates
share the same BF16 teacher, prompts, targets and full scoring masks. The
publication script verifies the original archived prompt/target bytes against
the calibrated evaluation hashes, plus teacher and candidate identities.
Perplexity uses pooled token NLL; KL is the teacher-to-candidate distribution KL.
These are subset scores, not full-corpus WikiText perplexity.

Pure Q4_0 and calibrated H128 each have 424,934,656 tensor bytes. At that budget,
H128 achieves 11.0% lower perplexity and 33.3% lower mean KL. Unsloth mixed quants
have larger tensor payloads and lower PPL/KL. Quality and speed are reported
separately; a successful arithmetic regression does not replace corpus evaluation.

The four-document calibration pilot uses disjoint English prose documents and
actual teacher projection inputs in the correct H128 basis. The small held-out
screening suite covers six documents and 1,024 scored tokens. These are bounded
evaluations, not broad certification or proof of general quality superiority.

Quality raw results and their existing provenance limitations are preserved in
the [original comparison archive](results/2026-09-06/raw-results.zip) and the
[calibrated evaluation archive](results/implementation-plan-2026-09-06/raw-results.zip).
The earlier comparator tie-handling change did not change NLL or KL formulas.

## Reproducibility

The [fresh speed archive](results/readme-calibrated-2026-09-06/raw-results.zip)
contains the exact matrix, prompt fixtures, profiles, command lines and provenance.
Extract it at the repository root, restore the frozen executable/DLL files
matching its hashes, and run its matrix with the sequential benchmark runner
using one warmup, three measured runs and affinity 21845. Choose a new output
directory; do not overwrite the recorded series.

`scripts/summarize-readme-comparison.py` validates work counts and quality pairing
before writing the [quality CSV](results/readme-calibrated-2026-09-06/quality.csv),
speed CSV and archive. Aggregation also needs the two earlier quality archives
restored to their original benchmark directories. The [manifest](results/readme-calibrated-2026-09-06/manifest.json)
and [SHA256SUMS](results/readme-calibrated-2026-09-06/SHA256SUMS) cover published
files. Weights and executable build outputs are excluded from Git.
