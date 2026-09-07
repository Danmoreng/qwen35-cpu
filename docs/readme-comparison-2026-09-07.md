# B+C 256 H128 versus llama.cpp

The native candidate is the validated 256-document B+C H128/Q4-G32-DOT4
artifact, SHA256 `013fbfaa03760e759181301ddaf964bb5c200c50c50617afe72557fd65bcbf0a`.
It is compared with llama.cpp pure Q4_0 and Unsloth Q4_0, Q4_K_M and IQ4_XS.
Quantization labels are not treated as equal quality or equal tensor payloads.

## Speed contract

All five candidates run together in a fresh sequential series through
`scripts/benchmark-inference-seq.ps1`, with one warmup and three measured runs,
alternating case order. No quality or calibration inference runs concurrently.

- Ryzen 9 9955HX3D, eight threads, affinity 21845 (`0x5555`, physical VCache
  cores), FP16 KV and maximum context 16,384.
- Identical externally supplied tokens and full-vocabulary logits.
- B1/P512/N128, B1/P4096/N2 and B16/P512/N128 workloads. The N2 workload supplies
  only the long-prefill number, not a decode-speed claim.
- Prefill forward throughput includes the final vocabulary head. Decode counts
  N-1 actual forwards per request. Batch 16 is aggregate throughput, compared
  only with batch 16. Loading, tokenization and HTTP are excluded; requests
  have private state and no prefix-cache credit.

The same frozen executable/DLL set as the previous public comparison is used;
every file's hash was verified before running. The upstream llama.cpp revision
is `73a43d1f69345aee8bb186ef4b3172cef892f2e5`, CPU-only MSVC with AVX-512/VNNI.
Exact commands, checkpoint/binary/DLL/input hashes, all profiles and the complete
matrix are retained locally under `benchmarks/readme-bc256-2026-09-07/`.
Only compact CSVs and [provenance](results/readme-bc256-2026-09-07/manifest.json)
are committed; raw logs and archives stay outside Git.

See [performance.csv](results/readme-bc256-2026-09-07/performance.csv) for medians
and min/max values. Three runs characterize this CPU/workload; they are not a
universal speed guarantee. The README speed table uses this fresh series in full,
not a new native row spliced into the previous comparator measurements.

## Quality contract

The native row uses the completed B+C 256 evaluation. Comparator rows reuse
their existing evaluations on the identical 16 WikiText-2 test article windows,
256 prompt tokens and 512 scored tokens per window (8,192 total). Teacher hash,
prompt/target hashes, scoring masks and comparator checkpoint hashes are checked
against the original archive. PPL uses pooled target NLL; KL is mean
teacher-to-candidate distribution KL. These are English-prose subset scores,
not full WikiText perplexity.

The native result is **PPL 15.80467 / KL 0.060190**, with 424,934,656 tensor
bytes and 424,964,864 complete file bytes. At the same tensor budget as pure
Q4_0, PPL is 12.3% lower and KL is 58.4% lower. Unsloth Q4_0 has lower PPL but
higher KL; the larger Q4_K_M and IQ4_XS have lower PPL and KL. Storage, quality
and speed are presented separately.

See [quality.csv](results/readme-bc256-2026-09-07/quality.csv), the
[larger calibration study](g32-large-calibration-2026-09-07.md), its
[compact results](results/g32-large-calibration-2026-09-07/summary.json),
and the [original comparator archive](results/2026-09-06/raw-results.zip).
The independent mixed final test is a separate suite and is not used to rank
Unsloth checkpoints, which were not all evaluated there.

## Reproduction

Restore the archived benchmark matrix and prompt fixtures at their recorded
paths, plus executables/DLLs matching the provenance hashes. Run the sequential
benchmark in a new output directory with one warmup, three measured runs and
affinity 21845. `scripts/summarize-bc256-readme-comparison.py` checks work counts
and quality pairing, then writes compact CSVs, manifest and SHA256SUMS to Git and retains the raw
archive only under ignored `benchmarks/`. Raw reproduction inputs must be
retained locally or regenerated; they are not bundled in the new Git results.
Weights and executable build outputs are excluded from Git.
