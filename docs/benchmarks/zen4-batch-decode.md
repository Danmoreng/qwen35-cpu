# Parallel batch decode on AMD Zen 4

The AVX-512 DeltaNet row-pair kernel improved single-request decode on the
Xeon test host. On the Ryzen 5 7600 it reduced throughput when the executor
updated multiple private request states concurrently. The engine now uses
the original AVX2 row kernel for that parallel H128 state update on AMD
family 19h with usable AVX-512. Matrix projections keep their AVX-512/VNNI
dispatch. Single requests and small batches that partition each state across
workers still use the existing state-update dispatch.

## End-to-end decode

AMD Ryzen 5 7600, GCC 16.2.1 Release, six physical cores on CPUs 0–5
(mask 63), FP16 KV and full logits. Each request used 512 fixed prompt
tokens and 128 forced output tokens. Decode counts 127 forwards per request.
Batch rows report total throughput. The sequential runner alternated case
order across one warmup and three measured runs. Rates are medians.

| Model and batch size | Before DeltaNet change | Stock | Updated | Change vs. stock |
| --- | ---: | ---: | ---: | ---: |
| 0.8B, 1 request | — | 115.78 | 114.20 | -1.36% |
| 0.8B, 4 requests | — | 291.00 | 290.43 | -0.20% |
| 0.8B, 16 requests | 557.93 | 536.34 | 555.54 | +3.58% |
| 2B, 1 request | — | 49.16 | 49.00 | -0.34% |
| 2B, 16 requests | — | 352.23 | 362.71 | +2.97% |

Rates are decode tokens/s. The separately built pre-DeltaNet revision
ran in the same series. The compact evidence includes every measured sample.

A separate screening round compared the AVX2 row kernel with the four-row
AVX2 causal-scan kernel in the same parallel decode path. At B16 the row
kernel measured 564.35 tok/s for 0.8B and 361.18 tok/s for 2B.
The causal-scan kernel measured 544.03 and 348.98 tok/s.

## Validation

All 24 CTest checks passed. The 0.8B and calibrated 2B model tests matched
single-request full-vocabulary logits bitwise for batches of 4, 8, 12, 16, 20
and 32 requests. The paged scheduler and HTTP smoke tests passed.

The [evidence](../results/zen4-batch-decode-2026-11-19/summary.json) records
the individual rates, source and binary hashes, fixed inputs and affinity.
Use the recorded 512-token fixture, forced output IDs and benchmark runner
to reproduce the comparison.
