# Prefill opportunities after the DOT4 decode audit

The next promising work is in the existing intrinsic matrix kernels and
prefill attention. The assembly decode experiment has been removed from the
production source; its measurements and local source archive remain available
in [the assembly report](asm-decode-2026-09-26.md).

No inference default or quantization changes in this investigation. The
benchmark now exposes the existing engine chunk setting as `--cpu-prefill-chunk`
and the existing stage instrumentation as `--profile-prefill`. The latter writes
`PROFILE.json.prefill.csv` and `PROFILE.json.operations.csv`, and marks both
instrumentation flags in the JSON. Profiling requires one cold, non-serving
request. The engine also exposes an optional caller-owned `prefill_profile`
sink; leave diagnostic sinks null for release timing.

## Exploratory chunk measurements

Ryzen 5 7600, GCC 16.2.1 Release, six physical cores on CPUs 0–5, affinity
`0x3f`, Linux, powersave governor, normal desktop session. H128/Q4-G32-DOT4,
B1, FP16 KV, full-logit head, maximum context 16,384, two forced output tokens.
Prefill throughput includes the initial vocabulary projection; the one decode
forward is excluded. Loading and tokenization are excluded. Each case has one
warmup and three measured runs, executed sequentially with alternating case
order through `scripts/benchmark-inference-seq.ps1`.

| Prompt tokens | Chunk | Median prefill tokens/s | Measured range |
| ---: | ---: | ---: | ---: |
| 512 | 128 (current default) | 1,421.02 | 1,295.89–1,525.30 |
| 512 | 256 | 1,445.14 | 1,261.29–1,446.74 |
| 512 | 512 | 1,324.64 | 1,235.08–1,399.73 |
| 4,096 | 128 (current default) | 1,274.55 | 1,072.01–1,287.40 |
| 4,096 | 256 | 1,225.26 | 1,041.88–1,233.67 |
| 4,096 | 512 | 1,116.05 | 1,076.77–1,204.64 |

The substantial run-to-run variance prevents a precise speed claim. Larger
chunks do not establish a useful improvement and do not justify changing the
default. Larger chunks also increase scratch size and the duration of a
synchronous prefill scheduling step. The first-chunk row/tiled attention
selection can change with chunk size; identical forced outputs are not evidence
of equal logits. No cross-chunk numerical or quality equivalence is claimed.

## Diagnostic profile

A separate series uses the same runner, affinity, warmup and measured-run count,
with 128-token chunks and instrumentation enabled. Values below are medians of
each stage's fraction of that run's prefill forward time. These are diagnostic
proportions, not predictions of uninstrumented latency; clock reads and executor
instrumentation add overhead. Independently computed medians need not sum to
100%. Matrix preparation/kernel intervals and the listed attention intervals
do not overlap. Matrix intervals include both attention projections and MLPs.

| Stage | 512 tokens | 4,096 tokens |
| --- | ---: | ---: |
| Q4 matrix kernels including dispatch/wait | 57.5% | 48.8% |
| H128/Q8 activation preparation | 6.7% | 3.4% |
| Linear-attention preparation | 6.7% | 5.8% |
| DeltaNet recurrent core | 14.3% | 11.8% |
| Full-attention preparation | 1.4% | 1.3% |
| Full-attention core | 4.8% | 22.0% |
| Remaining work | 8.6% | 6.6% |

The attention CSV also records summed elapsed worker intervals for packing,
QK, softmax and PV. Those sums must not be added to wall time or interpreted
as OS CPU accounting.

## Recommended experiments

1. **Reuse unpacked Q4 weights across more token work.**
   `src/cpu/q4_dot4_prefill_avx512.cpp` currently computes a 16-row by 8-token
   tile and unpacks weights again for each token tile. Test a 16-token tile or
   a cache-blocked loop that shares unpacking across token tiles while preserving
   each output's block/FMA order. Check compiler spills and activation/output
   locality: simply increasing register pressure may lose performance. This
   targets the largest measured component without changing the artifact format.
   As an illustration only, a 20% faster component occupying half the runtime
   yields about 9% end-to-end speedup, not 20%.
2. **Improve full-attention tiling and panel reuse for long prompts.**
   `src/cpu/full_attention_tiled_simd.inl` converts FP16 K/V and transposes K
   for every query tile; panels are reused inside a task but repacked by later
   tasks. Investigate bounded panel reuse and query/KV tile choices against
   cache footprint and available parallelism. The default AVX-512 query tile is
   16 and KV tile is 32. Gains matter more at 4,096 tokens than at 512. Preserve
   causal bounds and validate softmax/reduction changes numerically.
3. **Specialize the DeltaNet core and reduce preparation passes.**
   `src/cpu/delta_net_avx2.cpp` reads and decays state twice per token: once for
   the key dot product and once for update/output. A 128-column specialization
   could retain parts of the state in registers; an isolated AVX-512 variant
   needs runtime dispatch. The preparation in `src/runtime/prefill.inl` also
   performs normalization, gate calculations and token-to-head layout copies.
   Test fusion or head-parallel preparation. A chunkwise DeltaNet algorithm is
   a larger, separate numerical project, not a free replacement for recurrence.

None of these kernel changes has been implemented or assigned a measured
speedup here. Prioritize the matrix tile experiment, then long-prompt attention.

## Validation and reproduction

All 14 configured CTests pass, including the expanded 5,040-case DOT4 test.
Two optional NumPy evaluation tests are unavailable on this host. All chunk
and instrumented cases complete successfully. The existing 128-token default
and production kernels are retained.

The raw profiles, per-run diagnostic CSVs, commands, binary/checkpoint hashes,
source hashes, source patch, frozen binary and host data remain under ignored
`benchmarks/prefill-audit-20260926/`. No external inference engine is compared.
[Chunk summary](results/prefill-audit-2026-09-26/chunks.csv),
[per-run stage breakdown](results/prefill-audit-2026-09-26/stage-breakdown.csv),
and [provenance](results/prefill-audit-2026-09-26/manifest.json) accompany this report.

```sh
cmake --build build-perf-audit -j 6
ctest --test-dir build-perf-audit --output-on-failure
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/prefill-audit-20260926/matrix.json -OutputDir benchmarks/prefill-chunks-new -Runs 3 -WarmupRuns 1 -Affinity 63
pwsh -NoProfile -File scripts/benchmark-inference-seq.ps1 -Matrix benchmarks/prefill-audit-20260926/profile-matrix.json -OutputDir benchmarks/prefill-profile-new -Runs 3 -WarmupRuns 1 -Affinity 63
```

Matrices are also preserved with the results. On another checkout, adjust
executable/model/token paths to the corresponding locally built binary and
matching inputs. Keep the instrumented and uninstrumented series separate.
