# ik_llama.cpp batch-16 decode

The October 22, 2026 comparison uses upstream
`98aca8692ae334e6e23e4d39ac43ce0d21029c35`. Stock pure-Q4_0 B16 still aborts with
`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`. A separate build applies the existing
seven-line graph-capacity correction. Only the published IK B16 speeds use that
build; B1, prefill and quality use unmodified upstream.

## Graph-capacity correction

Qwen3.5's mixed-sequence recurrent graph builds work separately for each
sequence. The original capacity estimate does not account for that expansion.
The [local patch](../../tools/ik-llama-comparison/patches/qwen35-multi-sequence-graph.patch)
extends `llama_context::max_nodes()` to reserve at least one single-token graph
budget per possible independent sequence, bounded by batch token count.
Single-sequence contexts retain their original capacity calculation.

This changes graph metadata reservation, not tensor arithmetic, model weights,
quantization, KV precision, token counts or attention kernels. It is a local
correction, not a claim that stock upstream B16 works. The same patch applies
cleanly to the new upstream revision without modification.

## Validation and measurement

Fresh validation covers all six IK quantizations at 512 prompt tokens and 128
forced outputs, with final full-vocabulary logits captured after 127 decode
forwards per sequence. Each candidate has stock B8, patched B8 and patched B16
runs. Diagnostic timings are excluded from published performance results.

- Stock and patched B8 logits match bit for bit for all six candidates.
- Every patched B16 run produces sixteen finite, distinct output rows.
- The first eight patched B16 rows are compared with the corresponding stock
  B8 rows: all 48 final top-1 tokens match across the six candidates.
- B8 and B16 are **not bitwise identical**. Maximum absolute logit differences
  range from 0.306 to 0.623; mean KL from stock B8 to patched B16 ranges from
  0.000762 to 0.005894 nats. The refresh does not compare the last eight B16 rows
  against a second B8 context or repeat the request-order permutation test.

These checks support preservation of the working B8 path and successful B16
execution for this workload. They do not establish identical free-running
outputs, equal quality at every batch size, or the precise numerical cause of
B8/B16 differences. The wider September validation, including a second B8
reference and request-order permutation, remains in the
[original report](https://github.com/Danmoreng/qwen35-cpu/blob/6d38cca33e548aaeda2dd27b21fef689d47c7d79/docs/benchmarks/ik-batch16.md)
and is evidence for its older revision only.

Performance uses one warmup and three measured runs in the same sequential
series as native and mainline comparators. All runs use eight threads on physical
V-Cache cores 0–7 (`0xff`), FP16 KV, context limit 16,384 per request, full-vocabulary
logits, runtime repacking, `n_batch=2048`, `n_ubatch=512` and no GPU/KQV offload.
See the [common measurement contract](cpu.md#performance-contract), including
the recorded CPU power controls and desktop state.

B16 throughput is the aggregate of 16 × 127 decode forwards across private
requests. It is not a single sequence with sixteen speculative tokens.
Recurrent graph work remains partly per sequence; batching support does not
imply that every operator batches efficiently.

The [results](../results/cpu-refresh-2026-10-22/performance.csv),
[validation](../results/cpu-refresh-2026-10-22/validation-summary.json) and
[manifest](../results/cpu-refresh-2026-10-22/manifest.json) preserve measurements,
source revisions, hashes and provenance. Raw profiles, commands, final logit
dumps and logs remain under ignored `benchmarks/cpu-refresh-2026-10-22/`.

## Reproduction

Use a separate checkout of the pinned IK revision:

```sh
git -C build-ik-source worktree add --detach ../build-ik-refresh-b16-source \
  98aca8692ae334e6e23e4d39ac43ce0d21029c35
git -C build-ik-refresh-b16-source apply \
  "$PWD/tools/ik-llama-comparison/patches/qwen35-multi-sequence-graph.patch"
cmake -S tools/ik-llama-comparison -B build-ik-refresh-b16 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIK_LLAMA_SOURCE_DIR="$PWD/build-ik-refresh-b16-source" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ik-refresh-b16 --target ik-fixed-cpu-bench -j 8
```

The fixed-token harness is unchanged. B16 uses `--cpu-batch 16` and `--ik-repack`;
exact arguments are in the published matrix. This validates the correction for
Qwen3.5-0.8B and the measured workload, not every sequence count or configuration.
