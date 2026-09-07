# ik_llama.cpp batch-16 decode

IK supports independent-sequence decode for Qwen3.5. The pinned upstream build
`fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf` runs batch 8 and prefills sixteen
private requests, but aborts when decoding sixteen requests together. The failure
is `GGML_ASSERT(cgraph->n_nodes < cgraph->size)`.

## Graph-capacity correction

For Qwen3.5, `llama_model::max_nodes()` estimates graph capacity from token count
and model tensor count. The mixed-sequence recurrent path in
`delta_net::build_layer_attn_linear()` builds its subgraph separately for each
sequence. The existing estimate does not account for that expansion.

The [local patch](../tools/ik-llama-comparison/patches/qwen35-multi-sequence-graph.patch)
extends `llama_context::max_nodes()` for Qwen3.5 contexts with multiple sequences.
It reserves at least one single-token graph budget per possible independent
sequence, bounded by the batch token count. The same estimate is used for graph
creation, scheduler capacity and graph metadata allocation. Single-sequence
contexts retain their original capacity calculation.

This changes graph metadata reservation, not tensor arithmetic, model weights,
quantization, KV precision, token counts or attention kernels. It is a local
correction to the pinned fork, not a claim that stock upstream B16 works. The
original checkout and binaries remain unchanged. The README identifies the patch
for the IK B16 scores.

Primary implementation references at the pinned revision:

- [Graph capacity](https://github.com/ikawrakow/ik_llama.cpp/blob/fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf/src/llama-model.h#L624).
- [Mixed-sequence recurrent graph](https://github.com/ikawrakow/ik_llama.cpp/blob/fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf/src/llama-delta-net.cpp#L646).

## Validation and measurement

The diagnostic runs isolate the failure: stock B16 prefill-only passes, stock
B8 decode passes, stock B16 decode fails, and patched B16 decode passes. For the
initial pure-Q4_0 check, patched B8 is bitwise identical to stock B8. Patched B16
also matches the concatenation of two independent stock B8 contexts bitwise for
all sixteen sequences after one decode forward.

The full validation uses all six IK quantizations, 512 prompt tokens and 128
forced outputs. Each patched B16 run is checked against two stock B8 runs after
127 decode forwards per sequence. All prompts are private: the final prompt
token is offset by sequence index, including offsets 8–15 in the second reference
context. Full-vocabulary final logits, finiteness, work counts, input hashes and
binary identities are checked. Validation dumps are collected after timing and
their diagnostic run timings are not published as performance results.

After 127 forwards, B16 and the two B8 reference contexts are **not bitwise
identical**. Across the six quants, maximum absolute final-logit differences are
0.298–0.692 and mean KL from stock B8 to patched B16 is 0.00082–0.00536 nats.
All 96 corresponding final top-1 tokens match. This does not prove identical
free-running generations or equal quality at every batch size.

A separate request-order check reverses the mapping of the sixteen private
prompts to sequence IDs. Reversing the resulting output rows restores the
original B16 logits **bit for bit for all six quants**; each run has sixteen
distinct output rows. A full-length patched B8 Q4_0 run also matches stock B8
bit for bit. These checks support request isolation and preservation of the
working B8 path for this workload. They do not establish the exact numerical
cause of the B8/B16 differences. The order check uses a local test-only harness
copy that reverses the prompt-token offsets; its source and binary hashes are
preserved in the manifest. Performance uses the unchanged harness.

Performance uses one warmup and three measured runs through
`scripts/benchmark-inference-seq.ps1`, alternating case order. All runs use eight
threads on physical V-Cache cores 0–7 (`0xff`), FP16 KV, a 16,384-token limit per
request, full-vocabulary logits, runtime repacking, `n_batch=2048`, `n_ubatch=512`
and no GPU/KQV offload. Codex is minimized and XFCE compositing disabled at
2560×1600 / 240 Hz; the wrapper restores the desktop. Native H128 and mainline
pure Q4_0 run alongside the six IK candidates as controls. No validation or
compilation runs concurrently with these measurements.

Batch-16 throughput is the aggregate of 16 × 127 decode forwards. It is not
sixteen independent process timings added together, and it is not a single
sequence with sixteen speculative tokens. IK's recurrent graph still processes
some work per sequence; batching support does not imply every operator is batched
efficiently.

The [results](results/ik-batch16-2026-09-07/performance.csv),
[validation](results/ik-batch16-2026-09-07/validation.csv) and
[manifest](results/ik-batch16-2026-09-07/manifest.json) preserve medians, min/max,
hashes and provenance. Raw profiles, exact commands, logit dumps, desktop state
and the build logs remain under ignored `benchmarks/ik-b16-20260907/`.
Builds, weights and raw archives are not committed.

## Reproduction

Use a separate checkout of the pinned IK revision, leaving the stock comparator
available for validation:

```sh
git -C build-ik-source worktree add --detach ../build-ik-b16-source \
  fe215a8ccdce6b844d2a3a3bbde08ae76a6284bf
git -C build-ik-b16-source apply \
  "$PWD/tools/ik-llama-comparison/patches/qwen35-multi-sequence-graph.patch"
cmake -S tools/ik-llama-comparison -B build-ik-b16-linux -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DIK_LLAMA_SOURCE_DIR="$PWD/build-ik-b16-source" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-ik-b16-linux --target ik-fixed-cpu-bench -j 8
```

The fixed-token harness is unchanged. B16 uses `--cpu-batch 16` and
`--ik-repack`; exact workload arguments are preserved in the manifest. This
experiment validates the correction for Qwen3.5-0.8B and the measured workload,
not every model, sequence count or serving configuration.
