# Historical CPU comparison

Imported from the original repository; these measurements predate the standalone
extraction. They provide a traceable baseline, not a benchmark of this HEAD.

- [All samples](results/cpu-engine-comparison-2026-09-05-samples.csv)
- [Median summary](results/cpu-engine-comparison-2026-09-05-summary.csv)
- [Metadata and checkpoint hashes](results/cpu-engine-comparison-2026-09-05-metadata.json)

September 5, 2026; Ryzen 9 9955HX3D, Windows, MSVC Release, same `0xffff`
affinity for both engines, 8 and 12 threads. Three measured new processes and
one warmup per configuration, executed sequentially. The comparison used the
same BF16 source model, H128/Q4-G32-DOT4 versus llama.cpp Q4_0 `--pure`, without
MTP, speculative decoding or GPU offload. FP16 KV and FP32 recurrent state;
reserved context 8192, one sequence.

llama.cpp revision `74a7c897f049c17e7080423aa2111776eff6ebbf` used its native CPU
build, repacking, Flash Attention, batch 2048 and microbatch 512. Its adapter
called the public API and unmodified upstream kernels. This was a fixed-token
comparison, not a default random-input llama-bench run.

The shared fixture in `configs/` supplied token IDs, repeated cyclically for
long prompts. Prefill was measured separately without a vocabulary-head output.
Decode evaluated the full vocabulary on both engines and used forced continuation
tokens. N output tokens corresponded to N-1 timed decode forwards; the first
prediction was produced by prefill. Loading, tokenization and checkpoint
conversion were excluded. Ratios compare median throughputs, not total request
latency or quantization quality.

Later optimizations changed prefill and multi-request performance. Rerun the
matched protocol before publishing a new headline. The standalone batch harness
currently includes first-token production in its initialization timing: do not
compare that column directly with these historical head-free prefill numbers.
