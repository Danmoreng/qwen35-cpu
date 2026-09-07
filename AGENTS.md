# Contributor instructions

- Keep code, documentation and commit messages in English.
- Keep README benchmarks focused on the current product: one performance table
  comparing the native engine, llama.cpp and ik_llama.cpp, followed by quantization
  quality. Put historical measurements and optimization narratives in docs.
  Use readable quant names rather than CLI flags in table labels (for example,
  `Q4_0` instead of `Q4_0 --pure`; distinguish the vendor variant as `Unsloth Q4_0`).
- Scope: CPU inference for Qwen3.5-0.8B with H128/Q4-G32-DOT4 artifacts, native pure Q4_0, and experimental Q4_K_M GGUF execution.
- Do not add GPU backends, generic model support, weights or build outputs.
- Keep ISA-specific instructions in isolated translation units with runtime dispatch.
- Run performance measurements sequentially through `scripts/benchmark-inference-seq.ps1`.
- Use three measured runs and one warmup by default. Preserve raw profiles, command lines,
  binary/checkpoint hashes, CPU affinity and upstream revisions for published comparisons.
- Do not equate quantization labels with quality, or compare batched throughput with
  a single-request baseline. Report quality and speed separately.
