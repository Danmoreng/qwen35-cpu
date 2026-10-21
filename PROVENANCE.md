# Source provenance

CPU kernels and runtime extracted from [qwen35x](https://github.com/Danmoreng/qwen35x),
revision f766837, on September 6, 2026. Original MIT attribution is retained.
This repository has an independent history and focuses on Qwen3.5-0.8B,
H128/Q4-G32-DOT4 weights, experimental Q4_K_M GGUF, and CPU execution. Model weights are not included.

The GGUF-v3 reader and Qwen3.5 GGUF tensor mapping were ported from the same
qwen35x project. K-quant block layouts and signed Q8_K activation preparation
follow ggml at llama.cpp revision
`73a43d1f69345aee8bb186ef4b3172cef892f2e5` (MIT; see
[ggml license](licenses/ggml.txt)). The independent golden fixtures were generated
with its Python GGUF dequantizer. The native runtime does not link llama.cpp or
ggml; its executor, attention, recurrent state and sampling remain this engine's.

The text-only qev System One prompt renderer and answer formulas in
`src/systemone.h` are adapted from the Apache-2.0 `decider-ai` 1.9.0 protocol
implementation ([Mapika/decider](https://github.com/Mapika/decider)). The native
port retains that file's Apache-2.0 license; its license text is shipped in
`licenses/decider-ai.txt`. Python/decider are used by optional validation tools,
not by the HTTP server at runtime.

The tokenizer statically links utf8proc 2.10.0 for NFC normalization and Unicode
letter/mark/number categories, matching the supported Qwen tokenizer profiles.
Its pinned source archive and checksum are in CMakeLists.txt; the MIT and Unicode
data notices are installed as `licenses/utf8proc.txt`. No shared utf8proc library
is required at runtime.
