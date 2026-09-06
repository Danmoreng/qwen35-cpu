Independent golden dequantization fixtures generated with NumPy 2.2.6 and
`gguf.quants.dequantize` from llama.cpp commit
`73a43d1f69345aee8bb186ef4b3172cef892f2e5` (MIT).

Each case contains 512 values. `.bin` is the original GGML block layout;
`.f32` is little-endian float32 reference output. These are synthetic data,
not model weights. Random seed: `numpy.random.default_rng(3508)`, sequential
Q4_K, Q5_K, Q6_K, Q8_0 uint8 draws. Superblock scales are overwritten with
alternating +/-0.03125, and Q4_K/Q5_K minima with +/-0.0625.

The fixtures exercise all packed scale/high-bit fields and signed Q6 scales.
Tests also compare scalar and SIMD dots against independent dequantized
weights multiplied by the actual prepared activations, including 1-4 rows,
non-contiguous strides and zero activations. No Python dependency in CTest.
