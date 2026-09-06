#define DOT4_FN(name) q4_dot4_##name##_avx2
#define DOT4_AVX2 1
#define DOT4_WIDE 0
#define DOT4_DPBUSD
#include "q4_dot4_simd.inl"
