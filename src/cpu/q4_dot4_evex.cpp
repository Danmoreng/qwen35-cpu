#define DOT4_FN(name) q4_dot4_##name##_evex
#define DOT4_AVX2 0
#define DOT4_WIDE 1
#define DOT4_DPBUSD _mm256_dpbusd_epi32
#include "q4_dot4_simd.inl"
