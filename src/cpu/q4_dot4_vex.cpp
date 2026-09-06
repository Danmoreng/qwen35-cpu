#define DOT4_FN(name) q4_dot4_##name##_vex
#define DOT4_AVX2 0
#define DOT4_WIDE 0
#define DOT4_DPBUSD _mm256_dpbusd_avx_epi32
#include "q4_dot4_simd.inl"
