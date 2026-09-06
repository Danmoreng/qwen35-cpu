// Requires AVX512F/VL/VNNI, not the independent VEX AVX-VNNI extension.
#define QWEN35X_VNNI_FN(name) name##_evex_vnni
#define QWEN35X_DPBUSD _mm256_dpbusd_epi32
#include "q4_0_vnni_impl.inl"
