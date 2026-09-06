// VEX AVX-VNNI implementation; shared arithmetic also builds as EVEX-256.
#define QWEN35X_VNNI_FN(name) name##_avx_vnni
#define QWEN35X_DPBUSD _mm256_dpbusd_avx_epi32
#include "q4_0_vnni_impl.inl"
