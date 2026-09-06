#define TILE_NAME attention_tile_avx2
#define LANES 8
#define SIMD(name) _mm256_##name
#define VEC __m256
#include "full_attention_tiled_simd.inl"
