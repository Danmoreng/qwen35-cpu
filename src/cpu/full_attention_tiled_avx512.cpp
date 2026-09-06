#define TILE_NAME attention_tile_avx512
#define LANES 16
#define SIMD(name) _mm512_##name
#define VEC __m512
#include "full_attention_tiled_simd.inl"
