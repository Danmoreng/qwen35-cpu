#include "qwen35x/cpu/full_attention_tiled.h"
#include <algorithm>
#include <cmath>
namespace qwen35x::cpu {
namespace {
bool valid(const TiledAttention &a) noexcept {
  return a.dimension == 256 && a.heads > 0 && a.kv_heads > 0 && a.heads % a.kv_heads == 0 &&
         (bool(a.partial_maxima) == bool(a.partial_sums)) && (!a.kv_end || a.kv_end >= a.kv_begin) &&
         (!a.share_gqa || a.heads / a.kv_heads <= 4) &&
         (a.query_tile == 4 || a.query_tile == 8 || a.query_tile == 16) &&
         (a.kv_tile == 32 || a.kv_tile == 64 || a.kv_tile == 128);
}
} // namespace
#if QWEN35X_Q8_0_HAS_AVX2_TU
void attention_tile_avx2(const TiledAttention &, std::size_t, float *,
                         AttentionTileTimes *) noexcept;
#endif
#if QWEN35X_Q8_0_HAS_AVX512_TU
void attention_tile_avx512(const TiledAttention &, std::size_t, float *,
                           AttentionTileTimes *) noexcept;
#endif
const char *tiled_attention_kernel(Q8_0Backend backend) noexcept {
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend))
    return "tiled-avx512-fp32";
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend))
    return "tiled-avx2-fp32";
#endif
  return "rows";
}
std::size_t tiled_attention_scratch_floats(const TiledAttention &a) noexcept {
  if (!valid(a))
    return 0;
  return 2ULL * a.kv_tile * a.dimension +
         a.query_tile * (a.share_gqa ? a.heads / a.kv_heads : 1) * (a.kv_tile + 2 * a.dimension);
}
std::size_t tiled_attention_tasks(const TiledAttention &a) noexcept {
  if (!valid(a))
    return 0;
  return (a.share_gqa ? a.kv_heads : a.heads) * ((a.tokens + a.query_tile - 1) / a.query_tile);
}
bool causal_attention_tiled(const TiledAttention &a, std::size_t task, float *scratch,
                            Q8_0Backend backend, AttentionTileTimes *times) noexcept {
  if (!valid(a) || task >= tiled_attention_tasks(a) || !scratch || !a.queries || (!a.gates && !a.partial_sums) ||
      !a.output || (!a.keys && !a.keys_f16 && !(a.pages && a.pages->keys)) ||
      (!a.values && !a.values_f16 && !(a.pages && a.pages->values)) ||
      a.query_stride < static_cast<std::size_t>(a.heads) * 256 ||
      a.kv_stride < static_cast<std::size_t>(a.kv_heads) * 256)
    return false;
#if QWEN35X_Q8_0_HAS_AVX512_TU
  if (q8_0_backend_uses_avx512(backend)) {
    attention_tile_avx512(a, task, scratch, times);
    return true;
  }
#endif
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    attention_tile_avx2(a, task, scratch, times);
    return true;
  }
#endif
  return false;
}
void merge_attention_partial(float &maximum, float &sum, float *values,
                             float other_maximum, float other_sum,
                             const float *other_values, std::size_t dimension) noexcept {
  if (other_sum == 0) return;
  if (sum == 0) {
    maximum=other_maximum;sum=other_sum;
    std::copy_n(other_values,dimension,values);return;
  }
  const float m=std::max(maximum,other_maximum);
  const float a=std::exp(maximum-m),b=std::exp(other_maximum-m);
  for(std::size_t i=0;i<dimension;++i)values[i]=a*values[i]+b*other_values[i];
  sum=a*sum+b*other_sum;maximum=m;
}
} // namespace qwen35x::cpu
