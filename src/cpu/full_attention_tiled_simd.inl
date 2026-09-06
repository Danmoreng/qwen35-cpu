#include "qwen35x/cpu/full_attention_tiled.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <limits>

namespace qwen35x::cpu {
namespace {
#if LANES == 8
[[nodiscard]] __m256 tile_exp(__m256 value) noexcept {
  value = _mm256_min_ps(value, _mm256_set1_ps(88.3762626647949F));
  value = _mm256_max_ps(value, _mm256_set1_ps(-88.3762626647949F));

  __m256 exponent =
      _mm256_fmadd_ps(value, _mm256_set1_ps(1.44269504088896341F), _mm256_set1_ps(0.5F));
  exponent = _mm256_floor_ps(exponent);
  value = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(0.693359375F), value);
  value = _mm256_fnmadd_ps(exponent, _mm256_set1_ps(-2.12194440e-4F), value);

  const __m256 squared = _mm256_mul_ps(value, value);
  __m256 polynomial = _mm256_set1_ps(1.9875691500e-4F);
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(1.3981999507e-3F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(8.3334519073e-3F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(4.1665795894e-2F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(1.6666665459e-1F));
  polynomial = _mm256_fmadd_ps(polynomial, value, _mm256_set1_ps(5.0000001201e-1F));
  polynomial = _mm256_fmadd_ps(polynomial, squared, value);
  polynomial = _mm256_add_ps(polynomial, _mm256_set1_ps(1.0F));

  __m256i integer_exponent = _mm256_cvttps_epi32(exponent);
  integer_exponent = _mm256_add_epi32(integer_exponent, _mm256_set1_epi32(127));
  integer_exponent = _mm256_slli_epi32(integer_exponent, 23);
  return _mm256_mul_ps(polynomial, _mm256_castsi256_ps(integer_exponent));
}
#else
[[nodiscard]] __m512 tile_exp(__m512 value) noexcept {
  value = _mm512_min_ps(value, _mm512_set1_ps(88.3762626647949F));
  value = _mm512_max_ps(value, _mm512_set1_ps(-88.3762626647949F));
  __m512 exponent =
      _mm512_fmadd_ps(value, _mm512_set1_ps(1.44269504088896341F), _mm512_set1_ps(0.5F));
  exponent = _mm512_floor_ps(exponent);
  value = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(0.693359375F), value);
  value = _mm512_fnmadd_ps(exponent, _mm512_set1_ps(-2.12194440e-4F), value);
  const __m512 squared = _mm512_mul_ps(value, value);
  __m512 polynomial = _mm512_set1_ps(1.9875691500e-4F);
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(1.3981999507e-3F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(8.3334519073e-3F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(4.1665795894e-2F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(1.6666665459e-1F));
  polynomial = _mm512_fmadd_ps(polynomial, value, _mm512_set1_ps(5.0000001201e-1F));
  polynomial = _mm512_fmadd_ps(polynomial, squared, value);
  polynomial = _mm512_add_ps(polynomial, _mm512_set1_ps(1.0F));
  __m512i integer_exponent = _mm512_cvttps_epi32(exponent);
  integer_exponent = _mm512_add_epi32(integer_exponent, _mm512_set1_epi32(127));
  integer_exponent = _mm512_slli_epi32(integer_exponent, 23);
  return _mm512_mul_ps(polynomial, _mm512_castsi512_ps(integer_exponent));
}
#endif
using Clock = std::chrono::steady_clock;
double tile_elapsed(Clock::time_point start) noexcept {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
template <int R>
void qk(const float *q, const float *panel, float *out, int B, int stride, int count,
        float scale) noexcept {
  // Two independent SIMD columns hide FMA latency and reuse broadcasts.
  for (int col = 0; col < B; col += 2 * LANES) {
    VEC a00 = SIMD(setzero_ps)();
    VEC a01 = SIMD(setzero_ps)();
    VEC a10 = SIMD(setzero_ps)();
    VEC a11 = SIMD(setzero_ps)();
    VEC a20 = SIMD(setzero_ps)();
    VEC a21 = SIMD(setzero_ps)();
    VEC a30 = SIMD(setzero_ps)();
    VEC a31 = SIMD(setzero_ps)();
    for (int i = 0; i < 256; ++i) {
      const VEC v0 = SIMD(loadu_ps)(panel + i * B + col + 0 * LANES);
      const VEC v1 = SIMD(loadu_ps)(panel + i * B + col + 1 * LANES);
      if constexpr (R > 0) {
        const VEC x = SIMD(set1_ps)(q[0 * stride + i]);
        a00 = SIMD(fmadd_ps)(x, v0, a00);
        a01 = SIMD(fmadd_ps)(x, v1, a01);
      }
      if constexpr (R > 1) {
        const VEC x = SIMD(set1_ps)(q[1 * stride + i]);
        a10 = SIMD(fmadd_ps)(x, v0, a10);
        a11 = SIMD(fmadd_ps)(x, v1, a11);
      }
      if constexpr (R > 2) {
        const VEC x = SIMD(set1_ps)(q[2 * stride + i]);
        a20 = SIMD(fmadd_ps)(x, v0, a20);
        a21 = SIMD(fmadd_ps)(x, v1, a21);
      }
      if constexpr (R > 3) {
        const VEC x = SIMD(set1_ps)(q[3 * stride + i]);
        a30 = SIMD(fmadd_ps)(x, v0, a30);
        a31 = SIMD(fmadd_ps)(x, v1, a31);
      }
    }
    if constexpr (R > 0)
      SIMD(storeu_ps)(out + 0 * B + col + 0 * LANES, SIMD(mul_ps)(a00, SIMD(set1_ps)(scale)));
    if constexpr (R > 0)
      SIMD(storeu_ps)(out + 0 * B + col + 1 * LANES, SIMD(mul_ps)(a01, SIMD(set1_ps)(scale)));
    if constexpr (R > 1)
      SIMD(storeu_ps)(out + 1 * B + col + 0 * LANES, SIMD(mul_ps)(a10, SIMD(set1_ps)(scale)));
    if constexpr (R > 1)
      SIMD(storeu_ps)(out + 1 * B + col + 1 * LANES, SIMD(mul_ps)(a11, SIMD(set1_ps)(scale)));
    if constexpr (R > 2)
      SIMD(storeu_ps)(out + 2 * B + col + 0 * LANES, SIMD(mul_ps)(a20, SIMD(set1_ps)(scale)));
    if constexpr (R > 2)
      SIMD(storeu_ps)(out + 2 * B + col + 1 * LANES, SIMD(mul_ps)(a21, SIMD(set1_ps)(scale)));
    if constexpr (R > 3)
      SIMD(storeu_ps)(out + 3 * B + col + 0 * LANES, SIMD(mul_ps)(a30, SIMD(set1_ps)(scale)));
    if constexpr (R > 3)
      SIMD(storeu_ps)(out + 3 * B + col + 1 * LANES, SIMD(mul_ps)(a31, SIMD(set1_ps)(scale)));
  }
}
template <int R>
void pv(const float *q, const float *panel, float *out, int B, int stride, int count,
        float scale) noexcept {
  // Two independent SIMD columns hide FMA latency and reuse broadcasts.
  for (int col = 0; col < 256; col += 2 * LANES) {
    VEC a00;
    if constexpr (R > 0)
      a00 = SIMD(loadu_ps)(out + 0 * 256 + col + 0 * LANES);
    VEC a01;
    if constexpr (R > 0)
      a01 = SIMD(loadu_ps)(out + 0 * 256 + col + 1 * LANES);
    VEC a10;
    if constexpr (R > 1)
      a10 = SIMD(loadu_ps)(out + 1 * 256 + col + 0 * LANES);
    VEC a11;
    if constexpr (R > 1)
      a11 = SIMD(loadu_ps)(out + 1 * 256 + col + 1 * LANES);
    VEC a20;
    if constexpr (R > 2)
      a20 = SIMD(loadu_ps)(out + 2 * 256 + col + 0 * LANES);
    VEC a21;
    if constexpr (R > 2)
      a21 = SIMD(loadu_ps)(out + 2 * 256 + col + 1 * LANES);
    VEC a30;
    if constexpr (R > 3)
      a30 = SIMD(loadu_ps)(out + 3 * 256 + col + 0 * LANES);
    VEC a31;
    if constexpr (R > 3)
      a31 = SIMD(loadu_ps)(out + 3 * 256 + col + 1 * LANES);
    for (int i = 0; i < count; ++i) {
      const VEC v0 = SIMD(loadu_ps)(panel + i * 256 + col + 0 * LANES);
      const VEC v1 = SIMD(loadu_ps)(panel + i * 256 + col + 1 * LANES);
      if constexpr (R > 0) {
        const VEC x = SIMD(set1_ps)(q[0 * stride + i]);
        a00 = SIMD(fmadd_ps)(x, v0, a00);
        a01 = SIMD(fmadd_ps)(x, v1, a01);
      }
      if constexpr (R > 1) {
        const VEC x = SIMD(set1_ps)(q[1 * stride + i]);
        a10 = SIMD(fmadd_ps)(x, v0, a10);
        a11 = SIMD(fmadd_ps)(x, v1, a11);
      }
      if constexpr (R > 2) {
        const VEC x = SIMD(set1_ps)(q[2 * stride + i]);
        a20 = SIMD(fmadd_ps)(x, v0, a20);
        a21 = SIMD(fmadd_ps)(x, v1, a21);
      }
      if constexpr (R > 3) {
        const VEC x = SIMD(set1_ps)(q[3 * stride + i]);
        a30 = SIMD(fmadd_ps)(x, v0, a30);
        a31 = SIMD(fmadd_ps)(x, v1, a31);
      }
    }
    if constexpr (R > 0)
      SIMD(storeu_ps)(out + 0 * 256 + col + 0 * LANES, a00);
    if constexpr (R > 0)
      SIMD(storeu_ps)(out + 0 * 256 + col + 1 * LANES, a01);
    if constexpr (R > 1)
      SIMD(storeu_ps)(out + 1 * 256 + col + 0 * LANES, a10);
    if constexpr (R > 1)
      SIMD(storeu_ps)(out + 1 * 256 + col + 1 * LANES, a11);
    if constexpr (R > 2)
      SIMD(storeu_ps)(out + 2 * 256 + col + 0 * LANES, a20);
    if constexpr (R > 2)
      SIMD(storeu_ps)(out + 2 * 256 + col + 1 * LANES, a21);
    if constexpr (R > 3)
      SIMD(storeu_ps)(out + 3 * 256 + col + 0 * LANES, a30);
    if constexpr (R > 3)
      SIMD(storeu_ps)(out + 3 * 256 + col + 1 * LANES, a31);
  }
}
// Compile profiling out of the throughput path, including all clock reads.
template <bool Profile>
void tile_impl(const TiledAttention &a, std::size_t task, float *scratch,
               AttentionTileTimes *times) noexcept {
  constexpr int D = 256;
  const int B = a.kv_tile;
  const std::size_t tile_count = (a.tokens + a.query_tile - 1) / a.query_tile;
  const int group = a.share_gqa ? a.heads / a.kv_heads : 1;
  const int head = static_cast<int>(task / tile_count) * group;
  const std::size_t first = (task % tile_count) * a.query_tile;
  const int tokens = static_cast<int>(std::min<std::size_t>(a.query_tile, a.tokens - first));
  const int rows = tokens * group;
  const int kv_head = head / (a.heads / a.kv_heads);
  float *K = scratch, *V = K + D * B, *S = V + B * D, *A = S + a.query_tile * group * B,
        *Q = A + a.query_tile * group * D;
  for (int r = 0; r < rows; ++r)
    std::copy_n(a.queries + (first + r / group) * a.query_stride + (head + r % group) * D, D,
                Q + r * D);
  float maxima[64], sums[64]{};
  std::fill_n(maxima, rows, -std::numeric_limits<float>::infinity());
  std::fill_n(A, rows * D, 0.f);
  Clock::time_point start;
  const auto end = a.kv_end ? std::min(a.kv_end, a.position + first + tokens) : a.position + first + tokens;
  for (std::size_t base = a.kv_begin; base < end; base += B) {
    const int count =
        static_cast<int>(std::min<std::size_t>(B, end - base));
    if constexpr (Profile)
      start = Clock::now();
    for (int d = 0; d < D; d += 8)
      for (int k = 0; k < B; k += 8) {
        __m256 x[8];
        for (int j = 0; j < 8; ++j) {
          const auto offset = (base + k + j) * a.kv_stride + kv_head * D + d;
          if (k + j < count) {
            x[j] = (a.keys_f16 || (a.pages && a.pages->keys)) ? _mm256_cvtph_ps(_mm_loadu_si128(
                                    reinterpret_cast<const __m128i *>(attention_cache_row(a.keys_f16, a.pages ? a.pages->keys : nullptr, base+k+j, a.kv_stride) + kv_head*D+d)))
                              : _mm256_loadu_ps(a.keys + offset);
            const auto v = (a.values_f16 || (a.pages && a.pages->values))
                               ? _mm256_cvtph_ps(_mm_loadu_si128(
                                     reinterpret_cast<const __m128i *>(attention_cache_row(a.values_f16, a.pages ? a.pages->values : nullptr, base+k+j, a.kv_stride) + kv_head*D+d)))
                               : _mm256_loadu_ps(a.values + offset);
            _mm256_storeu_ps(V + (k + j) * D + d, v);
          } else {
            x[j] = _mm256_setzero_ps();
            _mm256_storeu_ps(V + (k + j) * D + d, x[j]);
          }
        }
        const auto t0 = _mm256_unpacklo_ps(x[0], x[1]), t1 = _mm256_unpackhi_ps(x[0], x[1]);
        const auto t2 = _mm256_unpacklo_ps(x[2], x[3]), t3 = _mm256_unpackhi_ps(x[2], x[3]);
        const auto t4 = _mm256_unpacklo_ps(x[4], x[5]), t5 = _mm256_unpackhi_ps(x[4], x[5]);
        const auto t6 = _mm256_unpacklo_ps(x[6], x[7]), t7 = _mm256_unpackhi_ps(x[6], x[7]);
        const auto u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xee);
        const auto u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xee);
        const auto u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xee);
        const auto u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xee);
        _mm256_storeu_ps(K + (d + 0) * B + k, _mm256_permute2f128_ps(u0, u4, 0x20));
        _mm256_storeu_ps(K + (d + 1) * B + k, _mm256_permute2f128_ps(u1, u5, 0x20));
        _mm256_storeu_ps(K + (d + 2) * B + k, _mm256_permute2f128_ps(u2, u6, 0x20));
        _mm256_storeu_ps(K + (d + 3) * B + k, _mm256_permute2f128_ps(u3, u7, 0x20));
        _mm256_storeu_ps(K + (d + 4) * B + k, _mm256_permute2f128_ps(u0, u4, 0x31));
        _mm256_storeu_ps(K + (d + 5) * B + k, _mm256_permute2f128_ps(u1, u5, 0x31));
        _mm256_storeu_ps(K + (d + 6) * B + k, _mm256_permute2f128_ps(u2, u6, 0x31));
        _mm256_storeu_ps(K + (d + 7) * B + k, _mm256_permute2f128_ps(u3, u7, 0x31));
      }
    if constexpr (Profile) {
      times->pack_ms += tile_elapsed(start);
      start = Clock::now();
    }
    for (int r = 0; r < rows; r += 4) {
      const auto *q = Q + r * D;
      switch (std::min(4, rows - r)) {
      case 4:
        qk<4>(q, K, S + r * B, B, D, count, a.scale);
        break;
      case 3:
        qk<3>(q, K, S + r * B, B, D, count, a.scale);
        break;
      case 2:
        qk<2>(q, K, S + r * B, B, D, count, a.scale);
        break;
      case 1:
        qk<1>(q, K, S + r * B, B, D, count, a.scale);
        break;
      }
    }
    if constexpr (Profile) {
      times->qk_ms += tile_elapsed(start);
      start = Clock::now();
    }
    for (int r = 0; r < rows; ++r) {
      const int valid = static_cast<int>(
          std::min<std::size_t>(count, a.position + first + r / group + 1 > base
                                           ? a.position + first + r / group + 1 - base
                                           : 0));
      float *s = S + r * B;
      if (valid == 0) {
        std::fill_n(s, B, 0.f);
        continue;
      }
      float maximum = maxima[r];
      for (int k = 0; k < valid; ++k)
        maximum = std::max(maximum, s[k]);
      if (maximum != maxima[r]) {
        const float rescale = std::exp(maxima[r] - maximum);
        sums[r] *= rescale;
        for (int d = 0; d < D; d += LANES)
          SIMD(storeu_ps)(A + r * D + d,
                          SIMD(mul_ps)(SIMD(loadu_ps)(A + r * D + d), SIMD(set1_ps)(rescale)));
        maxima[r] = maximum;
      }
      float sum = 0.f;
      int k = 0;
      for (; k + LANES <= valid; k += LANES)
        SIMD(storeu_ps)(s + k,
                        tile_exp(SIMD(sub_ps)(SIMD(loadu_ps)(s + k), SIMD(set1_ps)(maximum))));
      for (; k < valid; ++k)
        s[k] = std::exp(s[k] - maximum);
      for (k = 0; k < valid; ++k)
        sum += s[k];
      sums[r] += sum;
      // Never feed masked scores through an exp approximation.
      std::fill(s + valid, s + B, 0.f);
    }
    if constexpr (Profile) {
      times->softmax_ms += tile_elapsed(start);
      start = Clock::now();
    }
    for (int r = 0; r < rows; r += 4) {
      switch (std::min(4, rows - r)) {
      case 4:
        pv<4>(S + r * B, V, A + r * D, B, B, count, 1);
        break;
      case 3:
        pv<3>(S + r * B, V, A + r * D, B, B, count, 1);
        break;
      case 2:
        pv<2>(S + r * B, V, A + r * D, B, B, count, 1);
        break;
      case 1:
        pv<1>(S + r * B, V, A + r * D, B, B, count, 1);
        break;
      }
    }
    if constexpr (Profile)
      times->pv_ms += tile_elapsed(start);
  }
  if constexpr (Profile)
    start = Clock::now();
  for (int r = 0; r < rows; ++r) {
    if (a.partial_sums) {
      const auto row = (first+r/group)*a.heads+head+r%group;
      a.partial_maxima[row]=maxima[r];a.partial_sums[row]=sums[r];
    }
    for (int d = 0; d < D; ++d) {
      const auto offset = (first + r / group) * a.query_stride + (head + r % group) * D + d;
      if (a.partial_sums) {a.output[offset]=A[r*D+d];continue;}
      const float g = a.gates[offset], e = std::exp(-std::abs(g));
      const float gate = g >= 0 ? 1.f / (1.f + e) : e / (1.f + e);
      a.output[offset] = (A[r * D + d] / sums[r]) * gate;
    }
  }
  if constexpr (Profile)
    times->pv_ms += tile_elapsed(start);
}
} // namespace
void TILE_NAME(const TiledAttention &a, std::size_t task, float *scratch,
               AttentionTileTimes *times) noexcept {
  if (times)
    tile_impl<true>(a, task, scratch, times);
  else
    tile_impl<false>(a, task, scratch, nullptr);
}
} // namespace qwen35x::cpu
