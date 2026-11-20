#include <algorithm>
#include <cstddef>
#include <immintrin.h>

namespace qwen35x::cpu::detail {
namespace {
// Each 256-bit half holds eight columns of one row, preserving the AVX2
// FMA chains and the order of their final sum.
__m512 join(__m256 low, __m256 high) noexcept {
  return _mm512_castpd_ps(_mm512_insertf64x4(
    _mm512_castps_pd(_mm512_castps256_ps512(low)), _mm256_castps_pd(high), 1));
}
__m256 high_half(__m512 value) noexcept {
  return _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(value), 1));
}
__m512 repeat8(const float *p) noexcept {
  return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castps_pd(_mm256_loadu_ps(p))));
}
float sum8(__m256 value) noexcept {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value), _mm256_extractf128_ps(value, 1));
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  return _mm_cvtss_f32(_mm_add_ss(sum, _mm_movehdup_ps(sum)));
}

// Each lane belongs to a different value row. Within each row, combine the
// eight column accumulators in the same order as the AVX2 horizontal sum.
__m512 sum_column_chains(const __m512 (&chains)[8]) noexcept {
  const __m512 even = _mm512_add_ps(_mm512_add_ps(chains[0], chains[4]),
                                    _mm512_add_ps(chains[2], chains[6]));
  const __m512 odd = _mm512_add_ps(_mm512_add_ps(chains[1], chains[5]),
                                   _mm512_add_ps(chains[3], chains[7]));
  return _mm512_add_ps(even, odd);
}

template <unsigned Pairs>
void update(float *state, const float *q, const float *k, const float *v,
            float alpha, float beta, float *output) noexcept {
  const __m512 decay = _mm512_set1_ps(alpha);
  __m512 sum_k[Pairs]{};
  for (unsigned c = 0; c < 128; c += 8) {
    const __m512 keys = repeat8(k + c);
    for (unsigned p = 0; p < Pairs; ++p) {
      const float *rows = state + p * 256 + c;
      const __m512 values = join(_mm256_loadu_ps(rows), _mm256_loadu_ps(rows + 128));
      sum_k[p] = _mm512_fmadd_ps(_mm512_mul_ps(values, decay), keys, sum_k[p]);
    }
  }
  __m512 delta[Pairs];
  for (unsigned p = 0; p < Pairs; ++p) {
    const float lo = (v[2 * p] - sum8(_mm512_castps512_ps256(sum_k[p]))) * beta;
    const float hi = (v[2 * p + 1] - sum8(high_half(sum_k[p]))) * beta;
    delta[p] = join(_mm256_set1_ps(lo), _mm256_set1_ps(hi));
  }
  __m512 sum_q[Pairs]{};
  for (unsigned c = 0; c < 128; c += 8) {
    const __m512 keys = repeat8(k + c);
    const __m512 queries = repeat8(q + c);
    for (unsigned p = 0; p < Pairs; ++p) {
      float *rows = state + p * 256 + c;
      const __m512 values = join(_mm256_loadu_ps(rows), _mm256_loadu_ps(rows + 128));
      const __m512 next = _mm512_fmadd_ps(delta[p], keys, _mm512_mul_ps(values, decay));
      _mm256_storeu_ps(rows, _mm512_castps512_ps256(next));
      _mm256_storeu_ps(rows + 128, high_half(next));
      sum_q[p] = _mm512_fmadd_ps(next, queries, sum_q[p]);
    }
  }
  for (unsigned p = 0; p < Pairs; ++p) {
    output[2 * p] = sum8(_mm512_castps512_ps256(sum_q[p]));
    output[2 * p + 1] = sum8(high_half(sum_q[p]));
  }
}
} // namespace

void gated_delta_net_update_rows_avx2(float *, const float *, const float *, const float *,
    const float *, const float *, float *, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;

void gated_delta_net_update_rows_avx512(float *state, const float *q, const float *k,
    const float *v, const float *alpha, const float *beta, float *output,
    std::size_t value_dim, std::size_t begin, std::size_t end) noexcept {
  while (begin < end) {
    const std::size_t head = begin / value_dim;
    const std::size_t stop = std::min(end, begin + (value_dim - begin % value_dim));
    for (; stop - begin >= 8; begin += 8)
      update<4>(state + begin * 128, q + head * 128, k + head * 128, v + begin,
                alpha[head], beta[head], output + begin);
    for (; stop - begin >= 2; begin += 2)
      update<1>(state + begin * 128, q + head * 128, k + head * 128, v + begin,
                alpha[head], beta[head], output + begin);
    if (begin < stop) {
      gated_delta_net_update_rows_avx2(state, q, k, v, alpha, beta, output, 128, value_dim, begin, stop);
      begin = stop;
    }
  }
}

void gated_delta_net_prefill_128_avx512(float *state, const float *q, const float *k,
    const float *v, const float *alpha, const float *beta, float *output,
    std::size_t batch, std::size_t heads, std::size_t value_dim,
    std::size_t head_begin, std::size_t head_end) noexcept {
  constexpr std::size_t columns = 128, rows = 16;
  alignas(64) float tile[columns][rows];
  for (std::size_t head = head_begin; head < head_end; ++head) {
    for (std::size_t row = 0; row < value_dim; row += rows) {
      float *state_rows = state + (head * value_dim + row) * columns;
      // Keep these 16 recurrent rows in an 8 KiB column-major tile for the
      // complete causal scan. Transpose only on entry and exit.
      for (std::size_t col = 0; col < columns; ++col)
        for (std::size_t lane = 0; lane < rows; ++lane)
          tile[col][lane] = state_rows[lane * columns + col];

      for (std::size_t token = 0; token < batch; ++token) {
        const std::size_t ht = head * batch + token;
        const float *keys = k + ht * columns;
        const float *queries = q + ht * columns;
        const __m512 decay = _mm512_set1_ps(alpha[ht]);
        __m512 key_sums[8]{};
        for (std::size_t col = 0; col < columns; col += 8) {
          for (std::size_t chain = 0; chain < 8; ++chain) {
            float *values = tile[col + chain];
            const __m512 decayed = _mm512_mul_ps(_mm512_load_ps(values), decay);
            _mm512_store_ps(values, decayed);
            key_sums[chain] = _mm512_fmadd_ps(decayed,
                _mm512_set1_ps(keys[col + chain]), key_sums[chain]);
          }
        }
        const __m512 target = _mm512_loadu_ps(v + ht * value_dim + row);
        const __m512 delta = _mm512_mul_ps(_mm512_sub_ps(target, sum_column_chains(key_sums)),
                                           _mm512_set1_ps(beta[ht]));
        __m512 query_sums[8]{};
        for (std::size_t col = 0; col < columns; col += 8) {
          for (std::size_t chain = 0; chain < 8; ++chain) {
            float *values = tile[col + chain];
            const __m512 next = _mm512_fmadd_ps(delta, _mm512_set1_ps(keys[col + chain]),
                                                _mm512_load_ps(values));
            _mm512_store_ps(values, next);
            query_sums[chain] = _mm512_fmadd_ps(next,
                _mm512_set1_ps(queries[col + chain]), query_sums[chain]);
          }
        }
        _mm512_storeu_ps(output + (token * heads + head) * value_dim + row,
                         sum_column_chains(query_sums));
      }
      for (std::size_t col = 0; col < columns; ++col)
        for (std::size_t lane = 0; lane < rows; ++lane)
          state_rows[lane * columns + col] = tile[col][lane];
    }
  }
}
} // namespace qwen35x::cpu::detail
