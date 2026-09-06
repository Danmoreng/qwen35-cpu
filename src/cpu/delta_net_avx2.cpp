#include "qwen35x/cpu/delta_net.h"

#include <immintrin.h>

#include <cstddef>

namespace qwen35x::cpu::detail {

namespace {

[[nodiscard]] float horizontal_sum_f32(const __m256 value) noexcept {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
  sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
  return _mm_cvtss_f32(sum);
}

template <std::size_t RowCount>
void update_value_rows(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float decay_scalar,
  const float beta_scalar,
  float * output,
  const std::size_t key_dim) noexcept {
  const __m256 decay = _mm256_set1_ps(decay_scalar);
  __m256 state_dot_k_vectors[RowCount]{};
  for (std::size_t row = 0; row < RowCount; ++row) {
    state_dot_k_vectors[row] = _mm256_setzero_ps();
  }

  std::size_t column = 0;
  for (; column + 8 <= key_dim; column += 8) {
    const __m256 k_values = _mm256_loadu_ps(k + column);
    for (std::size_t row = 0; row < RowCount; ++row) {
      const __m256 decayed = _mm256_mul_ps(
        _mm256_loadu_ps(state + row * key_dim + column), decay);
      state_dot_k_vectors[row] = _mm256_fmadd_ps(
        decayed, k_values, state_dot_k_vectors[row]);
    }
  }

  float state_dot_k[RowCount];
  for (std::size_t row = 0; row < RowCount; ++row) {
    state_dot_k[row] = horizontal_sum_f32(state_dot_k_vectors[row]);
  }
  for (; column < key_dim; ++column) {
    for (std::size_t row = 0; row < RowCount; ++row) {
      const float decayed = state[row * key_dim + column] * decay_scalar;
      state_dot_k[row] += decayed * k[column];
    }
  }

  float deltas[RowCount];
  for (std::size_t row = 0; row < RowCount; ++row) {
    deltas[row] = (v[row] - state_dot_k[row]) * beta_scalar;
  }

  __m256 state_dot_q_vectors[RowCount]{};
  for (std::size_t row = 0; row < RowCount; ++row) {
    state_dot_q_vectors[row] = _mm256_setzero_ps();
  }
  column = 0;
  for (; column + 8 <= key_dim; column += 8) {
    const __m256 k_values = _mm256_loadu_ps(k + column);
    const __m256 q_values = _mm256_loadu_ps(q + column);
    for (std::size_t row = 0; row < RowCount; ++row) {
      const __m256 updated = _mm256_fmadd_ps(
        _mm256_set1_ps(deltas[row]),
        k_values,
        _mm256_mul_ps(
          _mm256_loadu_ps(state + row * key_dim + column), decay));
      _mm256_storeu_ps(state + row * key_dim + column, updated);
      state_dot_q_vectors[row] = _mm256_fmadd_ps(
        updated, q_values, state_dot_q_vectors[row]);
    }
  }
  float state_dot_q[RowCount];
  for (std::size_t row = 0; row < RowCount; ++row) {
    state_dot_q[row] = horizontal_sum_f32(state_dot_q_vectors[row]);
  }
  for (; column < key_dim; ++column) {
    for (std::size_t row = 0; row < RowCount; ++row) {
      const float updated = state[row * key_dim + column] * decay_scalar +
        deltas[row] * k[column];
      state[row * key_dim + column] = updated;
      state_dot_q[row] += updated * q[column];
    }
  }
  for (std::size_t row = 0; row < RowCount; ++row) {
    output[row] = state_dot_q[row];
  }
}

void update_value_row_tile(
  const std::size_t row_count,
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float decay,
  const float beta,
  float * output,
  const std::size_t key_dim) noexcept {
  switch (row_count) {
    case 4:
      update_value_rows<4>(state, q, k, v, decay, beta, output, key_dim);
      break;
    case 3:
      update_value_rows<3>(state, q, k, v, decay, beta, output, key_dim);
      break;
    case 2:
      update_value_rows<2>(state, q, k, v, decay, beta, output, key_dim);
      break;
    default:
      update_value_rows<1>(state, q, k, v, decay, beta, output, key_dim);
      break;
  }
}

} // namespace

void gated_delta_net_update_rows_avx2(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  const std::size_t key_dim,
  const std::size_t value_dim,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  for (std::size_t flat_row = row_begin; flat_row < row_end; ++flat_row) {
    const std::size_t head = flat_row / value_dim;
    const std::size_t value_row = flat_row - head * value_dim;
    float * state_row = state + flat_row * key_dim;
    const float * q_head = q + head * key_dim;
    const float * k_head = k + head * key_dim;
    const __m256 decay = _mm256_set1_ps(alpha[head]);

    __m256 state_dot_k_vector = _mm256_setzero_ps();
    std::size_t column = 0;
    for (; column + 8 <= key_dim; column += 8) {
      const __m256 decayed = _mm256_mul_ps(_mm256_loadu_ps(state_row + column), decay);
      _mm256_storeu_ps(state_row + column, decayed);
      state_dot_k_vector = _mm256_fmadd_ps(
        decayed, _mm256_loadu_ps(k_head + column), state_dot_k_vector);
    }
    float state_dot_k = horizontal_sum_f32(state_dot_k_vector);
    for (; column < key_dim; ++column) {
      const float decayed = state_row[column] * alpha[head];
      state_row[column] = decayed;
      state_dot_k += decayed * k_head[column];
    }

    const float delta_scalar =
      (v[head * value_dim + value_row] - state_dot_k) * beta[head];
    const __m256 delta = _mm256_set1_ps(delta_scalar);
    __m256 state_dot_q_vector = _mm256_setzero_ps();
    column = 0;
    for (; column + 8 <= key_dim; column += 8) {
      const __m256 updated = _mm256_fmadd_ps(
        delta, _mm256_loadu_ps(k_head + column), _mm256_loadu_ps(state_row + column));
      _mm256_storeu_ps(state_row + column, updated);
      state_dot_q_vector = _mm256_fmadd_ps(
        updated, _mm256_loadu_ps(q_head + column), state_dot_q_vector);
    }
    float state_dot_q = horizontal_sum_f32(state_dot_q_vector);
    for (; column < key_dim; ++column) {
      const float updated = state_row[column] + delta_scalar * k_head[column];
      state_row[column] = updated;
      state_dot_q += updated * q_head[column];
    }
    output[head * value_dim + value_row] = state_dot_q;
  }
}

void gated_delta_net_update_batch_rows_avx2(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  const std::size_t batch_size,
  const std::size_t head_count,
  const std::size_t key_dim,
  const std::size_t value_dim,
  const std::size_t head_begin,
  const std::size_t head_end) noexcept {
  const std::size_t state_rows = head_count * value_dim;
  for (std::size_t head = head_begin; head < head_end; ++head) {
    for (std::size_t token = 0; token < batch_size; ++token) {
      const std::size_t head_token = head * batch_size + token;
      const float * q_head = q + head_token * key_dim;
      const float * k_head = k + head_token * key_dim;
      for (std::size_t value_row = 0; value_row < value_dim; value_row += 4) {
        const std::size_t flat_row = head * value_dim + value_row;
        const std::size_t tile_rows = value_dim - value_row < 4
          ? value_dim - value_row : 4;
        update_value_row_tile(
          tile_rows,
          state + flat_row * key_dim,
          q_head,
          k_head,
          v + head_token * value_dim + value_row,
          alpha[head_token],
          beta[head_token],
          output + token * state_rows + flat_row,
          key_dim);
      }
    }
  }
}

} // namespace qwen35x::cpu::detail
