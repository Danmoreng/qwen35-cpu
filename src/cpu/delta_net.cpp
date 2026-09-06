#include "qwen35x/cpu/delta_net.h"

#include <cstddef>

namespace qwen35x::cpu {

namespace detail {

void gated_delta_net_update_rows_scalar(
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

    float state_dot_k = 0.0F;
    for (std::size_t column = 0; column < key_dim; ++column) {
      const float decayed = state_row[column] * alpha[head];
      state_row[column] = decayed;
      state_dot_k += decayed * k_head[column];
    }

    const float delta = (v[head * value_dim + value_row] - state_dot_k) * beta[head];
    float state_dot_q = 0.0F;
    for (std::size_t column = 0; column < key_dim; ++column) {
      const float updated = state_row[column] + delta * k_head[column];
      state_row[column] = updated;
      state_dot_q += updated * q_head[column];
    }
    output[head * value_dim + value_row] = state_dot_q;
  }
}

void gated_delta_net_update_batch_rows_scalar(
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
      for (std::size_t value_row = 0; value_row < value_dim; ++value_row) {
        const std::size_t flat_row = head * value_dim + value_row;
        float * state_row = state + flat_row * key_dim;
        float state_dot_k = 0.0F;
        for (std::size_t column = 0; column < key_dim; ++column) {
          const float decayed = state_row[column] * alpha[head_token];
          state_row[column] = decayed;
          state_dot_k += decayed * k_head[column];
        }
        const float delta =
          (v[head_token * value_dim + value_row] - state_dot_k) * beta[head_token];
        float state_dot_q = 0.0F;
        for (std::size_t column = 0; column < key_dim; ++column) {
          const float updated = state_row[column] + delta * k_head[column];
          state_row[column] = updated;
          state_dot_q += updated * q_head[column];
        }
        output[token * state_rows + flat_row] = state_dot_q;
      }
    }
  }
}

#if QWEN35X_Q8_0_HAS_AVX2_TU
void gated_delta_net_update_rows_avx2(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  std::size_t key_dim,
  std::size_t value_dim,
  std::size_t row_begin,
  std::size_t row_end) noexcept;

void gated_delta_net_update_batch_rows_avx2(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  std::size_t batch_size,
  std::size_t head_count,
  std::size_t key_dim,
  std::size_t value_dim,
  std::size_t row_begin,
  std::size_t row_end) noexcept;
#endif

} // namespace detail

void gated_delta_net_update_rows(
  float * state,
  const float * q,
  const float * k,
  const float * v,
  const float * alpha,
  const float * beta,
  float * output,
  const std::size_t head_count,
  const std::size_t key_dim,
  const std::size_t value_dim,
  const std::size_t row_begin,
  const std::size_t row_end,
  const Q8_0Backend backend) noexcept {
  const std::size_t total_rows = head_count * value_dim;
  if (row_begin >= row_end || row_begin >= total_rows) {
    return;
  }
  const std::size_t bounded_end = row_end < total_rows ? row_end : total_rows;
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::gated_delta_net_update_rows_avx2(
      state, q, k, v, alpha, beta, output, key_dim, value_dim, row_begin, bounded_end);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::gated_delta_net_update_rows_scalar(
    state, q, k, v, alpha, beta, output, key_dim, value_dim, row_begin, bounded_end);
}

void gated_delta_net_update_batch_rows(
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
  const std::size_t row_begin,
  const std::size_t row_end,
  const Q8_0Backend backend) noexcept {
  if (batch_size == 0 || row_begin >= row_end || row_begin >= head_count) {
    return;
  }
  const std::size_t bounded_end = row_end < head_count ? row_end : head_count;
#if QWEN35X_Q8_0_HAS_AVX2_TU
  if (q8_0_backend_uses_avx2(backend)) {
    detail::gated_delta_net_update_batch_rows_avx2(
      state,
      q,
      k,
      v,
      alpha,
      beta,
      output,
      batch_size,
      head_count,
      key_dim,
      value_dim,
      row_begin,
      bounded_end);
    return;
  }
#else
  static_cast<void>(backend);
#endif
  detail::gated_delta_net_update_batch_rows_scalar(
    state,
    q,
    k,
    v,
    alpha,
    beta,
    output,
    batch_size,
    head_count,
    key_dim,
    value_dim,
    row_begin,
    bounded_end);
}

} // namespace qwen35x::cpu
