#pragma once
#include <cstddef>
namespace qwen35x::cpu::detail {
void gated_delta_net_update_rows_wasm(float *, const float *, const float *, const float *,
  const float *, const float *, float *, std::size_t, std::size_t, std::size_t, std::size_t) noexcept;
void gated_delta_net_update_batch_rows_wasm(float *, const float *, const float *, const float *,
  const float *, const float *, float *, std::size_t, std::size_t, std::size_t, std::size_t,
  std::size_t, std::size_t) noexcept;
}
