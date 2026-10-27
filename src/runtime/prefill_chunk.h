#pragma once
#include <algorithm>
#include <cstddef>

namespace qwen35x::detail {
inline constexpr std::size_t prefill_attention_row_limit = 128;

// Finish both the canonical chunk and the initial row-attention interval.
// Larger chunks must not move the first queries to the tiled attention kernel.
constexpr std::size_t prefill_chunk_token_count(std::size_t position,
    std::size_t remaining, std::size_t chunk) noexcept {
  if (!remaining || !chunk) return 0;
  auto count = std::min(chunk - position % chunk, remaining);
  if (position < prefill_attention_row_limit)
    count = std::min(count, prefill_attention_row_limit - position);
  return count;
}
}
