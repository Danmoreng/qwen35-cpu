#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace qwen35x::detail {
// Prefix query-key pairs in token-major/head-minor row order. Runtime shapes
// have positive heads and nonnegative positions; arithmetic uses 64 bits even
// in WASM32. Every query retains its original accumulation order.
inline std::uint64_t attention_row_work(std::size_t rows, std::size_t heads,
                                        std::size_t position) noexcept {
  const std::uint64_t tokens = rows / heads, remainder = rows % heads;
  return heads * (tokens * (position + 1) + tokens * (tokens ? tokens - 1 : 0) / 2) +
         remainder * (position + tokens + 1);
}
inline std::size_t attention_partition_boundary(
    std::size_t partition, std::size_t partitions, std::size_t rows,
    std::size_t heads, std::size_t position) noexcept {
  if (!partition) return 0;
  if (partition >= partitions) return rows;
  const auto target = attention_row_work(rows, heads, position) * partition / partitions;
  std::size_t low = 0, high = rows;
  while (low < high) {
    const auto middle = low + (high - low) / 2;
    if (attention_row_work(middle, heads, position) < target) low = middle + 1;
    else high = middle;
  }
  return std::clamp(low, partition, rows - (partitions - partition));
}
} // namespace qwen35x::detail
