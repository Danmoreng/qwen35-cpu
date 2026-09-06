#pragma once
#include "qwen35x/cpu/full_attention.h"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>

namespace qwen35x::cpu {
// Append-only FP16 KV pages. Copies share immutable page references; appending
// to a shared partial page creates one private copy before writing. Row views
// remain valid until the owning cache is appended, assigned or destroyed.
class KvPagesF16 {
public:
  static constexpr std::size_t page_tokens = 256;
  struct Page {
    std::vector<std::uint16_t> keys, values;
    explicit Page(std::size_t width)
        : keys(page_tokens * width), values(page_tokens * width) {}
  };
  KvPagesF16() = default;
  KvPagesF16(std::size_t width, std::size_t capacity)
      : width_(width), capacity_(capacity) {
    if (!width || !capacity)
      throw std::invalid_argument("KV page dimensions must be positive");
  }
  std::size_t size() const { return size_; }
  std::size_t width() const { return width_; }
  std::size_t capacity() const { return capacity_; }
  std::size_t page_count() const { return pages_.size(); }
  std::size_t payload_bytes() const {
    return pages_.size() * page_tokens * width_ * 2 * sizeof(std::uint16_t);
  }
  const std::vector<std::shared_ptr<const Page>> &pages() const {
    return pages_;
  }
  const std::uint16_t *const *keys() const { return keys_.data(); }
  const std::uint16_t *const *values() const { return values_.data(); }
  void append(const float *key, const float *value, Q8_0Backend backend) {
    if (!key || !value || size_ >= capacity_)
      throw std::out_of_range("KV append exceeds capacity");
    const auto page = size_ / page_tokens, row = size_ % page_tokens;
    // Reserve row tables before mutating page ownership or writing values.
    if (keys_.capacity() <= size_ || values_.capacity() <= size_) {
      const auto reserve = std::min(capacity_, (page + 1) * page_tokens);
      keys_.reserve(reserve);
      values_.reserve(reserve);
    }
    std::shared_ptr<Page> writable;
    if (page == pages_.size()) {
      writable = std::make_shared<Page>(width_);
      pages_.push_back(writable);
    } else if (pages_[page].use_count() > 1) {
      writable = std::make_shared<Page>(*pages_[page]);
      pages_[page] = writable;
      for (std::size_t i = 0; i < row; ++i) {
        keys_[page * page_tokens + i] = writable->keys.data() + i * width_;
        values_[page * page_tokens + i] = writable->values.data() + i * width_;
      }
    } else
      writable = std::const_pointer_cast<Page>(pages_[page]);
    auto *k = writable->keys.data() + row * width_,
         *v = writable->values.data() + row * width_;
    attention_cache_store_f16(key, k, width_, backend);
    attention_cache_store_f16(value, v, width_, backend);
    keys_.push_back(k);
    values_.push_back(v);
    ++size_;
  }
  KvPagesF16 prefix(std::size_t tokens) const {
    if (!tokens || tokens > size_)
      throw std::out_of_range("Invalid KV prefix extent");
    KvPagesF16 result(width_, tokens);
    result.size_ = tokens;
    result.pages_.assign(pages_.begin(),
                         pages_.begin() +
                             (tokens + page_tokens - 1) / page_tokens);
    result.keys_.assign(keys_.begin(), keys_.begin() + tokens);
    result.values_.assign(values_.begin(), values_.begin() + tokens);
    return result;
  }
  KvPagesF16 branch(std::size_t capacity) const {
    if (capacity < size_)
      throw std::out_of_range("Branch capacity smaller than prefix");
    auto result = *this;
    result.capacity_ = capacity;
    return result;
  }

private:
  std::size_t width_ = 0, capacity_ = 0, size_ = 0;
  std::vector<std::shared_ptr<const Page>> pages_;
  std::vector<const std::uint16_t *> keys_, values_;
};
} // namespace qwen35x::cpu
