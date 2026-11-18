#pragma once
#include "qwen35x/runtime/cpu_engine.h"
#include <algorithm>
#include <array>
#include <unordered_map>

namespace qwen35x::detail {
struct DraftProposal {
  std::array<std::int32_t, 3> tokens{};
  std::size_t count = 0, source_offset = 0;
};

// Index only suffixes with a known successor. All vocabulary IDs fit in 18 bits.
class TokenDraft {
  std::unordered_map<std::uint64_t, std::size_t> successor_;
  std::size_t indexed_ = 3, cursor_ = 0;

  template<class TokenAt>
  static std::uint64_t key(TokenAt at, std::size_t end) {
    return (std::uint64_t(at(end-3)) << 36) | (std::uint64_t(at(end-2)) << 18) |
           std::uint64_t(at(end-1));
  }

public:
  DraftProposal propose(const CpuRequestSpec &spec, std::span<const std::int32_t> output,
                        std::size_t limit) {
    DraftProposal result;
    const bool source_mode = spec.draft_mode == CpuDraftMode::input;
    const auto &source = spec.draft_source_tokens.empty() ? spec.prompt_tokens : spec.draft_source_tokens;
    const auto prompt_size = spec.prompt_tokens.size();
    const auto history_size = prompt_size + output.size();
    auto history = [&](std::size_t i) {
      return i < prompt_size ? spec.prompt_tokens[i] : output[i-prompt_size];
    };
    const auto size = source_mode ? source.size() : history_size;
    auto at = [&](std::size_t i) { return source_mode ? source[i] : history(i); };
    for (; indexed_ < size; ++indexed_)
      successor_[key(at, indexed_)] = indexed_;

    auto start = size;
    if (source_mode && !output.empty()) {
      if (cursor_ < size && source[cursor_] == output.back()) {
        start = cursor_ + 1;
      } else {
        // Recover alignment near the previous copy position after insertions or deletions.
        const auto begin = cursor_ > 8 ? cursor_ - 8 : 0;
        std::size_t best = 1;
        for (auto i = begin; i < std::min(size, cursor_ + 32); ++i) {
          std::size_t n = 0;
          while (n < std::min({i+1, output.size(), std::size_t(8)}) &&
                 source[i-n] == output[output.size()-1-n]) ++n;
          if (n > best) { best = n; start = i+1; }
        }
      }
    }
    if (start == size && (source_mode ? output.size() : history_size) >= 3) {
      const auto match = successor_.find(key(history, history_size));
      if (match != successor_.end()) start = match->second;
    }
    if (source_mode && output.size() == 1 && start == size && !spec.draft_source_tokens.empty())
      start = 0;
    result.source_offset = start;
    result.count = std::min({limit, result.tokens.size(), size - start});
    for (std::size_t i = 0; i < result.count; ++i) result.tokens[i] = at(start+i);
    return result;
  }

  void accepted(const DraftProposal &proposal, std::size_t length) {
    cursor_ = proposal.source_offset + length;
  }
};
} // namespace qwen35x::detail
