// SPDX-License-Identifier: Apache-2.0
// Adapted from Padra ec909b64ce2a0caee67e0604ce8890a0674a35f5:
// src/text/decoder.cpp and src/text/stream.cpp. Modified for native Qwen bytes,
// header-only ownership. See licenses/padra.txt.
#pragma once
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <nlohmann/json.hpp>
namespace qwen35x::server {
class Utf8Decoder {
  std::string pending_, output_;
public:
  std::string_view push(std::string_view bytes, bool final = false) {
    pending_.append(bytes);
  output_.clear();
  std::size_t i = 0;
  while (i < pending_.size()) {
    const auto lead = static_cast<unsigned char>(pending_[i]);
    int length = lead < 128                     ? 1
                 : lead >= 0xc2 && lead <= 0xdf ? 2
                 : lead >= 0xe0 && lead <= 0xef ? 3
                 : lead >= 0xf0 && lead <= 0xf4 ? 4
                                                : 0;
    if (length == 1) {
      output_.push_back(pending_[i++]);
      continue;
    }
    if (length == 0) {
      output_.append("\xef\xbf\xbd");
      ++i;
      continue;
    }
    int good = 1;
    bool invalid = false;
    for (int j = 1; j < length && i + static_cast<std::size_t>(j) < pending_.size(); ++j) {
      const auto c = static_cast<unsigned char>(pending_[i + j]);
      bool valid = c >= 128 && c <= 191;
      if (j == 1) {
        if (lead == 0xe0)
          valid = valid && c >= 0xa0;
        if (lead == 0xed)
          valid = valid && c < 0xa0;
        if (lead == 0xf0)
          valid = valid && c >= 0x90;
        if (lead == 0xf4)
          valid = valid && c < 0x90;
      }
      if (!valid) {
        invalid = true;
        break;
      }
      ++good;
    }
    if (invalid) {
      output_.append("\xef\xbf\xbd");
      i += good;
      continue;
    }
    if (good < length) {
      if (final) {
        output_.append("\xef\xbf\xbd");
        i += good;
      }
      break;
    }
    output_.append(pending_, i, length);
    i += length;
  }
  pending_.erase(0, i);
  return output_;
  }
};
inline bool valid_utf8(const std::string &text) {
  try { (void)nlohmann::json(text).dump(); return true; }
  catch (const std::exception &) { return false; }
}
class StopMatcher {
  std::vector<std::string> stops_;
  std::size_t max_chunk_;
  std::string pending_, output_;
  bool stopped_ = false;
public:
  bool stopped() const { return stopped_; }
std::size_t HeldSuffix(std::string_view text, std::string_view pattern) {
  for (std::size_t n = std::min(text.size(), pattern.size() - 1); n > 0; --n)
    if (text.substr(text.size() - n) == pattern.substr(0, n))
      return n;
  return 0;
}
StopMatcher(std::vector<std::string> stops, std::size_t max_chunk)
    : stops_(std::move(stops)), max_chunk_(max_chunk) {
  if (max_chunk < 1 || max_chunk > 8192 || stops_.size() > 4)
    throw std::invalid_argument("stop/chunk limits exceeded");
  std::size_t longest = 0;
  for (const auto &stop : stops_) {
    if (stop.empty() || stop.size() > 256 || !valid_utf8(stop))
      throw std::invalid_argument("stop must be nonempty valid UTF8 <=256 bytes");
    longest = std::max(longest, stop.size());
  }
  pending_.reserve(max_chunk + longest);
  output_.reserve(max_chunk + longest);
}
std::string_view Push(std::string_view text) {
  output_.clear();
  if (stopped_)
    return output_;
  if (text.size() > max_chunk_)
    throw std::length_error("decoded chunk exceeds admitted maximum");
  pending_.append(text);
  std::size_t first = std::string::npos;
  for (const auto &stop : stops_)
    first = std::min(first, pending_.find(stop));
  if (first != std::string::npos) {
    output_.assign(pending_, 0, first);
    pending_.clear();
    stopped_ = true;
    return output_;
  }
  std::size_t hold = 0;
  for (const auto &stop : stops_)
    hold = std::max(hold, HeldSuffix(pending_, stop));
  output_.assign(pending_, 0, pending_.size() - hold);
  pending_.erase(0, pending_.size() - hold);
  return output_;
}
std::string_view Finish() {
  output_.clear();
  if (!stopped_)
    output_.swap(pending_);
  pending_.clear();
  return output_;
}
};
} // namespace qwen35x::server
