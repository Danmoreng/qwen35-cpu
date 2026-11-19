#pragma once
#include "server_protocol.h"
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>

namespace qwen35x::server {
struct Reply { int status; Json body; };
// One inference writer and one HTTP reader. Network writes never hold this lock.
// Terminal state has its own lane, so a full delta queue cannot lose an error.
class StreamQueue {
  std::mutex mutex_;
  std::condition_variable changed_;
  std::deque<std::string> text_;
  std::size_t bytes_ = 0;
  std::optional<Reply> terminal_;
public:
  static constexpr std::size_t capacity = 64, byte_capacity = 256 * 1024, max_chunk_bytes = 8192;
  std::size_t room_for_output() {
    std::lock_guard lock(mutex_);
    const auto slots = std::min(capacity - text_.size(), (byte_capacity - bytes_) / max_chunk_bytes);
    // Finishing a reply may flush both the UTF-8 decoder and a held stop suffix.
    return slots > 2 ? slots - 2 : 0;
  }
  void push(std::string text) {
    if (text.empty()) return;
    std::lock_guard lock(mutex_);
    if (text.size() > max_chunk_bytes || text_.size() >= capacity || bytes_ + text.size() > byte_capacity || terminal_)
      throw std::runtime_error("Stream queue bound exceeded");
    bytes_ += text.size(); text_.push_back(std::move(text)); changed_.notify_one();
  }
  void finish(Reply result) {
    std::lock_guard lock(mutex_);
    terminal_ = std::move(result); changed_.notify_one();
  }
  bool pop(std::string &text, std::optional<Reply> &terminal) {
    std::unique_lock lock(mutex_);
    changed_.wait_for(lock, std::chrono::milliseconds(50), [&] { return !text_.empty() || terminal_; });
    if (!text_.empty()) {
      text = std::move(text_.front()); bytes_ -= text.size(); text_.pop_front(); return true;
    }
    if (terminal_) terminal = terminal_;
    return false;
  }
};
} // namespace qwen35x::server
