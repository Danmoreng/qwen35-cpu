#include "../src/runtime/prefill_chunk.h"
#include <initializer_list>
#include <iostream>

using qwen35x::detail::prefill_chunk_token_count;

bool schedule(std::size_t total, std::size_t chunk,
              std::initializer_list<std::size_t> expected) {
  std::size_t position = 0;
  for (auto count : expected) {
    if (prefill_chunk_token_count(position, total - position, chunk) != count)
      return false;
    position += count;
  }
  return position == total;
}

int main() {
  if (!schedule(579, 512, {128, 384, 67}) ||
      !schedule(579, 128, {128, 128, 128, 128, 67}) ||
      !schedule(129, 64, {64, 64, 1}) ||
      !schedule(221, 96, {96, 32, 64, 29}) ||
      !schedule(127, 512, {127}) ||
      !schedule(128, 512, {128}) ||
      !schedule(129, 512, {128, 1}) ||
      prefill_chunk_token_count(74, 505, 512) != 54 ||
      prefill_chunk_token_count(221, 358, 512) != 291 ||
      prefill_chunk_token_count(512, 67, 512) != 67 ||
      prefill_chunk_token_count(0, 0, 512) != 0 ||
      prefill_chunk_token_count(0, 10, 0) != 0) {
    std::cerr << "Prefill attention/chunk boundary mismatch\n";
    return 1;
  }
  std::cout << "Prefill attention and resumed-prefix boundaries passed\n";
}
