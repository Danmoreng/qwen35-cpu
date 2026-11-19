#pragma once
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace qwen35x::server {
// Match Jinja/Python's Unicode whitespace trimming for the supported text template.
inline std::string trim(std::string_view text, bool left = true, bool right = true) {
  static const std::vector<std::string> spaces = {
    " ", "\t", "\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x1f",
    "\xc2\x85", "\xc2\xa0", "\xe1\x9a\x80", "\xe2\x80\x80", "\xe2\x80\x81",
    "\xe2\x80\x82", "\xe2\x80\x83", "\xe2\x80\x84", "\xe2\x80\x85", "\xe2\x80\x86",
    "\xe2\x80\x87", "\xe2\x80\x88", "\xe2\x80\x89", "\xe2\x80\x8a", "\xe2\x80\xa8",
    "\xe2\x80\xa9", "\xe2\x80\xaf", "\xe2\x81\x9f", "\xe3\x80\x80"};
  std::string_view view(text);
  for (const bool front : {true, false}) {
    if ((front && !left) || (!front && !right)) continue;
    while (!view.empty()) {
      const auto it = std::find_if(spaces.begin(), spaces.end(), [&](const auto &space) {
        return front ? view.starts_with(space) : view.ends_with(space);
      });
      if (it == spaces.end()) break;
      if (front) view.remove_prefix(it->size());
      else view.remove_suffix(it->size());
    }
  }
  return std::string(view);
}
}
