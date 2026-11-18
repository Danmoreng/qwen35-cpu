#pragma once
#include "systemone.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>

namespace qwen35x::server {
struct Classifier {
  std::vector<std::string> labels;
  std::vector<std::int32_t> tokens;

  static Classifier load(const std::filesystem::path &directory, const QwenTokenizer &tokenizer) {
    const auto path = directory / "classifier.json";
    Classifier profile;
    if (!std::filesystem::exists(path)) return profile;
    std::ifstream stream(path);
    const auto data = systemone::Json::parse(stream);
    const auto &labels = data.at("labels");
    if (!labels.is_object() || labels.size() < 2 || labels.size() > 256)
      throw std::runtime_error("classifier.json requires 2 to 256 label/token pairs");
    std::set<std::int32_t> seen;
    for (const auto &[name, text] : labels.items()) {
      std::vector<std::int32_t> ids;
      std::string error;
      if (name.empty() || !text.is_string() || !tokenizer.encode(text.get<std::string>(), ids, error) ||
          ids.size() != 1 || !seen.insert(ids.front()).second)
        throw std::runtime_error("Classifier labels require distinct single tokens: " + name);
      profile.labels.push_back(name);
      profile.tokens.push_back(ids.front());
    }
    return profile;
  }

  systemone::Json score(const std::vector<float> &logits) const {
    if (logits.size() != labels.size() || logits.empty()) throw std::runtime_error("Missing classifier logits");
    const auto best = std::max_element(logits.begin(), logits.end());
    std::vector<double> weights;
    double sum = 0;
    for (double value : logits) {
      weights.push_back(std::exp(value - *best));
      sum += weights.back();
    }
    systemone::Json probabilities = systemone::Json::object();
    for (std::size_t i = 0; i < labels.size(); ++i) probabilities[labels[i]] = weights[i] / sum;
    return {{"label", labels[best - logits.begin()]}, {"probabilities", probabilities}};
  }
};
} // namespace qwen35x::server
