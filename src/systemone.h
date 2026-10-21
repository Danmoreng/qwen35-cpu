// SPDX-License-Identifier: Apache-2.0
// Text-only qev System One protocol, adapted from decider-ai 1.9.0
// (https://github.com/Mapika/decider). See licenses/decider-ai.txt.
// Prompt/answer code is independent of the scheduler and inference executor.
#pragma once
#include "qwen35x/tokenizer/tokenizer.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace qwen35x::systemone {
using Json = nlohmann::ordered_json;
inline void require(bool value, const std::string &message) {
  if (!value) throw std::runtime_error(message);
}
// Python json.dumps(..., ensure_ascii=False) uses spaces after separators.
inline std::string json_text(const Json &value) {
  const auto compact = value.dump();
  std::string out; bool quoted = false, escaped = false;
  for (const char c : compact) {
    out += c;
    if (quoted) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') quoted = false;
    } else if (c == '"') quoted = true;
    else if (c == ',' || c == ':') out += ' ';
  }
  return out;
}
inline std::string text(const Json &v) { return v.is_string() ? v.get<std::string>() : json_text(v); }
inline bool empty(const Json &v) { return v.is_null() || (v.is_string() && v.get_ref<const std::string &>().empty()); }
inline Json annotate(const Json &v) {
  if (v.is_array()) {
    Json out = Json::array();
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto item = annotate(v[i]);
      if (v.size() >= 8) {
        Json marked = {{"_index", i}};
        if (item.is_object()) for (const auto &[k, x] : item.items()) marked[k] = x;
        else marked["value"] = item;
        item = std::move(marked);
      }
      out.push_back(std::move(item));
    }
    return out;
  }
  if (v.is_object()) {
    Json out = Json::object();
    for (const auto &[k, x] : v.items()) out[k] = annotate(x);
    return out;
  }
  return v;
}
inline std::string state_text(const Json &v) { return v.is_string() ? v.get<std::string>() : json_text(annotate(v)); }
struct Question {
  std::string id, type, instruction;
  std::vector<std::string> options, names, legend;
};
inline Question question(const std::string &id, const Json &spec) {
  require(spec.is_object(), "Question must be an object");
  Question q; q.id = id; q.type = spec.value("type", std::string("choice"));
  if (q.type == "bool") q.type = "noul";
  auto crit = spec.value("criteria", spec.value("options", Json()));
  auto raw = spec.value("instructions", spec.value("question", Json("")));
  if (q.type == "noul" && empty(raw)) {
    require(crit.is_object() && (!empty(crit.value("true", Json())) || !empty(crit.value("false", Json()))),
            "noul without instructions needs true/false descriptions");
    q.instruction = "Which answer fits the context?";
  } else q.instruction = text(raw);
  require(!q.instruction.empty(), "Question without instructions");
  if (q.type == "choice") {
    if (crit.is_array()) {
      Json mapped = Json::object();
      for (const auto &v : crit) mapped[v.is_string() ? v.get<std::string>() : text(v)] = nullptr;
      crit = std::move(mapped);
    }
    require(crit.is_object() && crit.size() >= 2 && crit.size() <= 255, "choice criteria must contain 2..255 options");
    for (const auto &[k, v] : crit.items()) { q.names.push_back(k); q.options.push_back(empty(v) ? k : k + ": " + text(v)); }
  } else if (q.type == "score") {
    if (crit.is_object()) {
      std::vector<std::pair<double, Json>> sorted;
      for (const auto &[k, v] : crit.items()) {
        std::size_t used; double n = std::stod(k, &used);
        require(used == k.size() && std::isfinite(n), "Score legend keys must be finite numbers");
        sorted.emplace_back(n, v);
      }
      std::stable_sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
      crit = Json::array(); for (const auto &[k, v] : sorted) crit.push_back(v);
    }
    require(crit.is_array() && crit.size() >= 2 && crit.size() <= 10, "score criteria must contain 2..10 ordered levels");
    for (std::size_t i = 0; i < crit.size(); ++i) {
      q.names.push_back(std::to_string(i)); q.legend.push_back(text(crit[i]));
      q.options.push_back(std::to_string(i) + ": " + q.legend.back());
    }
  } else if (q.type == "noul") {
    require(crit.is_null() || crit.is_object(), "noul criteria must be an object");
    q.names = {"false", "true"};
    for (int i = 0; i < 2; ++i) {
      auto description = crit.is_null() ? Json() : crit.value(q.names[i], Json());
      std::string name = i ? "yes" : "no";
      q.options.push_back(empty(description) ? name : name + ": " + text(description));
    }
  } else throw std::runtime_error("Unknown question type: " + q.type);
  // Same training-time neutralization as qev.py; never shuffle option order.
  for (auto &option : q.options) {
    auto first = option.find_first_not_of(" \t\r\n"), last = option.find_last_not_of(" \t\r\n");
    std::string key = first == std::string::npos ? "" : option.substr(first, last - first + 1);
    for (auto &c : key) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    if (key.starts_with("none of the above") || key == "none" || key == "n/a" || key == "none of these") option = "not listed here";
  }
  return q;
}
inline std::vector<std::int32_t> encode(const QwenTokenizer &tok, const std::string &s) {
  std::vector<std::int32_t> out; std::string error;
  require(tok.encode(s, out, error), error); return out;
}
struct Config {
  std::vector<std::string> labels;
  std::vector<std::int32_t> ids;
  double choice = 1.7, noul = .9, score = 1.25;
  static std::optional<Config> load(const std::string &directory, const QwenTokenizer &tok) {
    std::ifstream file(directory + "/qev_config.json");
    if (!file) return {};
    Json j; file >> j;
    require(j.at("format") == "qev-systemone/1" && j.at("model") == "qev-0.8b", "Unsupported qev configuration");
    Config c; c.labels = j.at("letter_names").get<std::vector<std::string>>();
    c.ids = j.at("letter_token_ids").get<std::vector<std::int32_t>>();
    require(c.labels.size() == 255 && c.ids.size() == 255, "Expected 255 option labels");
    for (std::size_t i = 0; i < c.ids.size(); ++i)
      require(c.ids[i] >= 0 && std::size_t(c.ids[i]) < tok.vocab_size() && encode(tok, c.labels[i]) == std::vector<std::int32_t>{c.ids[i]}, "qev option tokenizer mismatch");
    c.choice = j.at("temperatures").at("choice"); c.noul = j.at("temperatures").at("noul"); c.score = j.at("temperatures").at("score");
    require(c.choice == 1.7 && c.noul == .9 && c.score == 1.25, "Unsupported qev temperatures");
    return c;
  }
};
struct Request {
  Json state;
  std::vector<Question> questions;
  std::vector<std::vector<std::int32_t>> rows;
  std::string encoding;
  std::size_t input_tokens = 0;
  Json answers = Json::object();
};
inline Request parse(const std::string &body, const std::string &default_encoding) {
  auto j = Json::parse(body); require(j.is_object(), "Expected a JSON object");
  for (const auto &[k, v] : j.items())
    require(k == "model" || k == "state" || k == "questions" || k == "encoding", "Unsupported System One field: " + k);
  if (j.contains("model")) require(j["model"].is_string(), "model must be a string"); // Accepted and ignored, as upstream.
  Request r; r.state = j.at("state");
  require(r.state.is_string() || r.state.is_object() || r.state.is_array(), "state must be a string, object or array");
  require(state_text(r.state).size() <= 65536, "Rendered state exceeds 65536 bytes");
  r.encoding = j.value("encoding", default_encoding);
  require(r.encoding == "joint" || r.encoding == "per_question", "encoding must be joint or per_question");
  const auto &qs = j.at("questions");
  require(qs.is_object() && !qs.empty() && qs.size() <= 32, "questions must be an object with 1..32 questions");
  for (const auto &[id, spec] : qs.items()) r.questions.push_back(question(id, spec));
  return r;
}
inline void prepare(Request &r, const QwenTokenizer &tok, const Config &cfg, std::size_t max_context) {
  auto context = encode(tok, "Context:\n" + state_text(r.state));
  auto joint = context; std::size_t total = 0;
  for (std::size_t i = 0; i < r.questions.size(); ++i) {
    const auto &q = r.questions[i];
    auto ids = r.encoding == "joint" ? joint : context;
    auto append = [&](const std::string &s) { auto piece = encode(tok, s); ids.insert(ids.end(), piece.begin(), piece.end()); };
    const auto index = r.encoding == "joint" && r.questions.size() > 1 ? " " + std::to_string(i + 1) : "";
    std::string head = "\n\nQuestion" + index + ": " + q.instruction + "\nOptions:";
    if (q.options.size() <= 10) {
      for (std::size_t j = 0; j < q.options.size(); ++j) head += "\n(" + cfg.labels[j] + ") " + q.options[j];
      append(head + "\nAnswer" + index + ": (");
    } else {
      append(head);
      for (std::size_t j = 0; j < q.options.size(); ++j) { append("\n("); ids.push_back(cfg.ids[j]); append(") " + q.options[j]); }
      append("\nAnswer" + index + ": (");
    }
    require(!ids.empty() && ids.size() < max_context, "Decision prompt exceeds context; truncation is disabled");
    total += ids.size(); require(total <= 131072, "Decision request exceeds 131072 evaluated input tokens");
    if (r.encoding == "joint") joint = ids;
    r.rows.push_back(std::move(ids));
  }
  // Logical input usage: joint prompt once; per_question includes each row.
  r.input_tokens = r.encoding == "joint" ? joint.size() : total;
}
inline Json answer(const Question &q, const std::vector<float> &logits, const Config &cfg) {
  require(logits.size() == q.names.size(), "Missing decision logits");
  std::vector<double> p; const auto maximum = *std::max_element(logits.begin(), logits.end());
  for (const auto x : logits) { require(std::isfinite(x), "Nonfinite decision logit"); p.push_back(std::exp(double(x) - maximum)); }
  auto normalize = [&] { const double sum = std::accumulate(p.begin(), p.end(), 0.0); for (auto &x : p) x /= sum; };
  normalize(); const double t = q.type == "choice" ? cfg.choice : q.type == "noul" ? cfg.noul : cfg.score;
  for (auto &x : p) x = std::pow(std::max(x, 1e-9), 1 / t);
  normalize(); const auto top = std::max_element(p.begin(), p.end()) - p.begin();
  auto round = [](double x, int places = 4) { double scale = std::pow(10., places); return std::nearbyint(x * scale) / scale; };
  if (q.type == "noul") return {{"type", "noul"}, {"noul", round(p[1])}};
  double entropy = 0, spread = 0, uniform = 0, expected = 0;
  Json probabilities = Json::object(), legend = Json::object();
  for (std::size_t i = 0; i < p.size(); ++i) {
    entropy -= p[i] * std::log(p[i]); spread += p[i] * std::abs(double(i) - top);
    uniform += std::abs(double(i) - (p.size() - 1) / 2.0) / p.size(); expected += i * p[i];
    probabilities[q.names[i]] = round(p[i]); if (q.type == "score") legend[q.names[i]] = q.legend[i];
  }
  const double confidence = q.type == "choice" ? (p.size() * p[top] - 1) / (p.size() - 1) : 1 - spread / uniform;
  Json result = {{"type", q.type}, {"confidence", round(std::clamp(confidence, 0., 1.))}, {"x_p_max", round(p[top])},
    {"certainty", round(std::max(0., 1 - entropy / std::log(p.size())))}, {"probabilities", probabilities}};
  if (q.type == "choice") result["choice"] = q.names[top];
  else { result["score"] = round(expected, 2); result["legend"] = legend; }
  return result;
}
} // namespace qwen35x::systemone
