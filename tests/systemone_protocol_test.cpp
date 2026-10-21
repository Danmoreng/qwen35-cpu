#include "systemone.h"
#include <iostream>
using namespace qwen35x;
using namespace qwen35x::systemone;
int main(int argc, char **argv) try {
  if (argc == 1) {
    auto q = question("z", {{"type", "choice"}, {"instructions", "Pick"}, {"criteria", {{"z", nullptr}, {"a", "A"}}}});
    require(q.names == std::vector<std::string>{"z", "a"}, "Option order changed");
    auto result = answer(q, {0, 0}, Config{});
    require(result["choice"] == "z" && result["confidence"] == 0 && result["certainty"] == 0, "Uniform choice or tie handling failed");
    auto score = question("s", {{"type", "score"}, {"instructions", "Rate"}, {"criteria", {"bad", "okay", "good"}}});
    result = answer(score, {0, 0, 0}, Config{});
    require(result["score"] == 1.0 && result["legend"]["2"] == "good", "Score readout failed");
    auto yes = question("b", {{"type", "bool"}, {"criteria", {{"true", "allowed"}}}});
    require(yes.instruction == "Which answer fits the context?" && yes.options[1] == "yes: allowed", "Implicit noul question failed");
    Json array = Json::array(); for (int i = 0; i < 8; ++i) array.push_back(i);
    require(annotate(array)[7] == Json({{"_index", 7}, {"value", 7}}), "Array annotation failed");
    require(json_text(Json{{"text", "a,b:c\\\""}, {"array", {1, 2}}}) == "{\"text\": \"a,b:c\\\\\\\"\", \"array\": [1, 2]}", "JSON whitespace/escaping failed");
    bool rejected = false; try { question("bad", {{"type", "noul"}}); } catch (...) { rejected = true; }
    require(rejected, "Invalid noul question was accepted");
    std::cout << "System One protocol checks passed\n"; return 0;
  }
  require(argc == 2, "Usage: systemone_protocol_test [MODEL_DIR]");
  QwenTokenizer tok; std::string error;
  require(QwenTokenizer::load_from_hf_directory(argv[1], tok, error), error);
  auto cfg = Config::load(argv[1], tok); require(cfg.has_value(), "Missing qev configuration");
  std::string line;
  while (std::getline(std::cin, line)) {
    auto r = parse(line, "joint"); prepare(r, tok, *cfg, 32768);
    for (const auto &q : r.questions) {
      std::vector<float> logits;
      for (std::size_t i = 0; i < q.names.size(); ++i) logits.push_back(float(int(i * 7 % 17) - 8) * .125f);
      r.answers[q.id] = answer(q, logits, *cfg);
    }
    std::cout << Json{{"rows", r.rows}, {"answers", r.answers}, {"state", state_text(r.state)}}.dump() << '\n';
  }
  return 0;
} catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
