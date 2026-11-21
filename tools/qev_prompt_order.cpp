// Offline paired quality measurement using the production QEV tokenizer, renderer and readout.
#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "systemone.h"
#include <algorithm>
#include <fstream>
#include <iostream>

using namespace qwen35x;
using systemone::Json;
using Tokens = std::vector<std::int32_t>;

namespace {
void append(Tokens &out, const Tokens &block) {
  out.insert(out.end(), block.begin(), block.end());
}

bool capture(void *opaque, std::size_t, std::int32_t, const float *data,
             std::size_t count, std::string &) {
  static_cast<std::vector<float> *>(opaque)->assign(data, data + count);
  return true;
}

Json score(CpuEngine &engine, const Tokens &tokens, const Tokens &letters,
           const systemone::Question &question, const systemone::Config &cfg,
           const std::shared_ptr<const CpuPrefix> &prefix = {}) {
  std::vector<float> logits;
  CpuRequestSpec spec;
  spec.prompt_tokens = tokens;
  spec.max_new_tokens = 1;
  spec.forced_output_tokens = {letters.front()};
  spec.selected_logit_tokens = letters;
  spec.selected_logits_callback = capture;
  spec.selected_logits_callback_context = &logits;
  spec.trust_namespace = "qev-prompt-order";
  spec.prefix = prefix;
  std::string error;
  auto id = engine.submit(std::move(spec), error);
  if (!id) throw std::runtime_error(error);
  CpuRequestResult result;
  do {
    if (!engine.advance(id, error) || !engine.result(id, result))
      throw std::runtime_error(error);
    if (result.status == CpuRequestStatus::failed || result.status == CpuRequestStatus::cancelled)
      throw std::runtime_error(result.error);
  } while (result.status != CpuRequestStatus::complete);
  engine.release(id);
  if (result.cached_prefix_tokens != (prefix ? prefix->token_count() : 0) ||
      result.prefill_tokens + result.cached_prefix_tokens != tokens.size())
    throw std::runtime_error("Unexpected prefill accounting");
  auto answer = systemone::answer(question, logits, cfg);
  return {{"choice", answer.at("choice")}, {"probabilities", answer.at("probabilities")},
          {"logits", logits}, {"cached_tokens", result.cached_prefix_tokens}};
}
} // namespace

int main(int argc, char **argv) try {
  if (argc != 4) throw std::runtime_error("Usage: qev_prompt_order MODEL_DIR PANEL.json THREADS > rows.jsonl");
  const std::string dir = argv[1];
  std::ifstream input(argv[2]);
  const Json panel = Json::parse(input);
  std::string error;
  QwenTokenizer tok;
  if (!QwenTokenizer::load_from_hf_directory(dir, tok, error)) throw std::runtime_error(error);
  const auto cfg = systemone::Config::load(dir, tok);
  if (!cfg) throw std::runtime_error("Missing QEV config");
  auto profile = ProfileLoader::load_from_hf_directory(dir, error);
  if (!profile) throw std::runtime_error(error);
  CpuLoadOptions load;
  load.model_dir = dir;
  load.cpu_q4_h128_path = dir + "/model.q35h";
  auto model = CpuModel::load(*profile, load, error);
  if (!model) throw std::runtime_error(error);
  CpuEngineConfig settings;
  settings.threads = std::stoi(argv[3]);
  settings.shared_kv_pages = true;
  auto engine = CpuEngine::create(model, settings, error);
  if (!engine) throw std::runtime_error(error);

  Json spec = {{"type", "choice"}, {"instructions", panel.at("instruction")}, {"criteria", panel.at("criteria")}};
  const auto q = systemone::question("dept", spec);
  if (q.options.size() > 10) throw std::runtime_error("Panel expects at most ten options");
  const Tokens letters(cfg->ids.begin(), cfg->ids.begin() + q.options.size());
  std::string rubric = "Question: " + q.instruction + "\nOptions:";
  for (std::size_t i = 0; i < q.options.size(); ++i)
    rubric += "\n(" + cfg->labels[i] + ") " + q.options[i];
  const auto fixed = systemone::encode(tok, rubric);
  const auto gap = systemone::encode(tok, "\n\n");
  const auto slot = systemone::encode(tok, "\nAnswer: (");
  Tokens common = fixed;
  append(common, gap);
  append(common, systemone::encode(tok, "Context:\n"));
  auto prefix = engine->register_prefix(common, "qev-prompt-order", error);
  if (!prefix) throw std::runtime_error(error);

  for (const auto &ticket : panel.at("tickets")) {
    auto request = systemone::parse(Json{{"state", ticket.at("text")}, {"questions", {{"dept", spec}}}}.dump(), "per_question");
    systemone::prepare(request, tok, *cfg, settings.max_context);
    const auto &native = request.rows.at(0);
    const auto context = systemone::encode(tok, "Context:\n" + ticket.at("text").get<std::string>());
    Tokens control = context;
    append(control, gap);
    append(control, fixed);
    append(control, slot);
    if (control != native) throw std::runtime_error("Block tokenization differs from the native QEV prompt");
    Tokens reversed = fixed;
    append(reversed, gap);
    append(reversed, context);
    append(reversed, slot);
    if (reversed.size() < common.size() || !std::equal(common.begin(), common.end(), reversed.begin()))
      throw std::runtime_error("Ticket does not share the registered prefix");
    Json row = {{"id", ticket.at("id")}, {"tokens", native.size()}};
    row["context_first"] = score(*engine, native, letters, q, *cfg);
    row["question_first"] = score(*engine, reversed, letters, q, *cfg);
    row["question_first_cached"] = score(*engine, reversed, letters, q, *cfg, prefix);
    std::cout << row.dump() << std::endl;
  }
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
