#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace qwen35x;
namespace {
bool capture(void *context, std::size_t, std::int32_t, const float *values,
             std::size_t count, std::string &error) {
  if (!std::all_of(values, values + count, [](float x) { return std::isfinite(x); })) {
    error = "Non-finite logit";
    return false;
  }
  static_cast<std::vector<float> *>(context)->assign(values, values + count);
  return true;
}
std::vector<float> evaluate(CpuEngine &engine,
    const std::vector<std::int32_t> &tokens,
    std::shared_ptr<const CpuPrefix> prefix = {}) {
  std::vector<float> logits;
  CpuRequestSpec spec;
  spec.prompt_tokens = tokens;
  spec.max_new_tokens = 1;
  spec.forced_output_tokens = {19};
  spec.logits_callback = capture;
  spec.logits_callback_context = &logits;
  spec.trust_namespace = "chunk-regression";
  spec.prefix = std::move(prefix);
  std::string error;
  auto id = engine.submit(std::move(spec), error);
  if (!id) throw std::runtime_error(error);
  CpuRequestResult result;
  for (;;) {
    if (!engine.result(id, result)) throw std::runtime_error("Missing result");
    if (result.status == CpuRequestStatus::complete) break;
    if (result.status == CpuRequestStatus::failed) throw std::runtime_error(result.error);
    if (!engine.advance(id, error)) throw std::runtime_error(error);
  }
  if (!engine.release(id)) throw std::runtime_error("Could not release request");
  return logits;
}
void equal(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.empty() || a.size() != b.size() ||
      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)))
    throw std::runtime_error("Large-chunk/prefix logits differ from chunk 128");
}
}

int main(int argc, char **argv) try {
  if (argc != 3 && argc != 4)
    throw std::runtime_error("Usage: cpu_prefill_chunk_model_test MODEL_DIR WEIGHTS [--pages]");
  const bool pages = argc == 4;
  if (pages && std::string(argv[3]) != "--pages") throw std::runtime_error("Unknown option");
  std::string error;
  auto profile = ProfileLoader::load_from_hf_directory(argv[1], error);
  if (!profile) throw std::runtime_error(error);
  CpuLoadOptions load;
  load.model_dir = argv[1];
  load.cpu_q4_h128_path = argv[2];
  auto model = CpuModel::load(*profile, load, error);
  if (!model) throw std::runtime_error(error);
  CpuEngineConfig config;
  config.threads = 2;
  config.executor_spin_count = 0;
  config.max_context = 1024;
  config.shared_kv_pages = pages;
  config.prefill_chunk_size = 128;
  auto reference = CpuEngine::create(model, config, error);
  if (!reference) throw std::runtime_error(error);
  std::vector<std::int32_t> tokens(579);
  for (std::size_t i = 0; i < tokens.size(); ++i) tokens[i] = 1 + (i * 17) % 1024;
  const auto expected = evaluate(*reference, tokens);
  for (std::size_t chunk : {512U, 1024U}) {
    config.prefill_chunk_size = chunk;
    auto candidate = CpuEngine::create(model, config, error);
    if (!candidate) throw std::runtime_error(error);
    equal(expected, evaluate(*candidate, tokens));
    for (std::size_t length : {127U, 221U, 511U}) {
      auto prefix = candidate->register_prefix(std::span(tokens).first(length),
                                               "chunk-regression", error);
      if (!prefix) throw std::runtime_error(error);
      equal(expected, evaluate(*candidate, tokens, prefix));
    }
    std::cout << "Chunk " << chunk << ": cold and prefix-restored logits match bitwise\n";
  }
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
