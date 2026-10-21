// SPDX-License-Identifier: MIT
// Thin benchmark binding. All inference remains in unmodified upstream llama.cpp.
#include "llama.h"
#include <nlohmann/json.hpp>
#include <emscripten/heap.h>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>
using Json = nlohmann::ordered_json;
using Clock = std::chrono::steady_clock;
namespace {
llama_model *model = nullptr;
llama_context *context = nullptr;
std::string output;
const char *reply(const Json &value) { output = value.dump(); return output.c_str(); }
void unload() {
  if (context) llama_free(context);
  if (model) llama_model_free(model);
  context = nullptr; model = nullptr;
}
}
extern "C" {
const char *bench_unload() { unload(); return reply({{"status", "unloaded"}}); }
const char *bench_load(const char *path, int threads, int ubatch, int flash) try {
  if (threads < 1 || threads > 8 || ubatch < 1 || ubatch > 2048) throw std::runtime_error("Invalid configuration");
  unload(); llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0; mp.load_mode = LLAMA_LOAD_MODE_NONE;
  model = llama_model_load_from_file(path, mp);
  if (!model) throw std::runtime_error("Cannot load GGUF");
  auto cp = llama_context_default_params();
  cp.n_ctx = 2048; cp.n_batch = 2048; cp.n_ubatch = ubatch; cp.n_seq_max = 1;
  cp.n_threads = cp.n_threads_batch = threads;
  cp.offload_kqv = false; cp.op_offload = false;
  cp.flash_attn_type = static_cast<llama_flash_attn_type>(flash);
  context = llama_init_from_model(model, cp);
  if (!context) throw std::runtime_error("Cannot create context");
  return reply({{"status", "ready"}, {"threads", threads}, {"ubatch", ubatch}, {"flash_attn", flash},
    {"wasm_heap_bytes", emscripten_get_heap_size()}, {"system_info", llama_print_system_info()}});
} catch (const std::exception &e) { unload(); return reply({{"error", e.what()}}); }
const char *bench_score(const char *body) try {
  if (!context) throw std::runtime_error("Load the model first");
  const auto request = Json::parse(body);
  auto tokens = request.at("tokens").get<std::vector<llama_token>>();
  const auto ids = request.at("letter_ids").get<std::vector<llama_token>>();
  if (tokens.empty() || tokens.size() > 2048) throw std::runtime_error("Invalid prompt length");
  const auto *vocab = llama_model_get_vocab(model);
  for (auto id : tokens) if (id < 0 || id >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("Invalid token");
  for (auto id : ids) if (id < 0 || id >= llama_vocab_n_tokens(vocab)) throw std::runtime_error("Invalid label");
  const auto started = Clock::now();
  llama_memory_clear(llama_get_memory(context), true);
  const auto prefill = Clock::now();
  auto batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
  if (llama_decode(context, batch) != 0) throw std::runtime_error("llama_decode failed");
  const auto prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - prefill).count();
  const auto *values = llama_get_logits_ith(context, -1);
  if (!values) throw std::runtime_error("Missing logits");
  std::vector<float> logits;
  for (auto id : ids) logits.push_back(values[id]);
  return reply({{"logits", logits}, {"prefill_ms", prefill_ms},
    {"execution_ms", std::chrono::duration<double, std::milli>(Clock::now() - started).count()},
    {"evaluated_tokens", tokens.size()}, {"output_tokens", 0}, {"decode_forwards", 0},
    {"wasm_heap_bytes", emscripten_get_heap_size()}});
} catch (const std::exception &e) { return reply({{"error", e.what()}}); }
}
