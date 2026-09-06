// Fixed-token comparison harness: uses unmodified upstream llama kernels.
// Invoked only through scripts/benchmark-inference-seq.ps1 (cpu-llama-fixed).
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static std::vector<llama_token> tokens(const std::string & csv) {
  std::istringstream stream(csv); std::string item; std::vector<llama_token> result;
  while (std::getline(stream, item, ',')) result.push_back(std::stoi(item));
  return result;
}
int main(int argc, char ** argv) try {
  std::string model_path, profile_path, logits_path;
  std::vector<llama_token> prompt, forced;
  int threads = 8, context = 8192, generated = 128;
  bool prefill_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto value = [&]() -> std::string {
      if (++i >= argc) throw std::runtime_error("Missing argument: " + arg);
      return argv[i];
    };
    if (arg == "--cpu-gguf") model_path = value();
    else if (arg == "--profile-json") profile_path = value();
    else if (arg == "--logits-out") logits_path = value();
    else if (arg == "--prompt-tokens-file") {
      std::ifstream file(value()); std::stringstream text; text << file.rdbuf();
      if(!file) throw std::runtime_error("Could not read prompt tokens");
      prompt=tokens(text.str());
    }
    else if (arg == "--prompt-tokens") prompt = tokens(value());
    else if (arg == "--forced-output-tokens") forced = tokens(value());
    else if (arg == "--cpu-threads") threads = std::stoi(value());
    else if (arg == "--max-context") context = std::stoi(value());
    else if (arg == "--max-new-tokens") generated = std::stoi(value());
    else if (arg == "--prefill-only") prefill_only = true;
    else throw std::runtime_error("Unsupported argument: " + arg);
  }
  if (model_path.empty() || profile_path.empty() || prompt.empty() || threads < 1 ||
      (!prefill_only && (generated < 2 || forced.size() != std::size_t(generated))) ||
      context < int(prompt.size()) + (prefill_only ? 0 : generated))
    throw std::runtime_error("Invalid fixed benchmark configuration");
  ggml_backend_load_all();
  llama_backend_init();
  const auto load_start = Clock::now();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  // CPU-only build and explicit empty device list prevent accelerator offload.
  ggml_backend_dev_t devices[] = {nullptr}; mp.devices = devices;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
    llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
  if (!model) throw std::runtime_error("Model load failed");
  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
  for (const auto & list : {&prompt, &forced}) for (auto token : *list)
    if (token < 0 || token >= vocab) throw std::runtime_error("Token out of range");
  auto cp = llama_context_default_params();
  cp.n_ctx = context; cp.n_batch = 2048; cp.n_ubatch = 512;
  cp.n_threads = threads; cp.n_threads_batch = threads;
  cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
  cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
  cp.offload_kqv = false; cp.op_offload = false;
  std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
    llama_init_from_model(model.get(), cp), llama_free);
  if (!ctx) throw std::runtime_error("Context creation failed");
  const double load_ms = elapsed(load_start);
  std::cout << llama_print_system_info() << '\n';
  llama_batch batch = llama_batch_init(2048, 0, 1);
  struct BatchGuard { llama_batch & b; ~BatchGuard() { llama_batch_free(b); } } guard{batch};
  auto evaluate = [&](const llama_token * data, int count, int position, bool logits) {
    batch.n_tokens = count;
    for (int i = 0; i < count; ++i) {
      batch.token[i] = data[i]; batch.pos[i] = position + i;
      batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
      batch.logits[i] = logits && i + 1 == count;
    }
    if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("llama_decode failed");
    llama_synchronize(ctx.get());
  };
  const auto prefill_start = Clock::now();
  for (int position = 0; position < int(prompt.size());) {
    const int count = std::min(2048, int(prompt.size()) - position);
    evaluate(prompt.data() + position, count, position,
      !prefill_only && position + count == int(prompt.size()));
    position += count;
  }
  const double prefill_ms = elapsed(prefill_start);
  // Optional quality capture only. Its I/O invalidates decode timing; the
  // performance runner never enables it. Format matches qwen35x logit dumps.
  std::ofstream dump;
  if(!logits_path.empty()) {
    if(prefill_only) throw std::runtime_error("Logit capture requires outputs");
    dump.open(logits_path,std::ios::binary);
    const std::uint32_t version=1,vocabulary=vocab;
    const std::uint64_t records=generated;
    dump.write("Q35LGT1\0",8);
    dump.write(reinterpret_cast<const char*>(&version),4);
    dump.write(reinterpret_cast<const char*>(&vocabulary),4);
    dump.write(reinterpret_cast<const char*>(&records),8);
  }
  auto capture=[&](int i) {
    if(logits_path.empty()) return;
    const auto *logits=llama_get_logits_ith(ctx.get(),-1);
    if(!logits) throw std::runtime_error("Missing quality logits");
    for(int v=0;v<vocab;++v) if(!std::isfinite(logits[v])) throw std::runtime_error("Non-finite quality logit");
    dump.write(reinterpret_cast<const char*>(&forced[i]),4);
    dump.write(reinterpret_cast<const char*>(logits),vocab*sizeof(float));
    if(!dump) throw std::runtime_error("Could not write quality logits");
  };
  if(!prefill_only) capture(0);
  const auto decode_start = Clock::now();
  if (!prefill_only) for (int i = 0; i + 1 < generated; ++i) {
    evaluate(forced.data() + i, 1, int(prompt.size()) + i, true);
    capture(i+1);
  }
  const double decode_ms = prefill_only ? 0.0 : elapsed(decode_start);
  // Match engine semantics: first output is predicted during prefill. The
  // summary normalizes throughput by generated-1 actual decode forwards.
  const int outputs = prefill_only ? 0 : generated;
  if (!prefill_only) {
    const float * logits = llama_get_logits_ith(ctx.get(), -1);
    if (!logits) throw std::runtime_error("Missing final logits");
    for (int i = 0; i < vocab; ++i)
      if (!std::isfinite(logits[i])) throw std::runtime_error("Non-finite final logits");
  }
  std::ofstream out(profile_path);
  out << std::setprecision(12) << "{\"prefill_only\":" << (prefill_only ? "true" : "false")
      << ",\"quality_capture\":" << (logits_path.empty() ? "false" : "true")
      << ",\"prompt_tokens\":" << prompt.size() << ",\"generated_tokens\":" << outputs
      << ",\"decode_forward_steps\":" << (prefill_only ? 0 : generated - 1)
      << ",\"load_time_ms\":" << load_ms << ",\"prefill_time_ms\":" << prefill_ms
      << ",\"prefill_tokens_per_second\":" << prompt.size() * 1000.0 / prefill_ms
      << ",\"decode_time_ms\":" << decode_ms
      << ",\"tokens_per_second\":" << (prefill_only ? 0.0 : outputs * 1000.0 / decode_ms)
      << ",\"cpu_kv_cache\":\"fp16\",\"model_parameters\":" << llama_model_n_params(model.get())
      << ",\"model_tensor_bytes\":" << llama_model_size(model.get()) << "}\n";
  if (!out) throw std::runtime_error("Could not write profile");
  return 0;
} catch (const std::exception & e) {
  std::cerr << e.what() << '\n'; return 1;
}
