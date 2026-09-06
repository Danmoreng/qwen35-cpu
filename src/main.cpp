#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include "qwen35x/tokenizer/tokenizer.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using namespace qwen35x;
namespace {
std::string read(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot open " + path);
  return {std::istreambuf_iterator<char>(f), {}};
}
std::vector<std::int32_t> tokens(std::string text) {
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream in(text);
  std::vector<std::int32_t> result;
  std::int32_t value;
  while (in >> value) result.push_back(value);
  if (!in.eof()) throw std::runtime_error("Invalid token list");
  return result;
}
struct Dump {
  std::ofstream stream;
  std::uint64_t records = 0;
  std::uint32_t vocab = 0;
  static bool record(void *context, std::size_t, std::int32_t target,
                     const float *logits, std::size_t count, std::string &error) {
    auto &d = *static_cast<Dump *>(context);
    if (count != d.vocab) {error = "Vocabulary changed"; return false;}
    d.stream.write(reinterpret_cast<const char *>(&target), sizeof(target));
    d.stream.write(reinterpret_cast<const char *>(logits), count * sizeof(float));
    ++d.records;
    if (!d.stream) {error = "Cannot write logits"; return false;}
    return true;
  }
};
}

int main(int argc, char **argv) try {
  CpuLoadOptions load;
  CpuEngineConfig config;
  config.max_resident_requests = 1;
  CpuRequestSpec request;
  request.sampling.temperature = 0;
  request.sampling.repetition_penalty = 1;
  std::string prompt, tokenize_out, logits_out;
  bool has_prompt = false, has_tokens = false;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "Qwen3.5-0.8B CPU / H128-Q4-G32-DOT4\n"
        "--model-dir DIR --weights FILE --prompt TEXT | --prompt-file FILE | --tokens-file FILE\n"
        "[--threads N] [--max-context N] [--max-new-tokens N]\n"
        "[--tokenize-out FILE] [--forced-tokens-file FILE --logits-out FILE]\n"
        "Raw text completion; supply your own rendered chat template. Greedy decoding.\n";
      return 0;
    }
    if (++i == argc) throw std::runtime_error("Missing value for " + key);
    const std::string value = argv[i];
    auto number = [&]() {
      std::size_t used;
      const auto n = std::stoll(value, &used);
      if (used != value.size() || n <= 0 || n > 100000000)
        throw std::runtime_error("Invalid positive integer for " + key);
      return static_cast<std::size_t>(n);
    };
    if (key == "--model-dir") load.model_dir = value;
    else if (key == "--weights") load.cpu_q4_h128_path = value;
    else if (key == "--prompt" || key == "--prompt-file") {
      if (has_prompt || has_tokens) throw std::runtime_error("Specify one prompt input");
      prompt = key == "--prompt" ? value : read(value); has_prompt = true;
    } else if (key == "--tokens-file") {
      if (has_prompt || has_tokens) throw std::runtime_error("Specify one prompt input");
      request.prompt_tokens = tokens(read(value)); has_tokens = true;
    } else if (key == "--threads") config.threads = static_cast<int>(number());
    else if (key == "--max-context") config.max_context = number();
    else if (key == "--max-new-tokens") request.max_new_tokens = number();
    else if (key == "--tokenize-out") tokenize_out = value;
    else if (key == "--logits-out") logits_out = value;
    else if (key == "--forced-tokens-file") request.forced_output_tokens = tokens(read(value));
    else throw std::runtime_error("Unknown option: " + key);
  }
  std::string error;
  auto check = [&](bool ok) {if (!ok) throw std::runtime_error(error);};
  QwenTokenizer tokenizer;
  check(QwenTokenizer::load_from_hf_directory(load.model_dir, tokenizer, error));
  if (has_prompt) check(tokenizer.encode(prompt, request.prompt_tokens, error));
  if (request.prompt_tokens.empty()) throw std::runtime_error("Prompt must not be empty");
  if (!tokenize_out.empty()) {
    std::ofstream out(tokenize_out);
    for (auto token : request.prompt_tokens) out << token << '\n';
    if (!out) throw std::runtime_error("Cannot write tokens");
    return 0;
  }
  if (!request.forced_output_tokens.empty()) request.max_new_tokens = request.forced_output_tokens.size();
  const auto profile = ProfileLoader::load_from_hf_directory(load.model_dir, error);
  check(bool(profile));
  Dump dump;
  if (!logits_out.empty()) {
    if (request.forced_output_tokens.empty()) throw std::runtime_error("Logit dumps require forced target tokens");
    dump.stream.open(logits_out, std::ios::binary | std::ios::trunc);
    dump.vocab = static_cast<std::uint32_t>(profile->text.vocab_size);
    const std::uint32_t version = 1;
    dump.stream.write("Q35LGT1\0", 8);
    dump.stream.write(reinterpret_cast<const char *>(&version), 4);
    dump.stream.write(reinterpret_cast<const char *>(&dump.vocab), 4);
    dump.stream.write(reinterpret_cast<const char *>(&dump.records), 8);
    if (!dump.stream) throw std::runtime_error("Cannot create logit dump");
    request.logits_callback = Dump::record;
    request.logits_callback_context = &dump;
  } else if (request.forced_output_tokens.empty()) {
    for (const auto *name : {"<|im_end|>", "<|endoftext|>"})
      if (auto id = tokenizer.token_to_id(name)) request.stop_token_ids.push_back(*id);
  }
  auto model = CpuModel::load(*profile, load, error); check(bool(model));
  auto engine = CpuEngine::create(model, config, error); check(bool(engine));
  auto id = engine->submit(std::move(request), error); check(id != 0);
  CpuRequestResult result;
  for (;;) {
    check(engine->result(id, result));
    if (result.status == CpuRequestStatus::complete) break;
    if (result.status == CpuRequestStatus::failed) throw std::runtime_error(result.error);
    check(engine->advance(id, error));
  }
  if (dump.stream.is_open()) {
    dump.stream.seekp(16);
    dump.stream.write(reinterpret_cast<const char *>(&dump.records), 8);
    dump.stream.close();
    if (!dump.stream) throw std::runtime_error("Cannot finish logit dump");
  }
  std::string output;
  check(tokenizer.decode(result.output_tokens, output, error));
  std::cout << output << '\n';
  return 0;
} catch (const std::exception &e) {
  std::cerr << "error: " << e.what() << '\n';
  return 1;
}
