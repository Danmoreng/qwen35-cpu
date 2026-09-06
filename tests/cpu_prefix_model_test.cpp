#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace qwen35x;
namespace {
bool capture(void *context, std::size_t, std::int32_t, const float *values,
             std::size_t count, std::string &) {
  auto &out = *static_cast<std::vector<float> *>(context);
  out.insert(out.end(), values, values + count);
  return true;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 3 && argc != 4)
    throw std::runtime_error("Usage: cpu_prefix_model_test <HF directory> <Q4 "
                             "artifact> [token CSV]");
  std::string error;
  auto check = [&](bool ok) {
    if (!ok)
      throw std::runtime_error(error);
  };
  auto profile = ProfileLoader::load_from_hf_directory(argv[1], error);
  check(bool(profile));
  CpuLoadOptions options;
  options.model_dir = argv[1];
  options.cpu_q4_h128_path = argv[2];
  auto model = CpuModel::load(*profile, options, error);
  check(bool(model));
  CpuEngineConfig config;
  config.max_context = 8192;
  auto engine = CpuEngine::create(model, config, error);
  check(bool(engine));
  if (argc == 4) {
    std::ifstream input(argv[3]);
    if (!input)
      throw std::runtime_error("Could not open token fixture.");
    std::string text((std::istreambuf_iterator<char>(input)), {});
    std::replace(text.begin(), text.end(), ',', ' ');
    std::istringstream stream(text);
    std::vector<std::int32_t> fixture;
    std::int32_t token;
    while (stream >> token)
      fixture.push_back(token);
    if (fixture.size() < 512)
      throw std::runtime_error("Fixture needs 512 tokens.");
    for (std::size_t length :
         {127U, 128U, 129U, 255U, 511U, 512U, 1023U, 4095U})
      for (std::size_t suffix : {1U, 3U, 129U}) {
        std::vector<std::int32_t> tokens(length + suffix);
        for (std::size_t i = 0; i < tokens.size(); ++i)
          tokens[i] = fixture[i % fixture.size()];
        auto prefix = engine->register_prefix(std::span(tokens).first(length),
                                              "fixture", error);
        check(bool(prefix));
        std::vector<float> cold, warm;
        for (bool restore : {false, true}) {
          CpuRequestSpec request;
          request.prompt_tokens = tokens;
          request.max_new_tokens = 8;
          request.forced_output_tokens = {19, 13, 198, 18, 20, 21, 22, 23};
          request.logits_callback = capture;
          request.logits_callback_context = restore ? &warm : &cold;
          request.trust_namespace = "fixture";
          if (restore)
            request.prefix = prefix;
          auto id = engine->submit(std::move(request), error);
          check(id != 0);
          CpuRequestResult result;
          do {
            check(engine->advance(id, error));
            check(engine->result(id, result));
          } while (result.status != CpuRequestStatus::complete);
          engine->release(id);
        }
        if (cold.size() != warm.size())
          return 12;
        double maximum = 0;
        for (std::size_t i = 0; i < cold.size(); ++i)
          maximum = std::max(maximum, std::abs(double(cold[i]) - warm[i]));
        std::cout << "Realistic prefix=" << length << " suffix=" << suffix
                  << " max_logit_error=" << maximum << std::endl;
        if (std::memcmp(cold.data(), warm.data(), cold.size() * sizeof(float)))
          return 12;
      }
  }
  for (int length : {65, 128, 257}) {
    std::vector<std::int32_t> prefix_tokens(length, 1);
    auto prefix = engine->register_prefix(prefix_tokens, "tests", error);
    check(bool(prefix));
    if (prefix->token_count() != prefix_tokens.size())
      return 2;
    if (engine->register_prefix(prefix_tokens, "tests", error) != prefix)
      return 3;
    for (bool forced : {true, false})
      for (int suffix : {0, 3, 129}) {
        std::vector<float> expected, actual;
        std::vector<std::int32_t> expected_tokens;
        for (bool warm : {false, true}) {
          CpuRequestSpec request;
          request.prompt_tokens = prefix_tokens;
          request.prompt_tokens.insert(request.prompt_tokens.end(), suffix, 2);
          request.max_new_tokens = 8;
          request.sampling.seed = 123 + suffix;
          if (forced)
            request.forced_output_tokens = {19, 13, 198, 18, 20, 21, 22, 23};
          request.logits_callback = capture;
          request.logits_callback_context = warm ? &actual : &expected;
          request.trust_namespace = "tests";
          if (warm)
            request.prefix = prefix;
          auto id = engine->submit(std::move(request), error);
          check(id != 0);
          CpuRequestResult result;
          do {
            check(engine->advance(id, error));
            check(engine->result(id, result));
          } while (result.status != CpuRequestStatus::complete);
          if (result.cached_prefix_tokens != (warm ? length : 0) ||
              result.prefill_tokens !=
                  static_cast<std::size_t>((warm ? 0 : length) + suffix) ||
              result.committed_tokens !=
                  static_cast<std::size_t>(length + suffix + 7))
            return 4;
          if (!warm)
            expected_tokens = result.output_tokens;
          else if (result.output_tokens != expected_tokens)
            return 5;
          engine->release(id);
        }
        if (expected.size() != actual.size())
          return 6;
        double maximum = 0;
        for (std::size_t i = 0; i < expected.size(); ++i)
          maximum =
              std::max(maximum, std::abs(double(expected[i]) - actual[i]));
        std::cout << "Prefix=" << length << " suffix=" << suffix
                  << " forced=" << forced << " max_logit_error=" << maximum
                  << '\n';
        if (std::memcmp(expected.data(), actual.data(),
                        expected.size() * sizeof(float)))
          return 7;
      }
    CpuRequestSpec wrong;
    wrong.prompt_tokens = prefix_tokens;
    wrong.prefix = prefix;
    wrong.trust_namespace = "different";
    if (engine->submit(wrong, error))
      return 8;
    wrong.trust_namespace = "tests";
    wrong.prompt_tokens[0] = 2;
    if (engine->submit(wrong, error))
      return 9;
  }
  std::vector<std::int32_t> tokens(65, 1);
  auto prefix = engine->register_prefix(tokens, "budget", error);
  check(bool(prefix));
  config.max_prefix_bytes = prefix->size_bytes();
  auto bounded = CpuEngine::create(model, config, error);
  check(bool(bounded));
  auto pinned = bounded->register_prefix(tokens, "budget", error);
  check(bool(pinned));
  tokens[0] = 2;
  if (bounded->register_prefix(tokens, "budget", error))
    return 10;
  pinned.reset();
  check(bool(bounded->register_prefix(tokens, "budget", error)));
  auto other_model = CpuModel::load(*profile, options, error);
  check(bool(other_model));
  auto other = CpuEngine::create(other_model, config, error);
  check(bool(other));
  CpuRequestSpec incompatible;
  incompatible.prompt_tokens.assign(65, 1);
  incompatible.prefix = prefix;
  incompatible.trust_namespace = "budget";
  if (other->submit(incompatible, error))
    return 11;
  std::cout << "Complete hybrid prefixes, full hits, identity, namespace and "
               "pinned budgets passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
