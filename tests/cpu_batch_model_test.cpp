#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

using namespace qwen35x;
namespace {
struct Logits {
  std::vector<float> values;
  std::size_t compared = 0;
};
bool capture(void *context, std::size_t, std::int32_t, const float *values,
             std::size_t count, std::string &error) {
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i])) {
      error = "Non-finite reference logit";
      return false;
    }
  }
  auto &logits = *static_cast<Logits *>(context);
  logits.values.insert(logits.values.end(), values, values + count);
  return true;
}
bool compare(void *context, std::size_t, std::int32_t, const float *values,
             std::size_t count, std::string &error) {
  auto &logits = *static_cast<Logits *>(context);
  if (logits.compared + count > logits.values.size() ||
      std::memcmp(values, logits.values.data() + logits.compared,
                  count * sizeof(float)) != 0) {
    error = "Batched logits differ from single-request logits";
    return false;
  }
  logits.compared += count;
  return true;
}
} // namespace

int main(int argc, char **argv) try {
  if (argc != 3)
    throw std::runtime_error("Usage: cpu_batch_model_test <HF directory> <Q4 artifact>");
  std::string error;
  auto check = [&](bool ok) {
    if (!ok) throw std::runtime_error("Batch model check failed: " + error);
  };
  auto profile = ProfileLoader::load_from_hf_directory(argv[1], error);
  check(bool(profile));
  CpuLoadOptions options;
  options.model_dir = argv[1];
  options.cpu_q4_h128_path = argv[2];
  auto model = CpuModel::load(*profile, options, error);
  check(bool(model));
  CpuEngineConfig config;
  config.threads = 6; // Include uneven row partitions and eight-row tails.
  config.max_resident_requests = 32;
  config.max_context = 256;
  config.batch_greedy = false;
  auto engine = CpuEngine::create(model, config, error);
  check(bool(engine));
  std::vector<Logits> reference(32);
  std::vector<CpuRequestSpec> specs(32);
  const std::size_t lengths[] = {7, 31, 65, 127, 128, 129};
  for (std::size_t r = 0; r < specs.size(); ++r) {
    auto &spec = specs[r];
    spec.prompt_tokens.assign(lengths[r % 6], 1 + static_cast<int>(r));
    spec.forced_output_tokens = {19, 13, 198, 18};
    spec.max_new_tokens = spec.forced_output_tokens.size();
    spec.logits_callback = capture;
    spec.logits_callback_context = &reference[r];
    const auto id = engine->submit(spec, error);
    check(id != 0);
    CpuRequestResult result;
    do {
      check(engine->advance(id, error));
      check(engine->result(id, result));
      check(result.status != CpuRequestStatus::failed);
    } while (result.status != CpuRequestStatus::complete);
    check(reference[r].values.size() == spec.max_new_tokens * profile->text.vocab_size);
    check(engine->release(id));
    spec.logits_callback = compare;
  }
  for (std::size_t count : {4U, 8U, 12U, 16U, 20U, 32U}) {
    std::vector<CpuRequestId> ids;
    for (std::size_t r = 0; r < count; ++r) {
      reference[r].compared = 0;
      const auto id = engine->submit(specs[r], error);
      check(id != 0);
      ids.push_back(id);
      CpuRequestResult result;
      do {
        check(engine->advance(id, error));
        check(engine->result(id, result));
      } while (result.status == CpuRequestStatus::prefill);
      check(result.status == CpuRequestStatus::decode);
    }
    for (std::size_t token = 1; token < specs[0].max_new_tokens; ++token)
      check(engine->advance_batch(ids, error));
    for (std::size_t r = 0; r < count; ++r) {
      CpuRequestResult result;
      check(engine->result(ids[r], result));
      check(result.status == CpuRequestStatus::complete);
      check(reference[r].compared == reference[r].values.size());
      check(engine->release(ids[r]));
    }
    std::cout << "B=" << count << ": full-vocabulary logits match bitwise\n";
  }
  return 0;
} catch (const std::exception &error) {
  std::cerr << error.what() << '\n';
  return 1;
}
