#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace qwen35x;
namespace {
bool capture(void *context, std::size_t, std::int32_t, const float *v,
             std::size_t n, std::string &) {
  auto &out = *static_cast<std::vector<float> *>(context);
  out.insert(out.end(), v, v + n);
  return true;
}
bool terminal(CpuRequestStatus status) {
  return status == CpuRequestStatus::complete ||
         status == CpuRequestStatus::cancelled ||
         status == CpuRequestStatus::failed;
}
} // namespace
int main(int argc, char **argv) try {
  if (argc != 3 && argc != 4)
    throw std::runtime_error(
        "Usage: cpu_scheduler_model_test <HF directory> <Q4 artifact>");
  std::string error;
  auto check = [&](bool ok) {
    if (!ok)
      throw std::runtime_error("Scheduler check failed: " + error);
  };
  auto profile = ProfileLoader::load_from_hf_directory(argv[1], error);
  check(bool(profile));
  CpuLoadOptions options;
  options.model_dir = argv[1];
  options.cpu_q4_h128_path = argv[2];
  auto model = CpuModel::load(*profile, options, error);
  check(bool(model));
  CpuEngineConfig config;
  config.threads = 2;
  config.max_resident_requests = 2;
  config.max_context = 2048;
  auto reference = CpuEngine::create(model, config, error);
  check(bool(reference));
  if(argc==4) {
    check(std::string(argv[3])=="--pages");
    config.shared_kv_pages=true;
  }
  config.max_queued_requests = 8;
  config.max_decode_batch_size = 2;
  config.max_buffered_tokens = 2;
  auto engine = CpuEngine::create(model, config, error);
  check(bool(engine));
  constexpr std::size_t count = 4;
  std::vector<float> expected[count], actual[count];
  std::vector<std::int32_t> delivered[count];
  CpuRequestSpec specs[count];
  CpuRequestId ids[count]{};
  const std::size_t lengths[count] = {65, 257, 129, 1024};
  for (std::size_t r = 0; r < count; ++r) {
    auto &s = specs[r];
    s.prompt_tokens.assign(lengths[r], 1 + static_cast<int>(r));
    s.max_new_tokens = 8;
    s.forced_output_tokens = {19, 13, 198, 18, 20, 21, 22, 23};
    s.logits_callback = capture;
    s.logits_callback_context = &expected[r];
    auto id = reference->submit(s, error);
    check(id != 0);
    CpuRequestResult result;
    do {
      check(reference->advance(id, error));
      check(reference->result(id, result));
    } while (!terminal(result.status));
    check(result.status == CpuRequestStatus::complete);
    reference->release(id);
    s.logits_callback_context = &actual[r];
    if (r < 3) {
      ids[r] = engine->submit(s, error);
      check(ids[r] != 0);
    }
  }
  check(engine->stats().resident_requests == 0 &&
        engine->stats().queued_requests == 3);
  check(!engine->advance(ids[0], error));
  std::size_t tick = 0;
  bool other_progress_while_blocked = false;
  for (; tick < 1000; ++tick) {
    if (tick == 3) {
      ids[3] = engine->submit(specs[3], error);
      check(ids[3] != 0);
    }
    bool progressed = false;
    check(engine->step(progressed, error));
    if (tick < 20 && actual[1].size() > 0)
      other_progress_while_blocked = true;
    bool done = tick >= 3;
    for (std::size_t r = 0; r < count; ++r)
      if (ids[r]) {
        if (r != 0 || tick >= 20)
          check(engine->read_output(ids[r], 1, delivered[r], error));
        CpuRequestResult result;
        check(engine->result(ids[r], result));
        done &= terminal(result.status) && delivered[r].size() == 8;
      }
    check(engine->stats().resident_requests <= 2);
    if (done)
      break;
  }
  check(tick < 1000 && other_progress_while_blocked);
  for (std::size_t r = 0; r < count; ++r) {
    check(delivered[r] == specs[r].forced_output_tokens);
    check(expected[r].size() == actual[r].size());
    check(std::memcmp(expected[r].data(), actual[r].data(),
                      expected[r].size() * sizeof(float)) == 0);
    engine->release(ids[r]);
  }
  check(engine->stats().resident_requests == 0);

  // Withhold possible stop prefixes even when a consumer drains every tick.
  CpuRequestSpec stop;
  stop.prompt_tokens = {1, 2, 3};
  stop.max_new_tokens = 6;
  stop.forced_output_tokens = {2, 3, 4, 5, 6, 7};
  stop.stop_token_sequences = {{3, 4}};
  auto id = engine->submit(stop, error);
  check(id != 0);
  std::vector<std::int32_t> output;
  CpuRequestResult result;
  for (int i = 0; i < 20; ++i) {
    bool progressed;
    check(engine->step(progressed, error));
    check(engine->read_output(id, 2, output, error));
    check(engine->result(id, result));
    if (terminal(result.status))
      break;
  }
  check(result.status == CpuRequestStatus::complete &&
        output == std::vector<std::int32_t>{2});
  engine->release(id);
  stop.stop_token_sequences = {{3, 4, 5}};
  check(engine->submit(stop, error) == 0);

  // Queue capacity, cancellation before/after admission, and idle backpressure.
  config.max_resident_requests = 1;
  config.max_queued_requests = 1;
  auto bounded = CpuEngine::create(model, config, error);
  check(bool(bounded));
  stop.stop_token_sequences.clear();
  auto a = bounded->submit(stop, error);
  check(a != 0);
  check(bounded->submit(stop, error) == 0);
  bool progressed;
  check(bounded->step(progressed, error));
  auto b = bounded->submit(stop, error);
  check(b != 0);
  check(bounded->submit(stop, error) == 0);
  check(bounded->cancel(b));
  check(bounded->result(b, result));
  check(result.status == CpuRequestStatus::cancelled);
  bounded->release(b);
  b = bounded->submit(stop, error);
  check(b != 0);
  check(bounded->step(progressed, error));
  check(bounded->step(progressed, error));
  check(!progressed);
  check(bounded->cancel(a));
  check(bounded->stats().resident_requests == 0);
  check(bounded->step(progressed, error));
  check(progressed);
  check(bounded->result(b, result));
  check(result.status == CpuRequestStatus::decode);
  // Single-flight followers wait without state; cancellation reelects a leader.
  for (int scenario = 0; scenario < 3; ++scenario) {
    config.max_resident_requests = 3;
    config.max_queued_requests = 8;
    config.max_buffered_tokens = 16;
    config.max_prefix_bytes = scenario == 2 ? 1 : 256 * 1024 * 1024;
    auto shared = CpuEngine::create(model, config, error);
    check(bool(shared));
    CpuRequestId branches[3];
    std::vector<float> cold[3], warm[3];
    for (int r = 0; r < 3; ++r) {
      CpuRequestSpec request;
      request.prompt_tokens.assign(257, 1);
      request.prompt_tokens.push_back(2 + r);
      request.max_new_tokens = 4;
      request.forced_output_tokens = {19, 13, 198, 18};
      request.logits_callback = capture;
      request.logits_callback_context = &cold[r];
      auto reference_id = reference->submit(request, error);
      check(reference_id != 0);
      do {
        check(reference->advance(reference_id, error));
        check(reference->result(reference_id, result));
      } while (!terminal(result.status));
      reference->release(reference_id);
      request.logits_callback_context = &warm[r];
      request.register_prefix_tokens = 257;
      request.trust_namespace = "single-flight";
      branches[r] = shared->submit(request, error);
      check(branches[r] != 0);
    }
    check(shared->step(progressed, error));
    check(shared->stats().resident_requests == 1 &&
          shared->stats().queued_requests == 2);
    if (scenario == 1)
      check(shared->cancel(branches[0]));
    bool done = false;
    for (int i = 0; i < 100 && !done; ++i) {
      check(shared->step(progressed, error));
      done = true;
      for (auto branch : branches) {
        check(shared->result(branch, result));
        done &= terminal(result.status);
      }
    }
    check(done);
    std::size_t cached = 0;
    for (int r = 0; r < 3; ++r) {
      check(shared->result(branches[r], result));
      cached += result.cached_prefix_tokens;
      if (scenario == 1 && r == 0)
        continue;
      check(result.status == CpuRequestStatus::complete);
      check(cold[r].size() == warm[r].size());
      check(std::memcmp(cold[r].data(), warm[r].data(),
                        cold[r].size() * sizeof(float)) == 0);
    }
    check(shared->stats().prefix_builds == (scenario == 2 ? 0U : 1U));
    check(shared->stats().prefix_build_fallbacks == (scenario == 2 ? 1U : 0U));
    check(cached == (scenario == 0 ? 514U : scenario == 1 ? 257U : 0U));
  }
  std::cout << "Single-flight builds, leader cancellation, budget fallback and "
               "exact branch logits passed\n";
  std::cout
      << "Serving queues, delayed arrivals, private logits, fair progress, "
         "backpressure, stop withholding and cancellation passed\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
