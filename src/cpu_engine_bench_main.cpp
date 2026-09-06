#include "qwen35x/compiler/compiler.h"
#include "qwen35x/runtime/cpu_engine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>

#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using namespace qwen35x;
namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
std::vector<std::int32_t> tokens(std::string text) {
  std::replace(text.begin(), text.end(), ',', ' ');
  std::istringstream input(text);
  std::vector<std::int32_t> out;
  std::int32_t token;
  while (input >> token)
    out.push_back(token);
  if (!input.eof())
    throw std::runtime_error("Invalid token list");
  return out;
}
std::uint64_t peak_rss() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS info{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info)))
    return 0;
  return info.PeakWorkingSetSize;
#else
  rusage usage{};
  return getrusage(RUSAGE_SELF, &usage) == 0
             ? std::uint64_t(usage.ru_maxrss) * 1024
             : 0;
#endif
}
} // namespace

#include "cpu_serving_bench.inl"

int main(int argc, char **argv) try {
  CpuLoadOptions options;
  CpuEngineConfig config;
  CpuRequestSpec spec;
  std::size_t batch_size = 1;
  std::size_t prefix_tokens = 0;
  bool identical_prompts = false;
  std::string profile_path;
  std::string operation_profile_path;
  std::vector<CpuDecodeStage> operation_profile;
  bool serial = false, strict = false;
  bool serving = false;
  std::size_t residents = 16;
  double arrival_gap_ms = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--infer-reference")
      continue;
    if (arg == "--cpu-serving") { serving = true; continue; }
    if (arg == "--cpu-shared-kv-pages") { config.shared_kv_pages = true; continue; }
    if (arg == "--cpu-identical-prompts") {
      identical_prompts = true;
      continue;
    }
    if (arg == "--cpu-batch-greedy") {
      config.batch_greedy = true;
      continue;
    }
    if (arg == "--cpu-batch-full-logits") {
      config.batch_greedy = false;
      continue;
    }
    if (arg == "--cpu-batch-parallel-state") {
      config.parallel_state_batches = true;
      continue;
    }
    if (arg == "--cpu-batch-serial-state") {
      config.parallel_state_batches = false;
      continue;
    }
    if (arg == "--cpu-batch-serial") {
      serial = true;
      continue;
    }
    if (arg == "--cpu-isa-strict") {
      strict = true;
      continue;
    }
    if (i + 1 == argc)
      throw std::runtime_error("Missing value for " + arg);
    const std::string value = argv[++i];
    auto positive = [&]() {
      std::size_t used;
      auto n = std::stoll(value, &used);
      if (n <= 0 || used != value.size())
        throw std::runtime_error("Invalid positive integer: " + value);
      return static_cast<std::size_t>(n);
    };
    if (arg == "--hf-model-dir")
      options.model_dir = value;
    else if ((arg == "--cpu-q4-h128" || arg == "--cpu-gguf"))
      options.cpu_q4_h128_path = value;
    else if (arg == "--cpu-batch")
      batch_size = positive();
    else if (arg == "--cpu-residents") residents = positive();
    else if (arg == "--cpu-decode-batch-limit") config.max_decode_batch_size = positive();
    else if (arg == "--cpu-arrival-gap-ms") {
      arrival_gap_ms = std::stod(value);
      if (!std::isfinite(arrival_gap_ms) || arrival_gap_ms < 0) throw std::runtime_error("Invalid arrival gap");
    }
    else if (arg == "--cpu-registered-prefix")
      prefix_tokens = positive();
    else if (arg == "--cpu-single-flight-prefix")
      spec.register_prefix_tokens = positive();
    else if (arg == "--max-context")
      config.max_context = positive();
    else if (arg == "--max-new-tokens")
      spec.max_new_tokens = positive();
    else if (arg == "--cpu-threads")
      config.threads = std::stoi(value);
    else if (arg == "--temperature")
      spec.sampling.temperature = std::stof(value);
    else if (arg == "--top-p")
      spec.sampling.top_p = std::stof(value);
    else if (arg == "--top-k")
      spec.sampling.top_k = std::stoi(value);
    else if (arg == "--repeat-penalty")
      spec.sampling.repetition_penalty = std::stof(value);
    else if (arg == "--seed")
      spec.sampling.seed = std::stoll(value);
    else if (arg == "--profile-json")
      profile_path = value;
    else if (arg == "--operation-profile")
      operation_profile_path = value;
    else if (arg == "--prompt-tokens")
      spec.prompt_tokens = tokens(value);
    else if (arg == "--forced-output-tokens")
      spec.forced_output_tokens = tokens(value);
    else if (arg == "--prompt-tokens-file") {
      std::ifstream file(value);
      std::ostringstream text;
      text << file.rdbuf();
      if (!file)
        throw std::runtime_error("Cannot read token file");
      spec.prompt_tokens = tokens(text.str());
    } else if (arg == "--cpu-isa") {
      bool found = false;
      const char *names[] = {"auto",     "scalar", "avx2",
                             "avx-vnni", "avx512", "avx512-vnni"};
      for (auto backend :
           {cpu::Q8_0Backend::auto_select, cpu::Q8_0Backend::scalar,
            cpu::Q8_0Backend::avx2, cpu::Q8_0Backend::avx_vnni,
            cpu::Q8_0Backend::avx512, cpu::Q8_0Backend::avx512_vnni})
        if (value == names[static_cast<unsigned>(backend)]) {
          options.cpu_q8_backend = backend;
          found = true;
        }
      if (!found)
        throw std::runtime_error("Unknown CPU ISA");
    } else
      throw std::runtime_error("Unsupported benchmark option: " + arg);
  }
  if (batch_size > 100 || spec.prompt_tokens.empty() || profile_path.empty())
    throw std::runtime_error(
        "Require explicit prompt tokens, profile JSON, B=1..100");
  if (strict && !cpu::q8_0_backend_available(options.cpu_q8_backend))
    throw std::runtime_error("Requested CPU ISA unavailable");
  if (!operation_profile_path.empty()) {
    if (batch_size != 1 || serving)
      throw std::runtime_error("Operation profiling requires a single non-serving request");
    config.operation_profile = &operation_profile;
  }
  std::string error;
  auto check = [&](bool ok) {
    if (!ok)
      throw std::runtime_error(error);
  };
  const auto load_start = Clock::now();
  auto profile =
      ProfileLoader::load_from_hf_directory(options.model_dir, error);
  check(bool(profile));
  auto model = CpuModel::load(*profile, options, error);
  check(bool(model));
  config.max_resident_requests = serving ? residents : batch_size;
  if (serving) {
    if (serial) throw std::runtime_error("Use decode batch limit 1 for serial serving");
    config.max_queued_requests = batch_size;
  }
  auto engine = CpuEngine::create(model, config, error);
  check(bool(engine));
  const double load_ms = ms(load_start);
  if (prefix_tokens > spec.prompt_tokens.size() ||
      (prefix_tokens == spec.prompt_tokens.size() && !identical_prompts))
    throw std::runtime_error(
        "Invalid prefix length; full hits require identical prompts");
  std::shared_ptr<const CpuPrefix> prefix;
  double prefix_build_ms = 0;
  if (prefix_tokens) {
    const auto start = Clock::now();
    prefix = engine->register_prefix(
        std::span<const std::int32_t>(spec.prompt_tokens.data(), prefix_tokens),
        "benchmark", error);
    check(bool(prefix));
    prefix_build_ms = ms(start);
  }
  if (serving) return serving_benchmark(*engine, spec, prefix, batch_size,
      profile->text.vocab_size, identical_prompts, arrival_gap_ms, load_ms,
      prefix_build_ms, profile_path);
  std::vector<CpuRequestId> ids;
  const auto prefill_start = Clock::now();
  for (std::size_t r = 0; r < batch_size; ++r) {
    auto request = spec;
    // Same prefix, deterministic distinct final prompt token; no prefix cache.
    request.prefix = prefix;
    request.trust_namespace = "benchmark";
    if (!identical_prompts)
      request.prompt_tokens.back() =
          (request.prompt_tokens.back() + static_cast<int>(r)) %
          profile->text.vocab_size;
    auto id = engine->submit(std::move(request), error);
    check(id != 0);
    ids.push_back(id);
    CpuRequestResult state;
    do {
      check(engine->advance(id, error));
      check(engine->result(id, state));
    } while (state.status == CpuRequestStatus::prefill);
  }
  const double prefill_ms = ms(prefill_start);
  std::vector<double> ticks;
  auto active = ids;
  if (spec.max_new_tokens == 1)
    active.clear();
  const auto decode_start = Clock::now();
  while (!active.empty()) {
    const auto tick_start = Clock::now();
    if (serial) {
      for (auto id : active)
        check(engine->advance(id, error));
    } else
      check(engine->advance_batch(active, error));
    std::erase_if(active, [&](auto id) {
      CpuRequestResult state;
      check(engine->result(id, state));
      return state.status == CpuRequestStatus::complete;
    });
    ticks.push_back(ms(tick_start));
  }
  const double decode_ms = ms(decode_start);
  const auto stats = engine->stats();
  std::size_t forwards = 0, delivered = 0, newly_prefilled = 0, cached = 0;
  double restore_ms = 0;
  double prefill_forward_ms = 0;
  std::vector<CpuRequestResult> results(batch_size);
  for (std::size_t r = 0; r < ids.size(); ++r) {
    check(engine->result(ids[r], results[r]));
    forwards += results[r].decode_forwards;
    delivered += results[r].output_tokens.size();
    newly_prefilled += results[r].prefill_tokens;
    cached += results[r].cached_prefix_tokens;
    restore_ms += results[r].prefix_restore_ms;
    prefill_forward_ms += results[r].prefill_ms;
  }
  std::sort(ticks.begin(), ticks.end());
  auto percentile = [&](double p) {
    return ticks.empty()
               ? 0.0
               : ticks[static_cast<std::size_t>(p * (ticks.size() - 1))];
  };
  const double throughput = forwards ? forwards * 1000.0 / decode_ms : 0.0;
  std::ofstream out(profile_path);
  if (!out)
    throw std::runtime_error("Cannot write benchmark profile");
  out << std::setprecision(12) << "{\n"
      << "\"operation_instrumentation\":" << (config.operation_profile ? "true" : "false") << ","
      << "\"weight_format\":\"" << model->weight_format() << "\",\"prefill_only\":false,\"cpu_batch\":" << batch_size
      << ",\"cpu_batch_serial\":" << (serial ? "true" : "false")
      << ",\"prompt_tokens\":" << spec.prompt_tokens.size() * batch_size
      << ",\"generated_tokens\":" << delivered
      << ",\"decode_forwards\":" << forwards << ",\"load_time_ms\":" << load_ms
      << ",\"prefill_time_ms\":" << prefill_ms
      << ",\"prefill_forward_time_ms\":" << prefill_forward_ms
      << ",\"prefill_forward_tokens_per_second\":" << newly_prefilled * 1000.0 / prefill_forward_ms
      << ",\"resolved_isa\":\"" << cpu::q8_0_backend_name(cpu::q8_0_resolve_backend(options.cpu_q8_backend)) << "\""
      << ",\"projection_kernel\":\"" << (std::string(model->weight_format())=="gguf-k-quants" ?
           (cpu::q8_0_backend_uses_avx2(options.cpu_q8_backend)?"k-quants-avx2":"k-quants-scalar") :
           (std::string(model->weight_format()).find("q8-") != std::string::npos ? "q4-dot4+q8_0" : "q4-dot4")) << "\""
      << ",\"prefill_tokens_per_second\":"
      << newly_prefilled * 1000.0 / prefill_ms
      << ",\"new_prefill_tokens\":" << newly_prefilled
      << ",\"cached_prefix_tokens\":" << cached
      << ",\"prefix_build_time_ms\":" << prefix_build_ms
      << ",\"prefix_cache_restore_time_ms\":" << restore_ms
      << ",\"prefix_cache_bytes\":" << (prefix ? prefix->size_bytes() : 0)
      << ",\"decode_time_ms\":" << decode_ms
      << ",\"tokens_per_second\":" << throughput
      << ",\"per_request_tokens_per_second\":" << throughput / batch_size
      << ",\"tick_p50_ms\":" << percentile(.50)
      << ",\"tick_p95_ms\":" << percentile(.95)
      << ",\"tick_p99_ms\":" << percentile(.99)
      << ",\"peak_rss_bytes\":" << peak_rss()
      << ",\"physical_kv_bytes\":" << stats.physical_kv_bytes
      << ",\"batch_ticks\":" << stats.batch_ticks
      << ",\"batch_rows\":" << stats.batch_rows
      << ",\"projection_batches\":" << stats.projection_batches
      << ",\"lm_head_batches\":" << stats.lm_head_batches
      << ",\"greedy_lm_head_batches\":" << stats.greedy_lm_head_batches
      << ",\"state_batches\":" << stats.state_batches << ",\"cpu_kv_cache\":\""
      << (cpu::q8_0_backend_uses_avx2(options.cpu_q8_backend) ? "fp16" : "fp32")
      << "\",\"requests\":[";
  for (std::size_t r = 0; r < results.size(); ++r) {
    if (r)
      out << ',';
    out << "[";
    for (std::size_t t = 0; t < results[r].output_tokens.size(); ++t) {
      if (t)
        out << ',';
      out << results[r].output_tokens[t];
    }
    out << "]";
  }
  out << "]}\n";
  if (!out)
    throw std::runtime_error("Failed writing profile");
  if (!operation_profile_path.empty()) {
    std::ofstream stages(operation_profile_path);
    stages << "kind,rows,columns,vectors,prepare_ms,wall_ms,dispatch_ms,caller_ms,wait_ms,participants\n";
    for (const auto & event : operation_profile)
      stages << event.kind << ',' << event.rows << ',' << event.columns << ',' << event.vectors << ','
             << event.prepare_ms << ',' << event.wall_ms << ',' << event.dispatch_ms << ','
             << event.caller_ms << ',' << event.wait_ms << ',' << event.participants << '\n';
    if (!stages) throw std::runtime_error("Failed writing operation profile");
  }
  std::cout << "B=" << batch_size << " aggregate_decode=" << throughput
            << " token/s tick_p50=" << percentile(.50) << " ms\n";
  return 0;
} catch (const std::exception &e) {
  std::cerr << e.what() << '\n';
  return 1;
}
