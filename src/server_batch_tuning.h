#pragma once
#include "server_protocol.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <thread>
#ifdef __linux__
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace qwen35x::server {
#ifdef __linux__
inline std::filesystem::path batch_cgroup() {
  const std::filesystem::path root = "/sys/fs/cgroup";
  std::string line;
  std::ifstream groups("/proc/self/cgroup");
  while (std::getline(groups, line))
    if (line.starts_with("0::/") && line.find("..") == std::string::npos)
      return root / std::filesystem::path(line.substr(3)).relative_path();
  return root;
}
#endif

inline std::size_t batch_memory_width(std::size_t limit) {
  std::uint64_t available = std::numeric_limits<std::uint64_t>::max();
#ifdef __linux__
  std::ifstream mem("/proc/meminfo");
  std::string line;
  while (std::getline(mem, line)) {
    if (line.starts_with("MemAvailable:")) {
      std::istringstream values(line.substr(13));
      std::uint64_t kib;
      if (values >> kib)
        available = kib * 1024;
      break;
    }
  }
  for (auto path = batch_cgroup(); path.has_relative_path(); path = path.parent_path()) {
    std::uint64_t maximum, used;
    if (std::ifstream(path / "memory.max") >> maximum && std::ifstream(path / "memory.current") >> used)
      available = std::min(available, maximum > used ? maximum - used : std::uint64_t(0));
    if (path == "/sys/fs/cgroup")
      break;
  }
#elif defined(_WIN32)
  MEMORYSTATUSEX memory{};
  memory.dwLength = sizeof(memory);
  if (GlobalMemoryStatusEx(&memory))
    available = memory.ullAvailPhys;
#endif
  // Budget for recurrent state, KV pages, batch scratch and runtime headroom.
  return static_cast<std::size_t>(
      std::clamp<std::uint64_t>(available / (96ULL * 1024 * 1024), 1, limit));
}
inline unsigned batch_cpu_capacity() {
  auto n = std::max(1u, std::thread::hardware_concurrency());
#ifdef __linux__
  cpu_set_t mask;
  if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    n = std::max(1, CPU_COUNT(&mask));
  // Read the task's effective v2 quota, including any tighter ancestor.
  const std::filesystem::path root = "/sys/fs/cgroup";
  auto path = batch_cgroup();
  for (;;) {
    long long quota = 0, period = 0;
    if (std::ifstream(path / "cpu.max") >> quota >> period && quota > 0 && period > 0)
      n = static_cast<unsigned>(std::min<long long>(n, std::max(1LL, (quota + period - 1) / period)));
    if (path == root || !path.has_relative_path())
      break;
    path = path.parent_path();
  }
#elif defined(_WIN32)
  DWORD_PTR process_mask = 0, system_mask = 0;
  if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
    unsigned count = 0;
    while (process_mask) {
      process_mask &= process_mask - 1;
      ++count;
    }
    if (count)
      n = count;
  }
#endif
  return n;
}

struct BatchTuning {
  unsigned cpus = batch_cpu_capacity();
  int threads = 1;
  std::size_t width = 1;
  Json probes = Json::array();
  Json report() const {
    return {{"cpu_capacity", cpus}, {"threads", threads}, {"batch_width", width}, {"probes", probes}};
  }
};

// Measure the actual dispatched model on this host. Each private probe engine
// is destroyed before the serving executor starts; all probes share the weights.
inline BatchTuning tune_batches(const std::shared_ptr<const CpuModel> &model, CpuEngineConfig &serving,
                                std::size_t fixed_width, std::span<const std::int32_t> decision_ids,
                                bool selected_head) {
  const bool decision = !decision_ids.empty();
  BatchTuning tuning;
  const bool auto_threads = serving.threads == 0;
  tuning.threads = auto_threads ? static_cast<int>(tuning.cpus) : serving.threads;
  const auto limit = batch_memory_width(std::min<std::size_t>(16, serving.max_resident_requests));
  tuning.width = fixed_width ? fixed_width : std::min<std::size_t>(4, limit);
  auto probe = [&](int threads, std::size_t width) {
    auto cfg = serving;
    cfg.threads = threads;
    cfg.max_queued_requests = 0;
    cfg.max_resident_requests = width;
    cfg.max_decode_batch_size = width;
    std::string error;
    auto engine = CpuEngine::create(model, cfg, error);
    if (!engine)
      throw std::runtime_error(error);
    std::vector<CpuRequestId> ids;
    CpuRequestSpec spec;
    spec.prompt_tokens = {785, 2747, 315, 279};
    spec.sampling.temperature = 0;
    spec.sampling.repetition_penalty = 1;
    spec.max_new_tokens = decision ? 1 : 8;
    if (decision) {
      spec.forced_output_tokens = {decision_ids.front()};
      if (selected_head) {
        spec.selected_logit_tokens.assign(
            decision_ids.begin(), decision_ids.begin() + std::min<std::size_t>(3, decision_ids.size()));
        spec.selected_logits_callback = [](void *, std::size_t, std::int32_t, const float *, std::size_t,
                                           std::string &) { return true; };
      }
    }
    auto prefill = [&] {
      for (std::size_t r = 0; r < width; ++r) {
        auto id = engine->submit(spec, error);
        if (!id)
          throw std::runtime_error(error);
        CpuRequestResult result;
        do {
          if (!engine->advance(id, error) || !engine->result(id, result))
            throw std::runtime_error(error);
        } while (result.status == CpuRequestStatus::prefill);
        if (decision)
          engine->release(id);
        else
          ids.push_back(id);
      }
    };
    auto decode = [&] {
      if (!engine->advance_batch(ids, error))
        throw std::runtime_error(error);
    };
    prefill();
    if (!decision)
      decode();
    std::array<double, 3> rates{};
    for (auto &sample : rates) {
      const auto start = std::chrono::steady_clock::now();
      if (decision)
        prefill();
      else {
        decode();
        decode();
      }
      const double seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      sample = width * (decision ? 1 : 2) / seconds;
    }
    std::sort(rates.begin(), rates.end());
    const double rate = rates[1];
    tuning.probes.push_back({{"threads", threads},
                             {"width", width},
                             {"positions_per_second", rate},
                             {"min_positions_per_second", rates.front()},
                             {"max_positions_per_second", rates.back()}});
    return rate;
  };
  if (serving.max_context >= (decision ? 4u : 11u) && (auto_threads || !fixed_width)) {
    double best = 0;
    if (auto_threads) {
      int cap = tuning.threads;
      std::set<int> candidates{std::max(1, cap / 2), std::max(1, cap - 1), cap};
      if (cap >= 12)
        candidates.insert(8);
      for (int threads : candidates) {
        double rate = probe(threads, tuning.width);
        if (rate > best) {
          best = rate;
          tuning.threads = threads;
        }
      }
    }
    if (!fixed_width) {
      for (auto width :
           std::set<std::size_t>{1, std::min<std::size_t>(2, limit), std::min<std::size_t>(4, limit),
                                 std::min<std::size_t>(8, limit), limit}) {
        if (auto_threads && width == std::min<std::size_t>(4, limit))
          continue;
        const double rate = probe(tuning.threads, width);
        if (rate > best) {
          best = rate;
          tuning.width = width;
        }
      }
    }
  }
  serving.threads = tuning.threads;
  return tuning;
}
} // namespace qwen35x::server
