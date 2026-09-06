#pragma once
#include "qwen35x/common/model_profile.h"
#include "qwen35x/cpu/q8_0.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace qwen35x {
struct SamplingOptions {
 float temperature=0.7f,top_p=0.8f;int top_k=20;
 float repetition_penalty=1.05f;std::int64_t seed=-1;
};
struct CpuLoadOptions {
 std::string model_dir,cpu_q4_h128_path;
 cpu::Q8_0Backend cpu_q8_backend=cpu::Q8_0Backend::auto_select;
};
using ReferenceLogitsCallback=bool(*)(void*,std::size_t,std::int32_t,const float*,std::size_t,std::string&);

struct CpuDecodeStage {
  std::string kind;
  std::size_t rows{}, columns{}, participants{};
  double prepare_ms{}, wall_ms{}, dispatch_ms{}, caller_ms{}, wait_ms{};
};

struct CpuPrefillStage {
  std::string kind, kernel;
  int query_tile{}, kv_tile{};
  bool shared_gqa{};
  std::size_t position{}, tokens{}, participants{};
  std::uint64_t query_key_pairs{};
  double projection_ms{}, prepare_ms{}, attention_wall_ms{}, output_ms{};
  // Summed elapsed worker intervals, not wall time or OS CPU accounting.
  double pack_worker_ms{}, qk_worker_ms{}, softmax_worker_ms{}, pv_worker_ms{};
};
}
