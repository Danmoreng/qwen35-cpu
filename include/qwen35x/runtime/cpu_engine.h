#pragma once

#include "qwen35x/runtime/options.h"
#include <span>

namespace qwen35x {

// Immutable CPU weights shared by engines. Loading does not create a thread
// pool.
class CpuModel final {
public:
  static std::shared_ptr<const CpuModel>
  load(const ModelProfile &profile, const CpuLoadOptions &options,
       std::string &error);
  ~CpuModel();
  const char* weight_format() const noexcept;
  const char* head_weight_format() const noexcept;

private:
  struct Impl;
  explicit CpuModel(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
  friend class CpuEngine;
};

using CpuRequestId = std::uint64_t;
// Immutable in-process hybrid snapshot. Handles pin storage until released.
// Compatibility is bound to the actual loaded CpuModel object, not filenames.
class CpuPrefix final {
public:
  ~CpuPrefix();
  std::size_t token_count() const;
  std::size_t size_bytes() const;

private:
  struct Impl;
  explicit CpuPrefix(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
  friend class CpuEngine;
};
enum class CpuRequestStatus { prefill, decode, complete, cancelled, failed, queued };
struct CpuEngineConfig {
  // Optional serial diagnostic sink. Caller owns it for the engine lifetime.
  // Timed release comparisons must leave this null.
  std::vector<CpuDecodeStage> *operation_profile = nullptr;
  int threads = 8;
  std::size_t max_resident_requests = 32;
  std::size_t max_context = 8192;
  std::size_t prefill_chunk_size = 0;
  // Shared vocabulary-tile reduction for all-greedy batches. False preserves
  // the full-logits reference. General sampling/callbacks always use logits.
  bool batch_greedy = true;
  // Dispatch private state rows jointly when B >= executor thread count.
  bool parallel_state_batches = true;
  std::size_t max_prefix_bytes = 256 * 1024 * 1024;
  // Nonzero enables bounded serving mode. Drive it through step/read_output.
  // The record limit is max_resident_requests + max_queued_requests, including
  // terminal results until release. Queued requests have no allocated KV state.
  std::size_t max_queued_requests = 0;
  std::size_t max_decode_batch_size = 32;
  std::size_t max_buffered_tokens = 256;
  // Opt-in FP16 shared prefix pages; contiguous caches remain the reference.
  bool shared_kv_pages = false;
};
struct CpuRequestSpec {
  std::vector<std::int32_t> prompt_tokens;
  std::vector<std::int32_t> forced_output_tokens;
  std::size_t max_new_tokens = 128;
  SamplingOptions sampling;
  std::vector<std::int32_t> stop_token_ids;
  std::vector<std::vector<std::int32_t>> stop_token_sequences;
  // Invoked synchronously, with raw logits before penalties/sampling. It must
  // not reenter this engine. The callback context must outlive the request.
  ReferenceLogitsCallback logits_callback = nullptr;
  void *logits_callback_context = nullptr;
  std::shared_ptr<const CpuPrefix> prefix;
  std::string trust_namespace;
  // Serving mode only: build these leading tokens once for concurrent requests
  // in the same namespace. Zero disables automatic single-flight registration.
  std::size_t register_prefix_tokens = 0;
};
struct CpuRequestResult {
  CpuRequestStatus status = CpuRequestStatus::prefill;
  std::vector<std::int32_t> output_tokens;
  std::size_t committed_tokens = 0;
  std::size_t prefill_tokens = 0, decode_forwards = 0;
  std::size_t cached_prefix_tokens = 0;
  double prefix_restore_ms = 0;
  double queue_ms = 0;
  // Forward-call time only; aggregate serving timing must also include sampling
  // and scheduling.
  double prefill_ms = 0, decode_ms = 0;
  std::string error;
};

struct CpuEngineStats {
  std::uint64_t batch_ticks = 0, batch_rows = 0;
  std::uint64_t projection_batches = 0, lm_head_batches = 0;
  std::uint64_t greedy_lm_head_batches = 0;
  std::uint64_t state_batches = 0;
  std::size_t queued_requests = 0, resident_requests = 0;
  std::uint64_t prefix_builds = 0, prefix_build_fallbacks = 0;
  std::size_t physical_kv_bytes = 0;
};

// One owner calls these methods serially. Each engine owns one executor and
// scratch workspace; independent engines can share CpuModel read-only.
// Request IDs are monotonic within an engine and are never reused.
class CpuEngine final {
public:
  static std::unique_ptr<CpuEngine>
  create(std::shared_ptr<const CpuModel> model, const CpuEngineConfig &config,
         std::string &error);
  ~CpuEngine();
  CpuEngine(const CpuEngine &) = delete;
  CpuEngine &operator=(const CpuEngine &) = delete;
  CpuRequestId submit(CpuRequestSpec spec, std::string &error);
  // Explicit synchronous build, deduplicated by exact tokens and namespace.
  // Budgets include KV, DeltaNet, convolution, token IDs and final hidden data.
  std::shared_ptr<const CpuPrefix>
  register_prefix(std::span<const std::int32_t> tokens,
                  std::string trust_namespace, std::string &error);
  // One bounded prefill chunk or one decode forward; no background work.
  bool advance(CpuRequestId id, std::string &error);
  // One decode row per distinct active request, in the supplied row order.
  // Prompts must be initialized first. Empty batches are a no-op; B=1 keeps
  // the existing fast path. Validation errors leave every request untouched.
  bool advance_batch(std::span<const CpuRequestId> ids, std::string &error);
  // At most one admission and one prefill chunk or decode batch. No waiting for
  // a full batch. Returns success separately from whether any work progressed.
  bool step(bool &progressed, std::string &error);
  // Append up to max_tokens confirmed tokens; incomplete stop sequences are
  // withheld until resolved. Drain to unblock a slow request independently.
  bool read_output(CpuRequestId id, std::size_t max_tokens,
                   std::vector<std::int32_t> &output, std::string &error);
  CpuEngineStats stats() const;
  bool cancel(CpuRequestId id);
  bool release(CpuRequestId id);
  bool result(CpuRequestId id, CpuRequestResult &out) const;

private:
  struct Impl;
  explicit CpuEngine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace qwen35x
