namespace {

// Session-owned, used serially by decode; executor jobs finish before reuse.
struct CpuDecodeWorkspace {
  struct Forward {
    std::vector<float> x, normed, attn_out, residual, post_norm;
    std::vector<float> mlp_gate, mlp_up, mlp_packed, mlp_hidden, mlp_out, final_hidden;
  } forward;
  struct Linear {
    std::vector<float> mixed_qkv, z_vec, b_vec, a_vec, packed;
    std::vector<float> beta, alpha, conv_out, core_out, gated_norm;
  } linear;
  struct Full {
    std::vector<float> q_full, k_flat, v_flat, packed, q, gate, q_normed, k_normed;
    std::vector<float> attn_cat, scores;
  } full;
};

// Separate from decode and from each nested stage; all jobs finish before reuse.
struct CpuPrefillWorkspace {
  struct Forward {
    std::vector<float> x, normed, attention, residual, post_norm;
    std::vector<float> gate_up, mlp_hidden, mlp_output, final_hidden;
  } forward;
  struct Linear {
    std::vector<float> projected, gated, conv_batch;
    std::vector<float> q_batch, k_batch, v_batch, alpha_batch, beta_batch, core_batch;
  } linear;
  struct Full {
    std::vector<float> projected, attention, query_batch, gate_batch;
    std::vector<float> q, q_normed, k_normed, scores, tiled_scratch;
    std::vector<cpu::AttentionTileTimes> tile_times;
  } full;
};

struct CpuExecutionContext {
  std::vector<CpuDecodeStage> *decode_stages = nullptr;
  std::vector<CpuPrefillStage> *stages = nullptr;
  bool tiled_attention = false, attention_gqa = false, automatic_attention = false;
  bool attention_rows_used = false, attention_tiles_used = false;
  std::string *attention_kernel_result = nullptr;
  cpu::Q8_0Backend attention_backend = cpu::Q8_0Backend::auto_select;
  int query_tile = 8, kv_tile = 64;
  CpuDecodeWorkspace decode;
  CpuPrefillWorkspace prefill;
  std::unique_ptr<cpu::CpuExecutor> executor;
  std::vector<float> q4_h128_transform_scratch;
  std::vector<cpu::Q8_0Block> quantized_input;
  std::vector<cpu::Q8_0BlockX1> prepared_q4_input;
  std::vector<cpu::Q8_0Block> quantized_batch;
  std::vector<float> quantized_batch_scales;
  std::vector<cpu::Q8_0BlockX4> packed_q8_0_batch;
  std::vector<cpu::Q4_0ArgmaxResult> greedy_results;
  std::vector<cpu::KQuantActivation> k_input, k_q8_input;
};

// A null sink avoids clock reads and allocations in production runs.
std::unique_ptr<CpuExecutionContext> make_cpu_execution_context(int threads, std::string &error) {
  auto context=std::make_unique<CpuExecutionContext>();
  cpu::CpuExecutorConfig config;
  config.thread_count=static_cast<std::size_t>(threads); config.min_parallel_rows=1;
  std::error_code ec;
  context->executor=cpu::CpuExecutor::create(config,ec);
  if(!context->executor) { error="Could not create CPU executor: "+ec.message(); return nullptr; }
  return context;
}

struct CpuDecodeProbe {
  using Clock = std::chrono::steady_clock;
  std::vector<CpuDecodeStage> *sink;
  CpuDecodeStage event;
  cpu::CpuExecutorTiming timing;
  Clock::time_point start{}, kernel_start{};
  CpuDecodeProbe(CpuExecutionContext *rt, const char *kind, std::size_t rows, std::size_t cols, std::size_t vectors=1)
      : sink(rt ? rt->decode_stages : nullptr) {
    if (sink) { event.kind=kind; event.rows=rows; event.columns=cols; event.vectors=vectors; start=kernel_start=Clock::now(); }
  }
  void prepared() {
    if (sink) { kernel_start=Clock::now(); event.prepare_ms=std::chrono::duration<double,std::milli>(kernel_start-start).count(); }
  }
  cpu::CpuExecutorTiming *executor_timing() { return sink ? &timing : nullptr; }
  ~CpuDecodeProbe() {
    if (sink) {
      event.wall_ms=std::chrono::duration<double,std::milli>(Clock::now()-kernel_start).count();
      event.participants=timing.participants; event.dispatch_ms=timing.dispatch_ms;
      event.caller_ms=timing.caller_ms; event.wait_ms=timing.wait_ms;
      sink->push_back(std::move(event));
    }
  }
};

struct CpuGreedySamplingState {
  const std::vector<int> * token_counts = nullptr;
  float repetition_penalty = 1.0F;
  int next_token = -1;
  bool enabled = false;
};

struct GgufRowPart {
  cpu::KQuantType type;
  std::size_t rows;
  std::shared_ptr<const std::vector<std::uint8_t>> bytes;
};
struct TensorData {
  // Optional identity-basis Q8 B/A gate rows following the H128 DOT4 rows.
  // Immutable after loading; output order remains QKV, Z, B, A.
  std::size_t q8_gate_rows = 0;
  std::vector<cpu::Q8_0Block> q8_gate_blocks;
  std::vector<float> q8_gate_scales;
  // Immutable original GGUF bytes, shared by individual and concatenated views.
  std::vector<GgufRowPart> gguf_parts;

  std::vector<std::int64_t> shape;
  std::vector<float> data;
  std::vector<cpu::Q4_0Block> q4_0_blocks;
  std::vector<cpu::Q4_0BlockX8> packed_q4_0_blocks;
  std::vector<float> q4_0_scales;
  std::vector<cpu::Q8_0Block> q8_0_blocks;
  std::vector<float> q8_0_scales;
  cpu::Q8_0Backend q8_0_backend = cpu::Q8_0Backend::auto_select;
  std::vector<cpu::Q4H128SignBlock> q4_h128_signs;
  bool q4_dot4 = false;
  bool uses_q4_h128_transform = false;
  std::uint64_t q4_h128_sign_seed = 0;

  bool is_q8_0() const noexcept {
    return !q8_0_blocks.empty();
  }

  bool is_q4_0() const noexcept {
    return !q4_0_blocks.empty() || !packed_q4_0_blocks.empty();
  }

  bool is_cpu_quantized() const noexcept {
    return !gguf_parts.empty() || is_q4_0() || is_q8_0();
  }
};

#include "gguf_matrix.inl"

struct PackedQ4PrefillJob {
  const cpu::Q4_0BlockX8 * matrix = nullptr;
  const cpu::Q8_0BlockX4 * vectors = nullptr;
  float * output = nullptr;
  std::size_t vector_count = 0;
  std::size_t blocks_per_row = 0;
  std::size_t output_row_stride = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
  bool dot4 = false;
};

struct PackedQ4MatvecJob {
  const cpu::Q4_0BlockX8 * matrix = nullptr;
  const cpu::Q8_0BlockX1 * vector = nullptr;
  float * output = nullptr;
  std::size_t blocks_per_row = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
  bool dot4 = false;
};

struct PackedQ4ArgmaxJob {
  const cpu::Q4_0BlockX8 * matrix = nullptr;
  const cpu::Q8_0BlockX1 * vector = nullptr;
  const int * token_counts = nullptr;
  cpu::Q4_0ArgmaxResult * results = nullptr;
  float repetition_penalty = 1.0F;
  std::size_t blocks_per_row = 0;
  std::size_t row_tile_count = 0;
  std::size_t partition_count = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
  bool dot4 = false;
};

struct Q4H128TransformJob {
  const float * input = nullptr;
  float * output = nullptr;
  std::size_t columns = 0;
  std::uint64_t sign_seed = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
};

struct Q4H128PreparePackedJob {
  const float * input = nullptr;
  cpu::Q8_0BlockX4 * output = nullptr;
  std::size_t columns = 0;
  std::size_t blocks_per_vector = 0;
  std::uint64_t sign_seed = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
  const cpu::Q4H128SignBlock * signs = nullptr;
};

void run_q4_h128_transform_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q4H128TransformJob *>(opaque_context);
  static_cast<void>(cpu::q4_h128_transform_rows_unscaled(
    job.input + row_begin * job.columns,
    job.output + row_begin * job.columns,
    row_end - row_begin,
    job.columns,
    job.sign_seed,
    job.backend));
}

void run_q4_h128_prepare_packed_tiles(
  void * opaque_context,
  const std::size_t tile_begin,
  const std::size_t tile_end) noexcept {
  auto & job = *static_cast<Q4H128PreparePackedJob *>(opaque_context);
  static_cast<void>(cpu::q4_h128_prepare_activations_4(
    job.input + tile_begin * cpu::q8_0_packed_vectors * job.columns,
    job.output + tile_begin * job.blocks_per_vector,
    (tile_end - tile_begin) * cpu::q8_0_packed_vectors,
    job.columns,
    job.sign_seed,
    job.backend, job.signs));
}

void run_packed_q4_prefill_tiles(
  void * opaque_context,
  const std::size_t tile_begin,
  const std::size_t tile_end) noexcept {
  auto & job = *static_cast<PackedQ4PrefillJob *>(opaque_context);
  (job.dot4 ? cpu::q4_dot4_matmul : cpu::q4_0_packed_matmul_q8_0)(
    job.matrix + tile_begin * job.blocks_per_row,
    job.vectors,
    job.output + tile_begin * cpu::q4_0_packed_rows,
    (tile_end - tile_begin) * cpu::q4_0_packed_rows,
    job.vector_count,
    job.blocks_per_row,
    job.output_row_stride,
    job.backend);
}

void run_packed_q4_matvec_tiles(
  void * opaque_context,
  const std::size_t tile_begin,
  const std::size_t tile_end) noexcept {
  auto & job = *static_cast<PackedQ4MatvecJob *>(opaque_context);
  (job.dot4 ? cpu::q4_dot4_matvec : cpu::q4_0_packed_matvec_prepared_q8_0)(
    job.matrix + tile_begin * job.blocks_per_row,
    job.vector,
    job.output + tile_begin * cpu::q4_0_packed_rows,
    (tile_end - tile_begin) * cpu::q4_0_packed_rows,
    job.blocks_per_row,
    job.backend);
}

void run_packed_q4_argmax_tiles(
  void * opaque_context,
  const std::size_t tile_begin,
  const std::size_t tile_end) noexcept {
  auto & job = *static_cast<PackedQ4ArgmaxJob *>(opaque_context);
  const std::size_t rows_per_partition =
    job.row_tile_count / job.partition_count;
  const std::size_t remainder = job.row_tile_count % job.partition_count;
  std::size_t partition = 0;
  for (; partition + 1 < job.partition_count; ++partition) {
    const std::size_t expected_begin =
      partition * rows_per_partition + std::min(partition, remainder);
    if (expected_begin == tile_begin) {
      break;
    }
  }
  job.results[partition] =
    (job.dot4 ? cpu::q4_dot4_argmax : cpu::q4_0_packed_matvec_prepared_q8_0_argmax)(
      job.matrix + tile_begin * job.blocks_per_row,
      job.vector,
      job.token_counts,
      job.repetition_penalty,
      tile_begin * cpu::q4_0_packed_rows,
      (tile_end - tile_begin) * cpu::q4_0_packed_rows,
      job.blocks_per_row,
      job.backend);
}

struct FullAttentionWeights {
  TensorData q_proj;
  TensorData k_proj;
  TensorData v_proj;
  TensorData o_proj;
  TensorData q_norm;
  TensorData k_norm;
  TensorData qkv_proj_cpu;
};

struct LinearAttentionWeights {
  TensorData in_proj_qkv;
  TensorData in_proj_z;
  TensorData in_proj_b;
  TensorData in_proj_a;
  TensorData in_proj_all_cpu;
  TensorData conv1d;
  std::vector<float> conv1d_kernel_major;
  TensorData out_proj;
  TensorData norm;
  TensorData a_log;
  TensorData dt_bias;
  std::vector<float> ssm_a;
};

struct LayerWeights {
  TensorData input_layernorm;
  TensorData post_attention_layernorm;
  TensorData mlp_gate;
  TensorData mlp_up;
  TensorData mlp_down;
  TensorData mlp_gate_up_cpu;
  bool is_linear = false;
  FullAttentionWeights full;
  LinearAttentionWeights linear;
};

struct ModelWeights {
  TensorData embed_tokens;
  TensorData final_norm;
  std::vector<LayerWeights> layers;
};

struct RuntimeDims {
  int n_layers = 0;
  int hidden = 0;
  int intermediate = 0;
  int vocab_size = 0;

  int n_heads = 0;
  int n_kv_heads = 0;
  int head_dim = 0;
  int rope_dim = 0;
  float rope_theta = 10000000.0f;
  float rms_eps = 1.0e-6f;

  int linear_kernel = 0;
  int linear_num_k_heads = 0;
  int linear_num_v_heads = 0;
  int linear_head_k_dim = 0;
  int linear_head_v_dim = 0;
  int linear_q_dim = 0;
  int linear_v_dim = 0;
  int linear_conv_channels = 0;
};

struct FullAttentionState {
  cpu::KvPagesF16 pages;
  std::vector<float> k_cache;
  std::vector<float> v_cache;
  std::vector<std::uint16_t> k_cache_f16;
  std::vector<std::uint16_t> v_cache_f16;
};

struct LinearAttentionState {
  std::vector<float> conv_state;
  std::vector<float> recurrent_state;
  std::size_t conv_ring_index = 0;
};

struct ModelState {
  std::vector<FullAttentionState> full_states;
  std::vector<LinearAttentionState> linear_states;
  std::vector<float> rope_inverse_frequency;
  std::vector<float> rope_cosine;
  std::vector<float> rope_sine;
};

constexpr std::uint32_t kCpuPrefixCacheStateAbiVersion = 2;

struct CpuPrefixCacheSnapshot {
 bool use_f16_cache=false;
 std::vector<std::int32_t> prefix_tokens;
 std::vector<FullAttentionState> full_states;
 std::vector<LinearAttentionState> linear_states;
};

void capture_cpu_prefix_state(
  const ModelState & state,
  const std::size_t prefix_tokens,
  const std::size_t kv_width,
  const bool use_f16_cache,
  CpuPrefixCacheSnapshot & snapshot) {
  const std::size_t cache_values = prefix_tokens * kv_width;
  snapshot.use_f16_cache = use_f16_cache;
  snapshot.full_states.resize(state.full_states.size());
  for (std::size_t layer = 0; layer < state.full_states.size(); ++layer) {
    const FullAttentionState & source = state.full_states[layer];
    FullAttentionState & destination = snapshot.full_states[layer];
    if (source.pages.width()) {
      destination.pages = source.pages.prefix(prefix_tokens);
    } else if (use_f16_cache && !source.k_cache_f16.empty()) {
      destination.k_cache_f16.assign(
        source.k_cache_f16.begin(), source.k_cache_f16.begin() +
          static_cast<std::ptrdiff_t>(cache_values));
      destination.v_cache_f16.assign(
        source.v_cache_f16.begin(), source.v_cache_f16.begin() +
          static_cast<std::ptrdiff_t>(cache_values));
    } else {
      destination.k_cache.assign(source.k_cache.begin(), source.k_cache.begin() +
        static_cast<std::ptrdiff_t>(cache_values));
      destination.v_cache.assign(source.v_cache.begin(), source.v_cache.begin() +
        static_cast<std::ptrdiff_t>(cache_values));
    }
  }
  snapshot.linear_states.resize(state.linear_states.size());
  for (std::size_t layer = 0; layer < state.linear_states.size(); ++layer) {
    snapshot.linear_states[layer].conv_state = state.linear_states[layer].conv_state;
    snapshot.linear_states[layer].recurrent_state = state.linear_states[layer].recurrent_state;
    snapshot.linear_states[layer].conv_ring_index = state.linear_states[layer].conv_ring_index;
  }
}

[[nodiscard]] bool restore_cpu_prefix_state(
  const CpuPrefixCacheSnapshot & snapshot,
  ModelState & state,
  const std::size_t prefix_tokens,
  const std::size_t kv_width) {
  const std::size_t cache_values = prefix_tokens * kv_width;
  if (snapshot.full_states.size() != state.full_states.size() ||
      snapshot.linear_states.size() != state.linear_states.size()) {
    return false;
  }
  for (std::size_t layer = 0; layer < state.full_states.size(); ++layer) {
    const FullAttentionState & source = snapshot.full_states[layer];
    FullAttentionState & destination = state.full_states[layer];
    if (source.pages.width() || destination.pages.width()) {
      if (!source.pages.width() || !destination.pages.width() ||
          source.pages.width()!=kv_width || source.pages.size()!=prefix_tokens ||
          destination.pages.capacity()<prefix_tokens) return false;
      destination.pages=source.pages.branch(destination.pages.capacity());
      continue;
    }
    const bool has_f32 = !source.k_cache.empty();
    const bool has_f16 = !source.k_cache_f16.empty();
    if (has_f32 == has_f16 || source.k_cache.size() != source.v_cache.size() ||
        source.k_cache_f16.size() != source.v_cache_f16.size() ||
        (has_f32 &&
          (source.k_cache.size() != cache_values ||
           destination.k_cache.size() < cache_values ||
           destination.v_cache.size() < cache_values)) ||
        (has_f16 &&
          (source.k_cache_f16.size() != cache_values ||
           destination.k_cache_f16.size() < cache_values ||
           destination.v_cache_f16.size() < cache_values))) {
      return false;
    }
    if (has_f16) {
      std::copy_n(source.k_cache_f16.data(), cache_values, destination.k_cache_f16.data());
      std::copy_n(source.v_cache_f16.data(), cache_values, destination.v_cache_f16.data());
    } else {
      std::copy_n(source.k_cache.data(), cache_values, destination.k_cache.data());
      std::copy_n(source.v_cache.data(), cache_values, destination.v_cache.data());
    }
  }
  for (std::size_t layer = 0; layer < state.linear_states.size(); ++layer) {
    const LinearAttentionState & source = snapshot.linear_states[layer];
    LinearAttentionState & destination = state.linear_states[layer];
    if (source.conv_state.size() != destination.conv_state.size() ||
        source.recurrent_state.size() != destination.recurrent_state.size()) {
      return false;
    }
    destination.conv_state = source.conv_state;
    destination.recurrent_state = source.recurrent_state;
    destination.conv_ring_index = source.conv_ring_index;
  }
  return true;
}

[[nodiscard]] std::size_t cpu_prefix_cache_size_bytes(
  const CpuPrefixCacheSnapshot & snapshot) noexcept {
  std::size_t bytes = snapshot.prefix_tokens.size() * sizeof(std::int32_t);
  for (const FullAttentionState & state : snapshot.full_states) {
    bytes += state.pages.payload_bytes() + state.pages.size()*2*sizeof(const std::uint16_t*);
    bytes += (state.k_cache.size() + state.v_cache.size()) * sizeof(float);
    bytes += (state.k_cache_f16.size() + state.v_cache_f16.size()) *
      sizeof(std::uint16_t);
  }
  for (const LinearAttentionState & state : snapshot.linear_states) {
    bytes += (state.conv_state.size() + state.recurrent_state.size()) * sizeof(float);
  }
  return bytes;
}

void pack_conv1d_kernel_major(
  LinearAttentionWeights & weights,
  const RuntimeDims & dims) {
  weights.conv1d_kernel_major.resize(weights.conv1d.data.size());
  for (int kernel = 0; kernel < dims.linear_kernel; ++kernel) {
    for (int channel = 0; channel < dims.linear_conv_channels; ++channel) {
      weights.conv1d_kernel_major[
        static_cast<std::size_t>(kernel * dims.linear_conv_channels + channel)] =
        weights.conv1d.data[
          static_cast<std::size_t>(channel * dims.linear_kernel + kernel)];
    }
  }
}

struct TiledAttentionCpuJob {
  cpu::TiledAttention args;
  float *scratch;
  std::size_t partitions, tasks, scratch_stride;
  cpu::Q8_0Backend backend;
  cpu::AttentionTileTimes *times = nullptr;
};
void run_tiled_attention_cpu(void *opaque, std::size_t begin, std::size_t end) noexcept {
  auto &job=*static_cast<TiledAttentionCpuJob *>(opaque);
  for(std::size_t p=begin;p<end;++p) {
    // Cyclic tiles distribute the triangular workload across participants.
    for(std::size_t t=p;t<job.tasks;t+=job.partitions)
      cpu::causal_attention_tiled(job.args,t,job.scratch+p*job.scratch_stride,job.backend,job.times ? job.times+p : nullptr);
  }
}

struct FullAttentionBatchCpuJob {
  const float * queries = nullptr;
  const float * gates = nullptr;
  const float * k_cache = nullptr;
  const float * v_cache = nullptr;
  const std::uint16_t * k_cache_f16 = nullptr;
  const std::uint16_t * v_cache_f16 = nullptr;
  float * scores = nullptr;
  float * output = nullptr;
  std::size_t context_stride = 0;
  std::size_t query_width = 0;
  std::size_t kv_width = 0;
  int position_start = 0;
  int head_count = 0;
  int kv_head_count = 0;
  int head_dim = 0;
  float attention_scale = 1.0F;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
  std::size_t score_partitions = 0;
  std::size_t attention_rows = 0;
  const cpu::AttentionKvRows *pages = nullptr;
};

void run_full_attention_batch_cpu_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<FullAttentionBatchCpuJob *>(opaque_context);
  // Each job item owns one scratch row; a worker can process several items.
  for (std::size_t partition = row_begin; partition < row_end; ++partition) {
    const std::size_t base = job.attention_rows / job.score_partitions;
    const std::size_t extra = job.attention_rows % job.score_partitions;
    const std::size_t begin = partition * base + std::min(partition, extra);
    const std::size_t end = begin + base + (partition < extra ? 1 : 0);
    cpu::causal_attention_batch_rows(
      job.queries, job.gates, job.k_cache, job.v_cache,
      job.k_cache_f16, job.v_cache_f16,
      job.scores + partition * job.context_stride, job.output,
      job.context_stride, job.query_width, job.kv_width, job.position_start,
      job.head_count, job.kv_head_count, job.head_dim, job.attention_scale,
      begin, end, job.backend, true, job.pages);
  }
}

void run_full_attention_decode_cpu_pairs(
  void * opaque_context,
  const std::size_t pair_begin,
  const std::size_t pair_end) noexcept {
  auto & job = *static_cast<FullAttentionBatchCpuJob *>(opaque_context);
  cpu::causal_attention_decode_gqa_pairs(
    job.queries, job.gates, job.k_cache, job.v_cache,
    job.k_cache_f16, job.v_cache_f16, job.scores, job.output,
    job.context_stride, job.query_width, job.kv_width, job.position_start + 1,
    job.head_count, job.kv_head_count, job.head_dim, job.attention_scale,
    pair_begin, pair_end, job.backend, job.pages);
}

struct DecodeProfilingAccumulator {
  double embedding_ms = 0.0;
  double attention_ms = 0.0;
  double mlp_ms = 0.0;
  double logits_ms = 0.0;
  double sampling_ms = 0.0;
  double stop_checks_ms = 0.0;
  int forward_pass_tokens = 0;
};
double elapsed_ms(const std::chrono::steady_clock::time_point start_time) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
}
float sigmoidf_stable(const float x) {
  if (x >= 0.0f) {
    const float z = std::exp(-x);
    return 1.0f / (1.0f + z);
  }
  const float z = std::exp(x);
  return z / (1.0f + z);
}

float siluf(const float x) {
  return x * sigmoidf_stable(x);
}

float softplusf_stable(const float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return std::exp(x);
  }
  return std::log1p(std::exp(x));
}

bool tensor_is_2d(const TensorData & t, std::int64_t rows, std::int64_t cols) {
  return t.shape.size() == 2 && t.shape[0] == rows && t.shape[1] == cols;
}

bool tensor_is_1d(const TensorData & t, std::int64_t size) {
  return t.shape.size() == 1 && t.shape[0] == size;
}

bool tensor_is_conv1d(const TensorData & t, std::int64_t channels, std::int64_t kernel) {
  if (t.shape.size() == 2) {
    return t.shape[0] == channels && t.shape[1] == kernel;
  }
  if (t.shape.size() == 3) {
    return t.shape[0] == channels && t.shape[1] == 1 && t.shape[2] == kernel;
  }
  return false;
}

bool q4_h128_shape_matches(
  const std::vector<std::uint64_t> & actual,
  const std::initializer_list<std::int64_t> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  std::size_t index = 0;
  for (const std::int64_t dimension : expected) {
    if (dimension <= 0 || actual[index] != static_cast<std::uint64_t>(dimension)) {
      return false;
    }
    ++index;
  }
  return true;
}

bool load_q4_h128_quantized_checked(
  Q4H128ArtifactReader & reader,
  const std::string & tensor_name,
  const std::int64_t rows,
  const std::int64_t cols,
  const bool expect_h128,
  const cpu::Q8_0Backend backend,
  TensorData & out,
  std::string & error_message,
  const bool retain_scales = true) {
  const Q4H128TensorInfo * info = reader.find_tensor(tensor_name);
  if (info && info->encoding == Q4H128TensorEncoding::q8_0 && !expect_h128 &&
      q4_h128_shape_matches(info->shape, {rows, cols})) {
    out = TensorData{}; out.shape = {rows, cols}; out.q8_0_backend = backend;
    out.q8_0_blocks.resize(static_cast<std::size_t>(rows)*(static_cast<std::size_t>(cols)/32));
    if (!reader.read_tensor_into(tensor_name, out.q8_0_blocks.data(),
          out.q8_0_blocks.size()*sizeof(cpu::Q8_0Block), error_message)) return false;
    out.q8_0_scales.resize(out.q8_0_blocks.size());
    cpu::q8_0_scales_to_f32(out.q8_0_blocks.data(), out.q8_0_scales.data(), out.q8_0_blocks.size());
    for (float scale : out.q8_0_scales) if (!std::isfinite(scale) || scale < 0) {
      error_message = "Invalid Q8 weight scale: " + tensor_name; return false;
    }
    return true;
  }
  const auto dot4_encoding = expect_h128 ? Q4H128TensorEncoding::q4_h128_cpu_dot4
                                         : Q4H128TensorEncoding::q4_0_cpu_dot4;
  if (info == nullptr || info->encoding != dot4_encoding ||
      !q4_h128_shape_matches(info->shape, {rows, cols})) {
    error_message = "Required Q4_H128 tensor is missing, misencoded, or has the wrong shape: " +
      tensor_name;
    return false;
  }
  if (q4_h128_encoding_cpu_packed(info->encoding)) {
    out.q4_dot4 = q4_h128_encoding_dot4(info->encoding);
    if (info->data_size > std::numeric_limits<std::size_t>::max() ||
        info->data_size % sizeof(cpu::Q4_0BlockX8) != 0) {
      error_message = "CPU-packed Q4 tensor storage is too large or misaligned.";
      return false;
    }
    out.shape = {rows, cols};
    out.data.clear();
    out.q4_0_blocks.clear();
    out.q4_0_scales.clear();
    out.q8_0_blocks.clear();
    out.q8_0_scales.clear();
    out.packed_q4_0_blocks.resize(static_cast<std::size_t>(info->data_size) / sizeof(cpu::Q4_0BlockX8));
    if (!reader.read_tensor_into(tensor_name, out.packed_q4_0_blocks.data(),
          static_cast<std::size_t>(info->data_size), error_message)) {
      out.packed_q4_0_blocks.clear();
      return false;
    }
    out.q8_0_backend = backend;

    out.uses_q4_h128_transform = expect_h128;
    out.q4_h128_sign_seed = expect_h128 ? info->sign_seed : 0;
    out.q4_h128_signs.resize(out.uses_q4_h128_transform
      ? static_cast<std::size_t>(out.shape[1]) / cpu::q4_h128_transform_size : 0);
    cpu::q4_h128_prepare_signs(out.q4_h128_signs.data(), out.q4_h128_signs.size(), out.q4_h128_sign_seed);
    return true;
  }
  std::vector<std::uint8_t> bytes;
  if (!reader.read_tensor_bytes(tensor_name, bytes, error_message) ||
      bytes.size() % sizeof(cpu::Q4_0Block) != 0) {
    return false;
  }
  out.shape = {rows, cols};
  out.data.clear();
  out.q4_0_blocks.resize(bytes.size() / sizeof(cpu::Q4_0Block));
  std::memcpy(out.q4_0_blocks.data(), bytes.data(), bytes.size());
  out.packed_q4_0_blocks.clear();
  out.q4_0_scales.clear();
  out.q8_0_blocks.clear();
  out.q8_0_scales.clear();
  const std::size_t row_count = static_cast<std::size_t>(rows);
  const std::size_t blocks_per_row =
    static_cast<std::size_t>(cols) / cpu::q4_0_values_per_block;
  if ((row_count % cpu::q4_0_packed_rows) == 0) {
    out.packed_q4_0_blocks.resize(
      (row_count / cpu::q4_0_packed_rows) * blocks_per_row);
    cpu::q4_0_pack_rows_8(
      out.q4_0_blocks.data(), out.packed_q4_0_blocks.data(),
      row_count, blocks_per_row);
  }
  if (retain_scales) {
    out.q4_0_scales.resize(out.q4_0_blocks.size());
    cpu::q4_0_scales_to_f32(
      out.q4_0_blocks.data(), out.q4_0_scales.data(), out.q4_0_blocks.size());
  }
  out.q8_0_backend = backend;

  out.uses_q4_h128_transform = expect_h128;
  out.q4_h128_sign_seed = expect_h128 ? info->sign_seed : 0;
  out.q4_h128_signs.resize(out.uses_q4_h128_transform
    ? static_cast<std::size_t>(out.shape[1]) / cpu::q4_h128_transform_size : 0);
  cpu::q4_h128_prepare_signs(out.q4_h128_signs.data(), out.q4_h128_signs.size(), out.q4_h128_sign_seed);
  return true;
}

bool load_q4_h128_f32_checked(
  Q4H128ArtifactReader & reader,
  const std::string & tensor_name,
  const std::initializer_list<std::int64_t> expected_shape,
  TensorData & out,
  std::string & error_message) {
  const Q4H128TensorInfo * info = reader.find_tensor(tensor_name);
  if (info == nullptr || info->encoding != Q4H128TensorEncoding::f32 ||
      !q4_h128_shape_matches(info->shape, expected_shape)) {
    error_message = "Required Q4_H128 F32 tensor is missing or has the wrong shape: " +
      tensor_name;
    return false;
  }
  std::vector<std::uint8_t> bytes;
  if (!reader.read_tensor_bytes(tensor_name, bytes, error_message) ||
      bytes.size() % sizeof(float) != 0) {
    return false;
  }
  out.shape.clear();
  for (const std::uint64_t dimension : info->shape) {
    out.shape.push_back(static_cast<std::int64_t>(dimension));
  }
  out.data.resize(bytes.size() / sizeof(float));
  std::memcpy(out.data.data(), bytes.data(), bytes.size());
  out.q4_0_blocks.clear();
  out.packed_q4_0_blocks.clear();
  out.q4_0_scales.clear();
  out.q8_0_blocks.clear();
  out.q8_0_scales.clear();
  out.uses_q4_h128_transform = false;
  out.q4_h128_sign_seed = 0;
  return true;
}

bool pack_quantized_row_concat(
  const std::initializer_list<TensorData *> parts,
  TensorData & out,
  std::string & error_message) {
  if (parts.size() == 0) {
    error_message = "Quantized packed tensor has no source tensors.";
    return false;
  }

  const auto* first_part=*parts.begin();
  if (first_part && first_part->shape.size()==2 && !first_part->gguf_parts.empty()) {
    out = TensorData{};
    out.shape = {0, (*parts.begin())->shape[1]};
    out.q8_0_backend = (*parts.begin())->q8_0_backend;
    for (const auto* part : parts) {
      if (!part || part->shape.size()!=2 || part->shape[1]!=out.shape[1] || part->gguf_parts.empty()) {
        error_message="Incompatible GGUF row concatenation.";return false;
      }
      out.shape[0]+=part->shape[0];
      out.gguf_parts.insert(out.gguf_parts.end(),part->gguf_parts.begin(),part->gguf_parts.end());
    }
    return true;
  }
  std::int64_t cols = -1;
  std::int64_t total_rows = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
  bool q4_dot4 = false;
  bool uses_q4_h128_transform = false;
  std::uint64_t q4_h128_sign_seed = 0;
  bool first = true;
  bool use_q4_0 = false;
  std::size_t total_blocks = 0;
  for (const TensorData * part : parts) {
    if (part == nullptr || part->shape.size() != 2 || !part->is_cpu_quantized()) {
      error_message = "Quantized packed tensor expects non-empty 2D source tensors.";
      return false;
    }
    const bool part_q4_0 = part->is_q4_0();
    const std::size_t part_blocks = part_q4_0
      ? part->q4_0_blocks.size() : part->q8_0_blocks.size();
    const std::size_t part_scales = part_q4_0
      ? part->q4_0_scales.size() : part->q8_0_scales.size();
    if (part_scales != part_blocks) {
      error_message = "Quantized packed tensor source scale storage mismatch.";
      return false;
    }
    if (first) {
      cols = part->shape[1];
      backend = part->q8_0_backend;
      use_q4_0 = part_q4_0;
      q4_dot4 = part->q4_dot4;
      uses_q4_h128_transform = part->uses_q4_h128_transform;
      q4_h128_sign_seed = part->q4_h128_sign_seed;
      first = false;
    } else if (part->shape[1] != cols || part->q8_0_backend != backend ||
               part_q4_0 != use_q4_0 ||
               part->q4_dot4 != q4_dot4 ||
               part->uses_q4_h128_transform != uses_q4_h128_transform ||
               part->q4_h128_sign_seed != q4_h128_sign_seed) {
      error_message = "Quantized packed tensor source tensors have incompatible formats.";
      return false;
    }
    if (part->shape[0] <= 0 || total_rows > std::numeric_limits<std::int64_t>::max() - part->shape[0] ||
        total_blocks > std::numeric_limits<std::size_t>::max() - part_blocks) {
      error_message = "Quantized packed tensor dimensions overflow.";
      return false;
    }
    total_rows += part->shape[0];
    total_blocks += part_blocks;
  }
  if (cols <= 0 || (cols % static_cast<std::int64_t>(cpu::q8_0_values_per_block)) != 0) {
    error_message = "Quantized packed tensor has an invalid column count.";
    return false;
  }
  const std::size_t expected_blocks =
    static_cast<std::size_t>(total_rows) *
    (static_cast<std::size_t>(cols) / cpu::q8_0_values_per_block);
  if (total_blocks != expected_blocks) {
    error_message = "Quantized packed tensor source storage size mismatch.";
    return false;
  }

  out.shape = {total_rows, cols};
  out.data.clear();
  out.q4_0_blocks.clear();
  out.packed_q4_0_blocks.clear();
  out.q4_0_scales.clear();
  out.q8_0_blocks.clear();
  out.q8_0_scales.clear();
  if (use_q4_0) {
    out.q4_0_blocks.reserve(total_blocks);
    out.q4_0_scales.reserve(total_blocks);
  } else {
    out.q8_0_blocks.reserve(total_blocks);
    out.q8_0_scales.reserve(total_blocks);
  }
  out.q8_0_backend = backend;

  out.q4_dot4 = q4_dot4;
  out.uses_q4_h128_transform = uses_q4_h128_transform;
  out.q4_h128_sign_seed = q4_h128_sign_seed;
  out.q4_h128_signs.resize(out.uses_q4_h128_transform
    ? static_cast<std::size_t>(out.shape[1]) / cpu::q4_h128_transform_size : 0);
  cpu::q4_h128_prepare_signs(out.q4_h128_signs.data(), out.q4_h128_signs.size(), out.q4_h128_sign_seed);
  for (const TensorData * part : parts) {
    if (use_q4_0) {
      out.q4_0_blocks.insert(
        out.q4_0_blocks.end(), part->q4_0_blocks.begin(), part->q4_0_blocks.end());
      out.q4_0_scales.insert(
        out.q4_0_scales.end(), part->q4_0_scales.begin(), part->q4_0_scales.end());
    } else {
      out.q8_0_blocks.insert(
        out.q8_0_blocks.end(), part->q8_0_blocks.begin(), part->q8_0_blocks.end());
      out.q8_0_scales.insert(
        out.q8_0_scales.end(), part->q8_0_scales.begin(), part->q8_0_scales.end());
    }
  }

  if (use_q4_0 &&
      (static_cast<std::size_t>(total_rows) % cpu::q4_0_packed_rows) == 0) {
    const std::size_t blocks_per_row =
      static_cast<std::size_t>(cols) / cpu::q4_0_values_per_block;
    out.packed_q4_0_blocks.resize(
      (static_cast<std::size_t>(total_rows) / cpu::q4_0_packed_rows) *
      blocks_per_row);
    (out.q4_dot4?cpu::q4_dot4_pack_rows_8:cpu::q4_0_pack_rows_8)(
      out.q4_0_blocks.data(), out.packed_q4_0_blocks.data(),
      static_cast<std::size_t>(total_rows), blocks_per_row);
  }

  // The concatenated tensors replace these source projections at runtime.
  for (TensorData * part : parts) {
    std::vector<cpu::Q4_0Block>().swap(part->q4_0_blocks);
    std::vector<cpu::Q4_0BlockX8>().swap(part->packed_q4_0_blocks);
    std::vector<float>().swap(part->q4_0_scales);
    std::vector<cpu::Q8_0Block>().swap(part->q8_0_blocks);
    std::vector<float>().swap(part->q8_0_scales);
  }
  return true;
}

bool prepare_q4_decode_activation(
  const TensorData & w, const float * input, const std::size_t columns,
  cpu::Q8_0BlockX1 * output, std::string & error_message) {
  if (w.uses_q4_h128_transform) {
    if (!cpu::q4_h128_prepare_activation_1(
          input, output, columns, w.q4_h128_sign_seed, w.q8_0_backend,
          w.q4_h128_signs.empty() ? nullptr : w.q4_h128_signs.data())) {
      error_message = "Q4_H128 decode activation preparation failed.";
      return false;
    }
  } else {
    cpu::q8_0_quantize_vector_1(input, output, columns / cpu::q8_0_values_per_block, w.q8_0_backend);
  }
  return true;
}

// Small Q8 tail: prepare each original-basis input once and write directly
// into the fused output stride. Avoid a second executor launch for 32 rows.
bool run_q8_gate_tail(CpuExecutionContext *context, const TensorData &w,
                     const float *inputs, std::size_t count, float *output,
                     std::string &error) {
  if (!w.q8_gate_rows) return true;
  const auto columns = static_cast<std::size_t>(w.shape[1]);
  const auto stride = static_cast<std::size_t>(w.shape[0]);
  const auto blocks = columns/32;
  if (w.q8_gate_rows >= stride || w.q8_gate_blocks.size() != w.q8_gate_rows*blocks ||
      w.q8_gate_scales.size() != w.q8_gate_blocks.size()) {
    error = "Invalid mixed Q4/Q8 gate storage"; return false;
  }
  std::vector<cpu::Q8_0Block> local;
  auto &prepared = context ? context->quantized_batch : local;
  prepared.resize(count*blocks);
  cpu::q8_0_quantize(inputs, prepared.data(), count*blocks, w.q8_0_backend);
  cpu::q8_0_matmul(w.q8_gate_blocks.data(), prepared.data(),
      output+stride-w.q8_gate_rows, w.q8_gate_rows, count, blocks, stride,
      w.q8_0_backend, nullptr, w.q8_gate_scales.data());
  return true;
}

bool matvec_2d(CpuExecutionContext *cpu_context,
  const TensorData & w,
  const std::vector<float> & x,
  std::vector<float> & out,
  std::string & error_message) {
  const int output_rows = static_cast<int>(w.shape[0]);
  const int rows = output_rows-static_cast<int>(w.q8_gate_rows);
  const int cols = static_cast<int>(w.shape[1]);
  if (static_cast<int>(x.size()) != cols) {
    error_message = "matvec input size mismatch.";
    return false;
  }



  if (!w.gguf_parts.empty())
    return gguf_matmul(cpu_context,w,x,1,out,error_message);

  if (w.is_q4_0()) {
    if ((cols % static_cast<int>(cpu::q4_0_values_per_block)) != 0) {
      error_message = "Q4_0 matvec column count is not divisible by 32.";
      return false;
    }
    const std::size_t blocks_per_row =
      static_cast<std::size_t>(cols) / cpu::q4_0_values_per_block;
    if ((static_cast<std::size_t>(rows) % cpu::q4_0_packed_rows) != 0 ||
        w.packed_q4_0_blocks.size() !=
          (static_cast<std::size_t>(rows) / cpu::q4_0_packed_rows) * blocks_per_row) {
      error_message = "Packed Q4_0 matvec weight storage size mismatch.";
      return false;
    }
    CpuDecodeProbe probe(cpu_context, "q4-matvec", rows, cols);
    std::vector<cpu::Q8_0BlockX1> local_prepared_input;
    auto & prepared_input = cpu_context != nullptr
      ? cpu_context->prepared_q4_input : local_prepared_input;
    prepared_input.resize(blocks_per_row);
    if (!prepare_q4_decode_activation(w, x.data(), static_cast<std::size_t>(cols),
          prepared_input.data(), error_message)) {
      return false;
    }
    probe.prepared();
    out.resize(static_cast<std::size_t>(output_rows));
    if (cpu_context != nullptr && cpu_context->executor != nullptr) {
      PackedQ4MatvecJob job{
        w.packed_q4_0_blocks.data(), prepared_input.data(), out.data(),
        blocks_per_row, w.q8_0_backend, w.q4_dot4,
      };
      const cpu::CpuExecutorStatus status =
        cpu_context->executor->parallel_for_rows(
          static_cast<std::size_t>(rows) / cpu::q4_0_packed_rows,
          run_packed_q4_matvec_tiles,
          &job, probe.executor_timing());
      if (status != cpu::CpuExecutorStatus::ok) {
        error_message = std::string("Q4_0 CPU executor failed: ") +
          cpu::cpu_executor_status_name(status) + ".";
        return false;
      }
    } else {
      (w.q4_dot4 ? cpu::q4_dot4_matvec : cpu::q4_0_packed_matvec_prepared_q8_0)(
        w.packed_q4_0_blocks.data(), prepared_input.data(), out.data(),
        static_cast<std::size_t>(rows), blocks_per_row, w.q8_0_backend);
    }
    return run_q8_gate_tail(cpu_context, w, x.data(), 1, out.data(), error_message);
  }

  if (w.is_q8_0()) {
    if ((cols % static_cast<int>(cpu::q8_0_values_per_block)) != 0) {
      error_message = "Q8_0 matvec column count is not divisible by 32.";
      return false;
    }
    const std::size_t blocks_per_row =
      static_cast<std::size_t>(cols) / cpu::q8_0_values_per_block;
    const std::size_t expected_blocks = static_cast<std::size_t>(rows) * blocks_per_row;
    if (w.q8_0_blocks.size() != expected_blocks) {
      error_message = "Q8_0 matvec weight storage size mismatch.";
      return false;
    }

    std::vector<cpu::Q8_0Block> local_quantized_input;
    std::vector<cpu::Q8_0Block> * quantized_input = &local_quantized_input;
    if (cpu_context != nullptr) {
      quantized_input = &cpu_context->quantized_input;
    }
    quantized_input->resize(blocks_per_row);
    cpu::q8_0_quantize(x.data(), quantized_input->data(), blocks_per_row, w.q8_0_backend);
    out.resize(static_cast<std::size_t>(rows));
    if (cpu_context != nullptr && cpu_context->executor != nullptr) {
      const cpu::CpuExecutorStatus status = cpu_context->executor->q8_0_matvec(
        w.q8_0_blocks.data(),
        quantized_input->data(),
        out.data(),
        static_cast<std::size_t>(rows),
        blocks_per_row,
        w.q8_0_backend);
      if (status != cpu::CpuExecutorStatus::ok) {
        error_message = std::string("Q8_0 CPU executor failed: ") + cpu::cpu_executor_status_name(status) + ".";
        return false;
      }
    } else {
      cpu::q8_0_matvec(
        w.q8_0_blocks.data(),
        quantized_input->data(),
        out.data(),
        static_cast<std::size_t>(rows),
        blocks_per_row,
        w.q8_0_backend);
    }
    return true;
  }

  if (w.data.size() != static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols)) {
    error_message = "F32 matvec weight storage size mismatch.";
    return false;
  }

  out.assign(static_cast<std::size_t>(rows), 0.0f);
  for (int r = 0; r < rows; ++r) {
    const float * row_ptr = w.data.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(cols);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) {
      sum += row_ptr[c] * x[static_cast<std::size_t>(c)];
    }
    out[static_cast<std::size_t>(r)] = sum;
  }
  return true;
}

bool greedy_q4_token(CpuExecutionContext *cpu_context,
  const TensorData & weights,
  const std::vector<float> & input,
  const std::vector<int> & token_counts,
  const float repetition_penalty,
  int & out_token,
  std::string & error_message) {
  if (!weights.gguf_parts.empty() || weights.is_q8_0()) {
    std::vector<float> logits;
    if(!matvec_2d(cpu_context,weights,input,logits,error_message))return false;
    if(token_counts.size()<logits.size()) {error_message="Token count size mismatch.";return false;}
    float best=-std::numeric_limits<float>::infinity();out_token=0;
    for(std::size_t i=0;i<logits.size();++i) {
      float v=logits[i];
      if(token_counts[i]>0 && repetition_penalty>1) v=v>0?v/repetition_penalty:v*repetition_penalty;
      if(v>best) {best=v;out_token=static_cast<int>(i);}
    }
    return true;
  }
  if (!weights.is_q4_0() || cpu_context == nullptr ||
      cpu_context->executor == nullptr || weights.shape.size() != 2) {
    error_message = "Fused greedy Q4 selection requires packed weights and a CPU executor.";
    return false;
  }
  const std::size_t rows = static_cast<std::size_t>(weights.shape[0]);
  const std::size_t cols = static_cast<std::size_t>(weights.shape[1]);
  if (input.size() != cols || token_counts.size() < rows ||
      (rows % cpu::q4_0_packed_rows) != 0 ||
      (cols % cpu::q4_0_values_per_block) != 0) {
    error_message = "Fused greedy Q4 selection received incompatible dimensions.";
    return false;
  }
  const std::size_t blocks_per_row = cols / cpu::q4_0_values_per_block;
  const std::size_t row_tiles = rows / cpu::q4_0_packed_rows;
  if (weights.packed_q4_0_blocks.size() != row_tiles * blocks_per_row) {
    error_message = "Fused greedy Q4 selection weight storage size mismatch.";
    return false;
  }

  CpuExecutionContext & runtime = *cpu_context;
  CpuDecodeProbe probe(&runtime, "q4-argmax", rows, cols);
  runtime.prepared_q4_input.resize(blocks_per_row);
  if (!prepare_q4_decode_activation(weights, input.data(), cols,
        runtime.prepared_q4_input.data(), error_message)) {
    return false;
  }

  probe.prepared();
  runtime.greedy_results.assign(
    runtime.executor->thread_count(),
    cpu::Q4_0ArgmaxResult{-std::numeric_limits<float>::infinity(), 0});
  PackedQ4ArgmaxJob job{
    weights.packed_q4_0_blocks.data(),
    runtime.prepared_q4_input.data(),
    token_counts.data(),
    runtime.greedy_results.data(),
    repetition_penalty,
    blocks_per_row,
    row_tiles,
    runtime.executor->thread_count(),
    weights.q8_0_backend, weights.q4_dot4,
  };
  const cpu::CpuExecutorStatus status = runtime.executor->parallel_for_rows(
    row_tiles, run_packed_q4_argmax_tiles, &job, probe.executor_timing());
  if (status != cpu::CpuExecutorStatus::ok) {
    error_message = std::string("Fused greedy Q4 executor failed: ") +
      cpu::cpu_executor_status_name(status) + ".";
    return false;
  }

  cpu::Q4_0ArgmaxResult best{-std::numeric_limits<float>::infinity(), 0};
  for (const cpu::Q4_0ArgmaxResult candidate : runtime.greedy_results) {
    if (candidate.value > best.value) {
      best = candidate;
    }
  }
  out_token = static_cast<int>(best.index);
  return out_token >= 0 && static_cast<std::size_t>(out_token) < rows;
}

bool matmul_2d_quantized_batch(CpuExecutionContext *cpu_context,
  const TensorData & w,
  const std::vector<float> & inputs,
  const std::size_t batch_size,
  std::vector<float> & out,
  std::string & error_message) {
  if (!w.is_cpu_quantized() || w.shape.size() != 2 || cpu_context == nullptr) {
    error_message = "CPU batched matmul requires a 2D Q4_0/Q8_0 tensor with a runtime.";
    return false;
  }
  const std::size_t output_stride = static_cast<std::size_t>(w.shape[0]);
  const std::size_t rows = output_stride-w.q8_gate_rows;
  const std::size_t cols = static_cast<std::size_t>(w.shape[1]);
  if (cols == 0 || (cols % cpu::q8_0_values_per_block) != 0 ||
      batch_size > std::numeric_limits<std::size_t>::max() / cols ||
      inputs.size() != batch_size * cols) {
    error_message = "CPU batched matmul input shape mismatch.";
    return false;
  }
  if (!w.gguf_parts.empty())
    return gguf_matmul(cpu_context,w,inputs,batch_size,out,error_message);
  const std::size_t blocks_per_row = cols / cpu::q8_0_values_per_block;
  if (rows > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      batch_size > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      rows == 0 || output_stride == 0 || batch_size > std::numeric_limits<std::size_t>::max() / output_stride) {
    error_message = "CPU batched matmul storage size overflow or mismatch.";
    return false;
  }
  const std::size_t expected_weight_blocks = rows * blocks_per_row;
  if ((w.is_q4_0() &&
       ((rows % cpu::q4_0_packed_rows) != 0 ||
        w.packed_q4_0_blocks.size() !=
          (rows / cpu::q4_0_packed_rows) * blocks_per_row)) ||
      (w.is_q8_0() &&
       (w.q8_0_blocks.size() != expected_weight_blocks ||
        w.q8_0_scales.size() != expected_weight_blocks))) {
    error_message = "CPU batched matmul weight storage size mismatch.";
    return false;
  }

  out.resize(batch_size * output_stride);
  if (batch_size == 0 || rows == 0) {
    return true;
  }
  const bool fuse_h128_packed_q4 =
    w.uses_q4_h128_transform && w.is_q4_0() &&
    (rows % cpu::q4_0_packed_rows) == 0 &&
    !w.packed_q4_0_blocks.empty();
  const float * quantization_inputs = inputs.data();
  if (w.uses_q4_h128_transform && !fuse_h128_packed_q4) {
    std::vector<float> & transformed = cpu_context->q4_h128_transform_scratch;
    transformed.resize(batch_size * cols);
    if (cpu_context->executor != nullptr && batch_size > 1) {
      Q4H128TransformJob transform_job{
        inputs.data(), transformed.data(), cols, w.q4_h128_sign_seed,
        w.q8_0_backend,
      };
      const cpu::CpuExecutorStatus status =
        cpu_context->executor->parallel_for_rows(
          batch_size, run_q4_h128_transform_rows, &transform_job);
      if (status != cpu::CpuExecutorStatus::ok) {
        error_message = std::string("Q4_H128 activation transform executor failed: ") +
          cpu::cpu_executor_status_name(status) + ".";
        return false;
      }
    } else if (!cpu::q4_h128_transform_rows_unscaled(
                 inputs.data(), transformed.data(), batch_size, cols,
                 w.q4_h128_sign_seed, w.q8_0_backend)) {
      error_message = "Q4_H128 batched activation transform failed.";
      return false;
    }
    quantization_inputs = transformed.data();
  }
  if (w.is_q4_0() &&
      (rows % cpu::q4_0_packed_rows) == 0 &&
      !w.packed_q4_0_blocks.empty()) {
    const std::size_t packed_vector_count =
      batch_size - (batch_size % cpu::q8_0_packed_vectors);
    const std::size_t expected_packed_blocks =
      (rows / cpu::q4_0_packed_rows) * blocks_per_row;
    if (w.packed_q4_0_blocks.size() != expected_packed_blocks) {
      error_message = "Packed Q4_0 prefill weight storage size mismatch.";
      return false;
    }
    if (packed_vector_count != 0) {
      CpuDecodeProbe probe(cpu_context, "q4-matmul", rows, cols, packed_vector_count);
      std::vector<cpu::Q8_0BlockX4> & packed =
        cpu_context->packed_q8_0_batch;
      packed.resize(
        (packed_vector_count / cpu::q8_0_packed_vectors) * blocks_per_row);
      if (fuse_h128_packed_q4) {
        const std::size_t vector_tiles =
          packed_vector_count / cpu::q8_0_packed_vectors;
        if (cpu_context->executor != nullptr && vector_tiles > 1) {
          Q4H128PreparePackedJob prepare_job{
            inputs.data(), packed.data(), cols, blocks_per_row,
            w.q4_h128_sign_seed, w.q8_0_backend, w.q4_h128_signs.data(),
          };
          const cpu::CpuExecutorStatus status =
            cpu_context->executor->parallel_for_rows(
              vector_tiles, run_q4_h128_prepare_packed_tiles, &prepare_job);
          if (status != cpu::CpuExecutorStatus::ok) {
            error_message =
              std::string("Q4_H128 packed activation executor failed: ") +
              cpu::cpu_executor_status_name(status) + ".";
            return false;
          }
        } else if (!cpu::q4_h128_prepare_activations_4(
                     inputs.data(), packed.data(), packed_vector_count, cols,
                     w.q4_h128_sign_seed, w.q8_0_backend, w.q4_h128_signs.data())) {
          error_message = "Q4_H128 packed activation preparation failed.";
          return false;
        }
      } else {
        cpu::q8_0_quantize_vectors_4(
          quantization_inputs, packed.data(), packed_vector_count, blocks_per_row,
          w.q8_0_backend);
      }
      probe.prepared();
      if (cpu_context->executor != nullptr) {
        PackedQ4PrefillJob job{
          w.packed_q4_0_blocks.data(), packed.data(), out.data(),
          packed_vector_count, blocks_per_row, output_stride, w.q8_0_backend, w.q4_dot4,
        };
        const cpu::CpuExecutorStatus status =
          cpu_context->executor->parallel_for_rows(
            rows / cpu::q4_0_packed_rows, run_packed_q4_prefill_tiles, &job, probe.executor_timing());
        if (status != cpu::CpuExecutorStatus::ok) {
          error_message = std::string("Packed Q4_0 CPU batch executor failed: ") +
            cpu::cpu_executor_status_name(status) + ".";
          return false;
        }
      } else {
        (w.q4_dot4 ? cpu::q4_dot4_matmul : cpu::q4_0_packed_matmul_q8_0)(
          w.packed_q4_0_blocks.data(), packed.data(), out.data(), rows,
          packed_vector_count, blocks_per_row, output_stride, w.q8_0_backend);
      }
    }

    const std::size_t tail_vector_count = batch_size - packed_vector_count;
    if (tail_vector_count != 0) {
      CpuDecodeProbe probe(cpu_context, "q4-matvec-tail", rows, cols, tail_vector_count);
      auto & prepared = cpu_context->prepared_q4_input;
      prepared.resize(tail_vector_count * blocks_per_row);
      for (std::size_t token = 0; token < tail_vector_count; ++token) {
        if (!prepare_q4_decode_activation(w,
              inputs.data() + (packed_vector_count + token) * cols,
              cols, prepared.data() + token * blocks_per_row, error_message)) {
          return false;
        }
      }
      probe.prepared();
      float * tail_output = out.data() + packed_vector_count * output_stride;
      for (std::size_t token = 0; token < tail_vector_count; ++token) {
        const cpu::Q8_0BlockX1 * tail_vector =
          prepared.data() + token * blocks_per_row;
        float * token_output = tail_output + token * output_stride;
        if (cpu_context->executor != nullptr) {
          PackedQ4MatvecJob job{
            w.packed_q4_0_blocks.data(), tail_vector, token_output,
            blocks_per_row, w.q8_0_backend, w.q4_dot4,
          };
          const cpu::CpuExecutorStatus status =
            cpu_context->executor->parallel_for_rows(
              rows / cpu::q4_0_packed_rows,
              run_packed_q4_matvec_tiles,
              &job, probe.executor_timing());
          if (status != cpu::CpuExecutorStatus::ok) {
            error_message = std::string("Packed Q4_0 CPU batch tail executor failed: ") +
              cpu::cpu_executor_status_name(status) + ".";
            return false;
          }
        } else {
          (w.q4_dot4 ? cpu::q4_dot4_matvec : cpu::q4_0_packed_matvec_prepared_q8_0)(
            w.packed_q4_0_blocks.data(), tail_vector, token_output, rows,
            blocks_per_row, w.q8_0_backend);
        }
      }
    }
    return run_q8_gate_tail(cpu_context, w, inputs.data(), batch_size, out.data(), error_message);
  }

  std::vector<cpu::Q8_0Block> & quantized = cpu_context->quantized_batch;
  quantized.resize(batch_size * blocks_per_row);
  std::vector<float> & quantized_scales = cpu_context->quantized_batch_scales;
  quantized_scales.resize(quantized.size());
  for (std::size_t token = 0; token < batch_size; ++token) {
    cpu::q8_0_quantize_with_scales(
      quantization_inputs + token * cols,
      quantized.data() + token * blocks_per_row,
      quantized_scales.data() + token * blocks_per_row,
      blocks_per_row,
      w.q8_0_backend);
  }
  if (cpu_context->executor != nullptr) {
    const cpu::CpuExecutorStatus status = cpu_context->executor->q8_0_matmul(
          w.q8_0_blocks.data(), quantized.data(), out.data(), rows, batch_size,
          blocks_per_row, w.q8_0_backend, quantized_scales.data(), w.q8_0_scales.data());
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Quantized CPU batch executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    cpu::q8_0_matmul(
      w.q8_0_blocks.data(), quantized.data(), out.data(), rows, batch_size,
      blocks_per_row, rows, w.q8_0_backend,
      quantized_scales.data(), w.q8_0_scales.data());
  }
  return true;
}

void rms_norm_qwen3next(
  const std::vector<float> & x,
  const TensorData & weight,
  const float eps,
  std::vector<float> & out) {
  out.resize(x.size());
  cpu::rms_norm_f32(
    x.data(), weight.data.data(), out.data(), 1, x.size(), eps, 1.0F);
}

void rms_norm_per_head_qwen3next(
  std::span<const float> x,
  const int num_heads,
  const int head_dim,
  const TensorData & weight,
  const float eps,
  std::vector<float> & out) {
  out.resize(x.size());
  cpu::rms_norm_f32(
    x.data(), weight.data.data(), out.data(), static_cast<std::size_t>(num_heads),
    static_cast<std::size_t>(head_dim), eps, 1.0F);
}

void l2_norm_per_head(
  std::span<float> x,
  const int num_heads,
  const int head_dim,
  const float eps = 1.0e-6f,
  const float output_scale = 1.0F) {
  cpu::l2_normalize_f32(
    x.data(), static_cast<std::size_t>(num_heads),
    static_cast<std::size_t>(head_dim), eps, output_scale);
}

bool build_runtime_dims(const ModelProfile & profile, RuntimeDims & dims, std::string & error_message) {
  dims.n_layers = profile.text.num_hidden_layers;
  dims.hidden = profile.text.hidden_size;
  dims.intermediate = profile.text.intermediate_size;
  dims.vocab_size = profile.text.vocab_size;

  dims.n_heads = profile.text.num_attention_heads;
  dims.n_kv_heads = profile.text.num_key_value_heads;
  dims.head_dim = profile.text.head_dim > 0 ? profile.text.head_dim : (dims.hidden / std::max(1, dims.n_heads));
  dims.rope_theta = profile.text.rope_theta;
  dims.rms_eps = profile.text.rms_norm_eps;

  dims.rope_dim = static_cast<int>(static_cast<float>(dims.head_dim) * profile.text.partial_rotary_factor);
  if ((dims.rope_dim % 2) != 0) {
    dims.rope_dim -= 1;
  }
  if (dims.rope_dim < 0) {
    dims.rope_dim = 0;
  }

  dims.linear_kernel = profile.text.linear_conv_kernel_dim;
  dims.linear_num_k_heads = profile.text.linear_num_key_heads;
  dims.linear_num_v_heads = profile.text.linear_num_value_heads;
  dims.linear_head_k_dim = profile.text.linear_key_head_dim;
  dims.linear_head_v_dim = profile.text.linear_value_head_dim;
  dims.linear_q_dim = dims.linear_num_k_heads * dims.linear_head_k_dim;
  dims.linear_v_dim = dims.linear_num_v_heads * dims.linear_head_v_dim;
  dims.linear_conv_channels = dims.linear_q_dim * 2 + dims.linear_v_dim;

  if (dims.n_layers <= 0 || dims.hidden <= 0 || dims.intermediate <= 0 || dims.vocab_size <= 0 || dims.n_heads <= 0 ||
      dims.n_kv_heads <= 0 || dims.head_dim <= 0 || dims.linear_kernel <= 1 || dims.linear_num_k_heads <= 0 ||
      dims.linear_num_v_heads <= 0 || dims.linear_head_k_dim <= 0 || dims.linear_head_v_dim <= 0) {
    error_message = "Model profile does not contain required runtime dimensions.";
    return false;
  }

  if (dims.n_heads % dims.n_kv_heads != 0) {
    error_message = "Invalid full-attention GQA ratio in profile.";
    return false;
  }
  if (dims.linear_num_v_heads % dims.linear_num_k_heads != 0) {
    error_message = "Invalid linear-attention grouped-head ratio in profile.";
    return false;
  }
  if (dims.linear_head_k_dim != dims.linear_head_v_dim) {
    error_message = "Reference linear path requires linear_key_head_dim == linear_value_head_dim.";
    return false;
  }
  if (static_cast<int>(profile.fingerprint.attention_schedule.size()) != dims.n_layers) {
    error_message = "Attention schedule length does not match num_hidden_layers.";
    return false;
  }

  return true;
}

bool load_model_weights_from_q4_h128(
  const std::string & artifact_path,
  const RuntimeDims & dims,
  const ModelProfile & profile,
  const cpu::Q8_0Backend backend,
  const int cpu_threads,
  ModelWeights & weights,
  std::string & error_message) {
  Q4H128ArtifactReader reader;
  if (!reader.open(artifact_path, error_message)) {
    return false;
  }
  const auto * embedding_info = reader.find_tensor("model.language_model.embed_tokens.weight");
  const bool cpu_packed = embedding_info != nullptr &&
    (q4_h128_encoding_cpu_packed(embedding_info->encoding) || embedding_info->encoding == Q4H128TensorEncoding::q8_0);
  if (!embedding_info || (!q4_h128_encoding_dot4(embedding_info->encoding) && embedding_info->encoding != Q4H128TensorEncoding::q8_0)) {
    error_message="Only CPU-ready H128/Q4-G32-DOT4 artifacts are supported.";return false;
  }
  const Q4H128ArtifactMetadata & metadata = reader.metadata();
  if (metadata.num_hidden_layers != static_cast<std::uint32_t>(dims.n_layers) ||
      metadata.hidden_size != static_cast<std::uint32_t>(dims.hidden) ||
      metadata.intermediate_size != static_cast<std::uint32_t>(dims.intermediate) ||
      metadata.vocabulary_size != static_cast<std::uint32_t>(dims.vocab_size) ||
      metadata.sign_seed == 0) {
    error_message = "Q4_H128 artifact model fingerprint does not match the model profile.";
    return false;
  }

  const auto load_q4 = [&](const std::string & name, const int rows,
                           const int cols, TensorData & output,
                           const bool retain_scales = true) {
    return load_q4_h128_quantized_checked(
      reader, name, rows, cols, true, backend, output,
      error_message, retain_scales);
  };
  const auto load_f32 = [&](const std::string & name,
                            const std::initializer_list<std::int64_t> shape,
                            TensorData & output) {
    return load_q4_h128_f32_checked(
      reader, name, shape, output, error_message);
  };

  if (!load_q4_h128_quantized_checked(
        reader, "model.language_model.embed_tokens.weight",
        dims.vocab_size, dims.hidden, false, backend,
        weights.embed_tokens, error_message, false) ||
      !load_f32(
        "model.language_model.norm.weight", {dims.hidden},
        weights.final_norm)) {
    return false;
  }

  weights.layers.resize(static_cast<std::size_t>(dims.n_layers));
  const int full_q_out = dims.n_heads * dims.head_dim * 2;
  const int full_kv_out = dims.n_kv_heads * dims.head_dim;
  const int full_o_in = dims.n_heads * dims.head_dim;
  for (int layer_index = 0; layer_index < dims.n_layers; ++layer_index) {
    LayerWeights & layer = weights.layers[static_cast<std::size_t>(layer_index)];
    const std::string base =
      "model.language_model.layers." + std::to_string(layer_index) + ".";
    layer.is_linear =
      profile.fingerprint.attention_schedule[static_cast<std::size_t>(layer_index)] ==
      AttentionBlock::linear;
    if (!load_f32(base + "input_layernorm.weight", {dims.hidden},
                  layer.input_layernorm) ||
        !load_f32(base + "post_attention_layernorm.weight", {dims.hidden},
                  layer.post_attention_layernorm) ||
        !(cpu_packed
          ? load_q4(base + "mlp.gate_up_proj.weight", 2 * dims.intermediate, dims.hidden,
                    layer.mlp_gate_up_cpu, false)
          : (load_q4(base + "mlp.gate_proj.weight", dims.intermediate, dims.hidden, layer.mlp_gate) &&
             load_q4(base + "mlp.up_proj.weight", dims.intermediate, dims.hidden, layer.mlp_up))) ||
        !load_q4(base + "mlp.down_proj.weight", dims.hidden, dims.intermediate,
                 layer.mlp_down)) {
      return false;
    }

    if (layer.is_linear) {
      const auto *gate_info = reader.find_tensor(base + "linear_attn.in_proj_ba.weight");
      const int gate_rows = gate_info ? 2*dims.linear_num_v_heads : 0;
      const int combined_rows = dims.linear_conv_channels + dims.linear_v_dim +
        2 * dims.linear_num_v_heads - gate_rows;
      if ((cpu_packed
            ? !load_q4(base + "linear_attn.in_proj_all.weight", combined_rows, dims.hidden,
                            layer.linear.in_proj_all_cpu, false)
            : (!load_q4(base + "linear_attn.in_proj_qkv.weight",
                        dims.linear_conv_channels, dims.hidden,
                        layer.linear.in_proj_qkv) ||
               !load_q4(base + "linear_attn.in_proj_z.weight",
                        dims.linear_v_dim, dims.hidden,
                        layer.linear.in_proj_z) ||
               !load_q4(base + "linear_attn.in_proj_b.weight",
                        dims.linear_num_v_heads, dims.hidden,
                        layer.linear.in_proj_b) ||
               !load_q4(base + "linear_attn.in_proj_a.weight",
                        dims.linear_num_v_heads, dims.hidden,
                        layer.linear.in_proj_a))) ||
          !load_f32(base + "linear_attn.conv1d.weight",
                    {dims.linear_conv_channels, 1, dims.linear_kernel},
                    layer.linear.conv1d) ||
          !load_q4(base + "linear_attn.out_proj.weight",
                   dims.hidden, dims.linear_v_dim,
                   layer.linear.out_proj) ||
          !load_f32(base + "linear_attn.norm.weight",
                    {dims.linear_head_v_dim}, layer.linear.norm) ||
          !load_f32(base + "linear_attn.A_log",
                    {dims.linear_num_v_heads}, layer.linear.a_log) ||
          !load_f32(base + "linear_attn.dt_bias",
                    {dims.linear_num_v_heads}, layer.linear.dt_bias)) {
        return false;
      }
      if (gate_info) {
        TensorData gate;
        if (!cpu_packed || gate_info->encoding != Q4H128TensorEncoding::q8_0 ||
            !load_q4_h128_quantized_checked(reader, base+"linear_attn.in_proj_ba.weight",
              gate_rows, dims.hidden, false, backend, gate, error_message)) {
          if (error_message.empty()) error_message = "Invalid Q8 recurrent gate group";
          return false;
        }
        auto &projection = layer.linear.in_proj_all_cpu;
        projection.q8_gate_rows = static_cast<std::size_t>(gate_rows);
        projection.q8_gate_blocks = std::move(gate.q8_0_blocks);
        projection.q8_gate_scales = std::move(gate.q8_0_scales);
        projection.shape[0] += gate_rows;
      }
      layer.linear.conv1d.shape = {dims.linear_conv_channels, dims.linear_kernel};
      layer.linear.ssm_a.resize(static_cast<std::size_t>(dims.linear_num_v_heads));
      for (int index = 0; index < dims.linear_num_v_heads; ++index) {
        layer.linear.ssm_a[static_cast<std::size_t>(index)] =
          -std::exp(layer.linear.a_log.data[static_cast<std::size_t>(index)]);
      }
      pack_conv1d_kernel_major(layer.linear, dims);
    } else {
      if ((cpu_packed
            ? !load_q4(base + "self_attn.qkv_proj.weight", full_q_out + 2 * full_kv_out,
                       dims.hidden, layer.full.qkv_proj_cpu, false)
            : (!load_q4(base + "self_attn.q_proj.weight", full_q_out, dims.hidden,
                        layer.full.q_proj) ||
               !load_q4(base + "self_attn.k_proj.weight", full_kv_out, dims.hidden,
                        layer.full.k_proj) ||
               !load_q4(base + "self_attn.v_proj.weight", full_kv_out, dims.hidden,
                        layer.full.v_proj))) ||
          !load_q4(base + "self_attn.o_proj.weight", dims.hidden, full_o_in,
                   layer.full.o_proj) ||
          !load_f32(base + "self_attn.q_norm.weight", {dims.head_dim},
                    layer.full.q_norm) ||
          !load_f32(base + "self_attn.k_norm.weight", {dims.head_dim},
                    layer.full.k_norm)) {
        return false;
      }
    }

    if (!cpu_packed) {
      if (!pack_quantized_row_concat(
            {&layer.mlp_gate, &layer.mlp_up},
            layer.mlp_gate_up_cpu, error_message)) {
        return false;
      }
      if (layer.is_linear) {
        if (!pack_quantized_row_concat(
              {&layer.linear.in_proj_qkv, &layer.linear.in_proj_z,
               &layer.linear.in_proj_b, &layer.linear.in_proj_a},
              layer.linear.in_proj_all_cpu, error_message)) {
          return false;
        }
      } else if (!pack_quantized_row_concat(
                   {&layer.full.q_proj, &layer.full.k_proj, &layer.full.v_proj},
                   layer.full.qkv_proj_cpu, error_message)) {
        return false;
      }
    }
  }

  const auto release_canonical_q4 = [](TensorData & tensor) {
    if (!tensor.packed_q4_0_blocks.empty()) {
      std::vector<cpu::Q4_0Block>().swap(tensor.q4_0_blocks);
      std::vector<float>().swap(tensor.q4_0_scales);
    }
  };
  release_canonical_q4(weights.embed_tokens);
  for (LayerWeights & layer : weights.layers) {
    release_canonical_q4(layer.mlp_gate_up_cpu);
    release_canonical_q4(layer.mlp_down);
    if (layer.is_linear) {
      release_canonical_q4(layer.linear.in_proj_all_cpu);
      release_canonical_q4(layer.linear.out_proj);
    } else {
      release_canonical_q4(layer.full.qkv_proj_cpu);
      release_canonical_q4(layer.full.o_proj);
    }
  }
  return true;
}

#include "gguf_weights.inl"
