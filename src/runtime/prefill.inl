namespace {

void rms_norm_qwen3next_batch(
  const std::vector<float> & input,
  const std::size_t batch_size,
  const std::size_t width,
  const TensorData & weight,
  const float eps,
  std::vector<float> & output) {
  output.resize(input.size());
  cpu::rms_norm_f32(
    input.data(), weight.data.data(), output.data(), batch_size, width,
    eps, 1.0F);
}

struct GatedDeltaNetBatchCpuJob {
  float * state = nullptr;
  const float * q = nullptr;
  const float * k = nullptr;
  const float * v = nullptr;
  const float * alpha = nullptr;
  const float * beta = nullptr;
  float * output = nullptr;
  std::size_t batch_size = 0;
  std::size_t head_count = 0;
  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
};

struct CausalConvBatchCpuJob {
  float * state = nullptr;
  std::size_t ring_index = 0;
  const float * input = nullptr;
  const float * weights = nullptr;
  float * output = nullptr;
  std::size_t batch_size = 0;
  std::size_t input_stride = 0;
  std::size_t channel_count = 0;
  std::size_t kernel_size = 0;
  cpu::Q8_0Backend backend = cpu::Q8_0Backend::auto_select;
};

void run_causal_conv_batch_cpu_channels(
  void * opaque_context,
  const std::size_t channel_begin,
  const std::size_t channel_end) noexcept {
  auto & job = *static_cast<CausalConvBatchCpuJob *>(opaque_context);
  cpu::causal_conv1d_silu_f32(
    job.state, job.ring_index, job.input, job.input_stride, job.weights,
    job.output, job.batch_size, job.channel_count, job.kernel_size,
    channel_begin, channel_end, job.backend);
}

void run_gated_delta_net_batch_cpu_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<GatedDeltaNetBatchCpuJob *>(opaque_context);
  cpu::gated_delta_net_update_batch_rows(
    job.state,
    job.q,
    job.k,
    job.v,
    job.alpha,
    job.beta,
    job.output,
    job.batch_size,
    job.head_count,
    job.key_dim,
    job.value_dim,
    row_begin,
    row_end,
    job.backend);
}

bool run_linear_attention_batch_cpu_q8(CpuExecutionContext *cpu_context,
  const LayerWeights & layer,
  const RuntimeDims & dims,
  LinearAttentionState & state,
  const std::vector<float> & input,
  const std::size_t batch_size,
  const int position_start,
  std::vector<float> & output,
  std::string & error_message) {
  if (!layer.linear.in_proj_all_cpu.is_cpu_quantized() || !layer.linear.out_proj.is_cpu_quantized()) {
    error_message = "Batched CPU linear attention requires packed Q8_0 projections.";
    return false;
  }

  const std::size_t conv_channels = static_cast<std::size_t>(dims.linear_conv_channels);
  const std::size_t value_width = static_cast<std::size_t>(dims.linear_v_dim);
  const std::size_t head_count = static_cast<std::size_t>(dims.linear_num_v_heads);
  const std::size_t key_dim = static_cast<std::size_t>(dims.linear_head_k_dim);
  const std::size_t value_dim = static_cast<std::size_t>(dims.linear_head_v_dim);
  const std::size_t q_width = static_cast<std::size_t>(dims.linear_q_dim);
  const std::size_t projection_width = conv_channels + value_width + 2 * head_count;
  const std::size_t expected_state = head_count * value_dim * key_dim;
  if (dims.linear_num_k_heads != dims.linear_num_v_heads || key_dim != value_dim ||
      state.recurrent_state.size() != expected_state ||
      state.conv_state.size() !=
        static_cast<std::size_t>(dims.linear_kernel - 1) * conv_channels ||
      layer.linear.conv1d_kernel_major.size() !=
        static_cast<std::size_t>(dims.linear_kernel) * conv_channels) {
    error_message = "Batched CPU DeltaNet state dimensions are unsupported or inconsistent.";
    return false;
  }

  CpuExecutionContext * const runtime = cpu_context;
  CpuPrefillWorkspace::Linear fallback;
  auto & scratch = runtime != nullptr ? runtime->prefill.linear : fallback;
  const bool fine = runtime && runtime->stages;
  CpuPrefillStage stage;
  std::chrono::steady_clock::time_point mark;
  if(fine) { stage.kind="linear"; stage.position=position_start; stage.tokens=batch_size; mark=std::chrono::steady_clock::now(); }
  auto & projected = scratch.projected;
  if (!matmul_2d_quantized_batch(cpu_context,
        layer.linear.in_proj_all_cpu, input, batch_size, projected, error_message)) {
    return false;
  }
  if(fine) {stage.projection_ms=elapsed_ms(mark);mark=std::chrono::steady_clock::now();}
  if (projected.size() != batch_size * projection_width) {
    error_message = "Batched linear-attention projection output size mismatch.";
    return false;
  }

  auto & gated = scratch.gated;
  gated.resize(batch_size * value_width);
  auto & conv_batch = scratch.conv_batch;
  conv_batch.resize(batch_size * conv_channels);
  auto & q_batch = scratch.q_batch;
  q_batch.resize(batch_size * q_width);
  auto & k_batch = scratch.k_batch;
  k_batch.resize(batch_size * q_width);
  auto & v_batch = scratch.v_batch;
  v_batch.resize(batch_size * value_width);
  auto & alpha_batch = scratch.alpha_batch;
  alpha_batch.resize(batch_size * head_count);
  auto & beta_batch = scratch.beta_batch;
  beta_batch.resize(batch_size * head_count);
  auto & core_batch = scratch.core_batch;
  core_batch.resize(batch_size * value_width);
  const float q_scale = 1.0F / std::sqrt(static_cast<float>(dims.linear_head_k_dim));
  CausalConvBatchCpuJob conv_job{
    state.conv_state.data(),
    state.conv_ring_index,
    projected.data(),
    layer.linear.conv1d_kernel_major.data(),
    conv_batch.data(),
    batch_size,
    projection_width,
    conv_channels,
    static_cast<std::size_t>(dims.linear_kernel),
    layer.linear.out_proj.q8_0_backend,
  };
  if (runtime != nullptr && runtime->executor != nullptr) {
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      conv_channels, run_causal_conv_batch_cpu_channels, &conv_job);
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Batched causal-conv CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    run_causal_conv_batch_cpu_channels(&conv_job, 0, conv_channels);
  }
  const std::size_t conv_history = static_cast<std::size_t>(dims.linear_kernel - 1);
  if (conv_history != 0) {
    state.conv_ring_index =
      (state.conv_ring_index + batch_size % conv_history) % conv_history;
  }

  for (std::size_t token = 0; token < batch_size; ++token) {
    const float * projection = projected.data() + token * projection_width;
    const float * mixed_qkv = projection;
    const float * z = mixed_qkv + conv_channels;
    const float * b = z + value_width;
    const float * a = b + head_count;

    for (std::size_t head = 0; head < head_count; ++head) {
      beta_batch[head * batch_size + token] = sigmoidf_stable(b[head]);
      const float pre_gate = softplusf_stable(a[head] + layer.linear.dt_bias.data[head]);
      alpha_batch[head * batch_size + token] = std::exp(pre_gate * layer.linear.ssm_a[head]);
    }

    float * conv_out = conv_batch.data() + token * conv_channels;
    std::span<float> q(conv_out, q_width);
    std::span<float> k(conv_out + q_width, q_width);
    const float * v = conv_out + 2 * q_width;
    l2_norm_per_head(
      q, dims.linear_num_k_heads, dims.linear_head_k_dim, 1.0e-6F, q_scale);
    l2_norm_per_head(k, dims.linear_num_k_heads, dims.linear_head_k_dim);
    for (std::size_t head = 0; head < head_count; ++head) {
      const std::size_t head_token = head * batch_size + token;
      std::memcpy(
        q_batch.data() + head_token * key_dim,
        q.data() + head * key_dim,
        key_dim * sizeof(float));
      std::memcpy(
        k_batch.data() + head_token * key_dim,
        k.data() + head * key_dim,
        key_dim * sizeof(float));
      std::memcpy(
        v_batch.data() + head_token * value_dim,
        v + head * value_dim,
        value_dim * sizeof(float));
    }
  }

  if(fine) {stage.prepare_ms=elapsed_ms(mark);mark=std::chrono::steady_clock::now();stage.kernel="deltanet";stage.participants=runtime->executor ? std::min(head_count,runtime->executor->thread_count()) : 1;}
  if (runtime != nullptr && runtime->executor != nullptr) {
    GatedDeltaNetBatchCpuJob job{
      state.recurrent_state.data(),
      q_batch.data(),
      k_batch.data(),
      v_batch.data(),
      alpha_batch.data(),
      beta_batch.data(),
      core_batch.data(),
      batch_size,
      head_count,
      key_dim,
      value_dim,
      layer.linear.out_proj.q8_0_backend,
    };
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      head_count, run_gated_delta_net_batch_cpu_rows, &job);
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Batched DeltaNet CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    cpu::gated_delta_net_update_batch_rows(
      state.recurrent_state.data(),
      q_batch.data(),
      k_batch.data(),
      v_batch.data(),
      alpha_batch.data(),
      beta_batch.data(),
      core_batch.data(),
      batch_size,
      head_count,
      key_dim,
      value_dim,
      0,
      head_count,
      layer.linear.out_proj.q8_0_backend);
  }

  if(fine) {stage.attention_wall_ms=elapsed_ms(mark);mark=std::chrono::steady_clock::now();}
  for (std::size_t token = 0; token < batch_size; ++token) {
    const float * projection = projected.data() + token * projection_width;
    const float * z = projection + conv_channels;
    const float * core_out = core_batch.data() + token * value_width;
    float * gated_token = gated.data() + token * value_width;
    cpu::rms_norm_f32(
      core_out, layer.linear.norm.data.data(), gated_token,
      head_count, value_dim, dims.rms_eps, 0.0F,
      layer.linear.out_proj.q8_0_backend);
    cpu::silu_mul_f32(
      z, gated_token, gated_token, value_width,
      layer.linear.out_proj.q8_0_backend);
  }

  const bool ok = matmul_2d_quantized_batch(cpu_context,
    layer.linear.out_proj, gated, batch_size, output, error_message);
  if(fine) {stage.output_ms=elapsed_ms(mark);runtime->stages->push_back(stage);}
  return ok;
}

bool run_full_attention_batch_cpu_q8(CpuExecutionContext *cpu_context,
  const LayerWeights & layer,
  const RuntimeDims & dims,
  FullAttentionState & state,
  const std::vector<float> & input,
  const std::size_t batch_size,
  const int position_start,
  const float * rope_cosine,
  const float * rope_sine,
  std::vector<float> & output,
  std::string & error_message) {
  if (!layer.full.qkv_proj_cpu.is_cpu_quantized() || !layer.full.o_proj.is_cpu_quantized()) {
    error_message = "Batched CPU full attention requires packed Q8_0 projections.";
    return false;
  }
  const std::size_t query_width = static_cast<std::size_t>(dims.n_heads * dims.head_dim);
  const std::size_t q_full_width = 2 * query_width;
  const std::size_t kv_width = static_cast<std::size_t>(dims.n_kv_heads * dims.head_dim);
  const std::size_t projection_width = q_full_width + 2 * kv_width;
  CpuExecutionContext * const runtime = cpu_context;
  CpuPrefillWorkspace::Full fallback;
  auto & scratch = runtime != nullptr ? runtime->prefill.full : fallback;
  const bool fine = runtime && runtime->stages;
  CpuPrefillStage stage;
  std::chrono::steady_clock::time_point mark;
  if(fine) { stage.kind="full"; stage.tokens=batch_size; mark=std::chrono::steady_clock::now(); }
  auto & projected = scratch.projected;
  if (!matmul_2d_quantized_batch(cpu_context,
        layer.full.qkv_proj_cpu, input, batch_size, projected, error_message)) {
    return false;
  }
  if(fine) {stage.projection_ms=elapsed_ms(mark);mark=std::chrono::steady_clock::now();}
  if (projected.size() != batch_size * projection_width) {
    error_message = "Batched full-attention projection output size mismatch.";
    return false;
  }

  auto & attention = scratch.attention;
  attention.resize(batch_size * query_width);
  auto & query_batch = scratch.query_batch;
  query_batch.resize(batch_size * query_width);
  auto & gate_batch = scratch.gate_batch;
  gate_batch.resize(batch_size * query_width);
  auto & q = scratch.q;
  q.resize(query_width);
  auto & q_normed = scratch.q_normed;
  auto & k_normed = scratch.k_normed;
  const int q_span = 2 * dims.head_dim;
  const float attention_scale = 1.0F / std::sqrt(static_cast<float>(dims.head_dim));
  const int rope_half = dims.rope_dim / 2;

  for (std::size_t token = 0; token < batch_size; ++token) {
    const int position = position_start + static_cast<int>(token);
    const float * projection = projected.data() + token * projection_width;
    const float * q_full = projection;
    const float * k_source = q_full + q_full_width;
    const float * v_flat = k_source + kv_width;
    for (int head = 0; head < dims.n_heads; ++head) {
      const std::size_t source = static_cast<std::size_t>(head * q_span);
      const std::size_t destination = static_cast<std::size_t>(head * dims.head_dim);
      std::memcpy(
        q.data() + destination,
        q_full + source,
        static_cast<std::size_t>(dims.head_dim) * sizeof(float));
      std::memcpy(
        gate_batch.data() + token * query_width + destination,
        q_full + source + static_cast<std::size_t>(dims.head_dim),
        static_cast<std::size_t>(dims.head_dim) * sizeof(float));
    }
    rms_norm_per_head_qwen3next(
      q, dims.n_heads, dims.head_dim, layer.full.q_norm, dims.rms_eps, q_normed);
    rms_norm_per_head_qwen3next(
      std::span<const float>(k_source, kv_width), dims.n_kv_heads, dims.head_dim, layer.full.k_norm, dims.rms_eps, k_normed);
    const float * token_cosine =
      rope_cosine + token * static_cast<std::size_t>(rope_half);
    const float * token_sine =
      rope_sine + token * static_cast<std::size_t>(rope_half);
    cpu::rope_f32(
      q_normed.data(), static_cast<std::size_t>(dims.n_heads),
      static_cast<std::size_t>(dims.head_dim), static_cast<std::size_t>(dims.rope_dim),
      token_cosine, token_sine, layer.full.o_proj.q8_0_backend);
    cpu::rope_f32(
      k_normed.data(), static_cast<std::size_t>(dims.n_kv_heads),
      static_cast<std::size_t>(dims.head_dim), static_cast<std::size_t>(dims.rope_dim),
      token_cosine, token_sine, layer.full.o_proj.q8_0_backend);

    std::memcpy(
      query_batch.data() + token * query_width,
      q_normed.data(),
      query_width * sizeof(float));

    if (state.pages.width()) {
      if (state.pages.size()!=static_cast<std::size_t>(position)) {
        error_message="Paged KV append position mismatch.";return false;
      }
      state.pages.append(k_normed.data(),v_flat,layer.full.o_proj.q8_0_backend);
    }
    if (!state.k_cache.empty()) {
      std::memcpy(
        state.k_cache.data() + static_cast<std::size_t>(position) * kv_width,
        k_normed.data(),
        kv_width * sizeof(float));
      std::memcpy(
        state.v_cache.data() + static_cast<std::size_t>(position) * kv_width,
        v_flat,
        kv_width * sizeof(float));
    }
    if (!state.k_cache_f16.empty() && !state.v_cache_f16.empty()) {
      cpu::attention_cache_store_f16(
        k_normed.data(),
        state.k_cache_f16.data() + static_cast<std::size_t>(position) * kv_width,
        kv_width,
        layer.full.o_proj.q8_0_backend);
      cpu::attention_cache_store_f16(
        v_flat,
        state.v_cache_f16.data() + static_cast<std::size_t>(position) * kv_width,
        kv_width,
        layer.full.o_proj.q8_0_backend);
    }
  }

  const cpu::AttentionKvRows page_rows{state.pages.keys(),state.pages.values()};
  // A short suffix after prefix restore must retain the same arithmetic as
  // those token positions in a larger cold-prefill tile.
  const bool tiled = runtime && runtime->tiled_attention &&
    (!runtime->automatic_attention || position_start + batch_size > 128);
  if(runtime && runtime->attention_kernel_result &&
     ((tiled && !runtime->attention_tiles_used) || (!tiled && !runtime->attention_rows_used))) {
    runtime->attention_tiles_used |= tiled;
    runtime->attention_rows_used |= !tiled;
    *runtime->attention_kernel_result = runtime->attention_tiles_used ?
      std::string(runtime->attention_rows_used ? "mixed:rows+" : "") + cpu::tiled_attention_kernel(runtime->attention_backend) : "rows";
  }
  if(fine) {
    stage.prepare_ms=elapsed_ms(mark);mark=std::chrono::steady_clock::now();
    stage.position=position_start;
    stage.query_key_pairs=static_cast<std::uint64_t>(dims.n_heads)*batch_size*(2ULL*position_start+batch_size+1)/2;
    stage.kernel=tiled ? cpu::tiled_attention_kernel(runtime->attention_backend) : "rows";
  }
  if (tiled) {
    cpu::TiledAttention args{
      query_batch.data(), gate_batch.data(), state.k_cache.data(), state.v_cache.data(),
      state.k_cache_f16.empty() ? nullptr : state.k_cache_f16.data(),
      state.v_cache_f16.empty() ? nullptr : state.v_cache_f16.data(), attention.data(),
      batch_size, static_cast<std::size_t>(position_start), query_width, kv_width,
      dims.n_heads, dims.n_kv_heads, dims.head_dim, attention_scale,
      runtime->query_tile, runtime->kv_tile, runtime->attention_gqa};
    args.pages=state.pages.width()?&page_rows:nullptr;
    if(runtime->automatic_attention && runtime->executor && args.share_gqa) {
      // Keep enough independent tasks for the actual pool, including short tails.
      while(args.query_tile>4 && cpu::tiled_attention_tasks(args)<runtime->executor->thread_count()) args.query_tile/=2;
      if(cpu::tiled_attention_tasks(args)<runtime->executor->thread_count()) args.share_gqa=false;
    }
    const auto tasks=cpu::tiled_attention_tasks(args);
    const auto partitions=runtime->executor ? std::min(tasks,runtime->executor->thread_count()) : 1;
    const auto stride=cpu::tiled_attention_scratch_floats(args);
    scratch.tiled_scratch.resize(partitions*stride);
    TiledAttentionCpuJob job{args,scratch.tiled_scratch.data(),partitions,tasks,stride,runtime->attention_backend};
    if(fine) {
      stage.query_tile=args.query_tile; stage.kv_tile=args.kv_tile; stage.shared_gqa=args.share_gqa;
      stage.participants=partitions;
      scratch.tile_times.assign(partitions,{});job.times=scratch.tile_times.data();
    }
    if(runtime->executor) {
      if(runtime->executor->parallel_for_rows(partitions,run_tiled_attention_cpu,&job)!=cpu::CpuExecutorStatus::ok) {
        error_message="Tiled attention executor failed."; return false;
      }
    } else run_tiled_attention_cpu(&job,0,partitions);
    if(fine) for(const auto &t:scratch.tile_times) {
      stage.pack_worker_ms+=t.pack_ms; stage.qk_worker_ms+=t.qk_ms;
      stage.softmax_worker_ms+=t.softmax_ms; stage.pv_worker_ms+=t.pv_ms;
    }
  } else {
  const std::size_t attention_rows =
    batch_size * static_cast<std::size_t>(dims.n_heads);
  const std::size_t context_stride =
    static_cast<std::size_t>(position_start) + batch_size;
  auto & scores = scratch.scores;
  const std::size_t score_partitions = runtime != nullptr && runtime->executor != nullptr
    ? std::min(attention_rows, runtime->executor->thread_count()) : 1;
  if(fine) stage.participants=score_partitions;
  scores.resize(score_partitions * context_stride);
  FullAttentionBatchCpuJob job{
    query_batch.data(),
    gate_batch.data(),
    state.k_cache.data(),
    state.v_cache.data(),
    state.k_cache_f16.empty() ? nullptr : state.k_cache_f16.data(),
    state.v_cache_f16.empty() ? nullptr : state.v_cache_f16.data(),
    scores.data(),
    attention.data(),
    context_stride,
    query_width,
    kv_width,
    position_start,
    dims.n_heads,
    dims.n_kv_heads,
    dims.head_dim,
    attention_scale,
    layer.full.o_proj.q8_0_backend,
    score_partitions,
    attention_rows,
    state.pages.width()?&page_rows:nullptr,
  };
  if (runtime != nullptr && runtime->executor != nullptr) {
    const cpu::CpuExecutorStatus status = runtime->executor->parallel_for_rows(
      score_partitions, run_full_attention_batch_cpu_rows, &job);
    if (status != cpu::CpuExecutorStatus::ok) {
      error_message = std::string("Batched full-attention CPU executor failed: ") +
        cpu::cpu_executor_status_name(status) + ".";
      return false;
    }
  } else {
    run_full_attention_batch_cpu_rows(&job, 0, score_partitions);
  }
  }

  if(fine) {stage.attention_wall_ms=elapsed_ms(mark);mark=std::chrono::steady_clock::now();}
  const bool ok = matmul_2d_quantized_batch(cpu_context,
    layer.full.o_proj, attention, batch_size, output, error_message);
  if(fine) {stage.output_ms=elapsed_ms(mark);runtime->stages->push_back(stage);}
  return ok;
}

bool run_forward_cpu_q8_batch(CpuExecutionContext *cpu_context,
  const ModelWeights & weights,
  const RuntimeDims & dims,
  ModelState & state,
  const std::int32_t * token_ids,
  const std::size_t batch_size,
  const int position_start,
  const bool compute_next_logits,
  std::vector<float> & next_logits,
  CpuGreedySamplingState * greedy_sampling,
  DecodeProfilingAccumulator * profiling,
  std::string & error_message,
  std::vector<float> * normalized_output = nullptr) {
  if (cpu_context == nullptr || token_ids == nullptr || batch_size == 0) {
    error_message = "Batched CPU Q8_0 forward received an invalid runtime or empty batch.";
    return false;
  }
  const std::size_t hidden = static_cast<std::size_t>(dims.hidden);
  const std::size_t intermediate = static_cast<std::size_t>(dims.intermediate);
  const std::size_t embedding_blocks = hidden / cpu::q8_0_values_per_block;
  if (!weights.embed_tokens.is_cpu_quantized() || (hidden % cpu::q8_0_values_per_block) != 0) {
    error_message = "Batched CPU quantized forward requires Q4_0/Q8_0 embeddings aligned to 32 values.";
    return false;
  }
  if (profiling != nullptr) {
    profiling->forward_pass_tokens += static_cast<int>(batch_size);
  }

  const auto embedding_start = std::chrono::steady_clock::now();
  auto & scratch = cpu_context->prefill.forward;
  auto & x = scratch.x;
  x.resize(batch_size * hidden);
  for (std::size_t token = 0; token < batch_size; ++token) {
    const int token_id = token_ids[token];
    if (token_id < 0 || token_id >= dims.vocab_size) {
      error_message = "Batched CPU prefill token id is outside the vocabulary.";
      return false;
    }
    if (weights.embed_tokens.is_q4_0()) {
      (weights.embed_tokens.q4_dot4 ? cpu::q4_dot4_dequantize_row : cpu::q4_0_packed_dequantize_row)(
        weights.embed_tokens.packed_q4_0_blocks.data(),
        static_cast<std::size_t>(token_id),
        x.data() + token * hidden, embedding_blocks);
    } else {
      const std::size_t block_offset =
        static_cast<std::size_t>(token_id) * embedding_blocks;
      cpu::q8_0_dequantize(
        weights.embed_tokens.q8_0_blocks.data() + block_offset,
        x.data() + token * hidden, embedding_blocks,
        weights.embed_tokens.q8_0_backend);
    }
  }
  if (profiling != nullptr) {
    profiling->embedding_ms += elapsed_ms(embedding_start);
  }

  auto & normed = scratch.normed;
  auto & attention = scratch.attention;
  auto & residual = scratch.residual;
  residual.resize(batch_size * hidden);
  auto & post_norm = scratch.post_norm;
  auto & gate_up = scratch.gate_up;
  auto & mlp_hidden = scratch.mlp_hidden;
  mlp_hidden.resize(batch_size * intermediate);
  auto & mlp_output = scratch.mlp_output;
  int full_index = 0;
  int linear_index = 0;

  for (int layer_index = 0; layer_index < dims.n_layers; ++layer_index) {
    const LayerWeights & layer = weights.layers[static_cast<std::size_t>(layer_index)];
    rms_norm_qwen3next_batch(
      x, batch_size, hidden, layer.input_layernorm, dims.rms_eps, normed);
    const auto attention_start = std::chrono::steady_clock::now();
    bool attention_ok = false;
    if (layer.is_linear) {
      attention_ok = run_linear_attention_batch_cpu_q8(cpu_context,
        layer,
        dims,
        state.linear_states[static_cast<std::size_t>(linear_index++)],
        normed,
        batch_size,
        position_start,
        attention,
        error_message);
    } else {
      attention_ok = run_full_attention_batch_cpu_q8(cpu_context,
        layer,
        dims,
        state.full_states[static_cast<std::size_t>(full_index++)],
        normed,
        batch_size,
        position_start,
        state.rope_cosine.data() +
          static_cast<std::size_t>(position_start) * static_cast<std::size_t>(dims.rope_dim / 2),
        state.rope_sine.data() +
          static_cast<std::size_t>(position_start) * static_cast<std::size_t>(dims.rope_dim / 2),
        attention,
        error_message);
    }
    if (!attention_ok) {
      return false;
    }
    if (profiling != nullptr) {
      profiling->attention_ms += elapsed_ms(attention_start);
    }

    const auto mlp_start = std::chrono::steady_clock::now();
    cpu::add_f32(x.data(), attention.data(), residual.data(), residual.size());
    rms_norm_qwen3next_batch(
      residual, batch_size, hidden, layer.post_attention_layernorm, dims.rms_eps, post_norm);
    if (!layer.mlp_gate_up_cpu.is_cpu_quantized() ||
        !matmul_2d_quantized_batch(cpu_context,
          layer.mlp_gate_up_cpu, post_norm, batch_size, gate_up, error_message)) {
      if (error_message.empty()) {
        error_message = "Batched CPU MLP requires packed Q4_0/Q8_0 gate/up weights.";
      }
      return false;
    }
    for (std::size_t token = 0; token < batch_size; ++token) {
      const float * gate = gate_up.data() + token * 2 * intermediate;
      cpu::silu_mul_f32(
        gate,
        gate + intermediate,
        mlp_hidden.data() + token * intermediate,
        intermediate,
        layer.mlp_down.q8_0_backend);
    }
    if (!matmul_2d_quantized_batch(cpu_context,
          layer.mlp_down, mlp_hidden, batch_size, mlp_output, error_message)) {
      return false;
    }
    cpu::add_f32(
      residual.data(), mlp_output.data(), x.data(), x.size());
    if (profiling != nullptr) {
      profiling->mlp_ms += elapsed_ms(mlp_start);
    }
  }

  if (!compute_next_logits && normalized_output == nullptr) {
    return true;
  }
  const auto logits_start = std::chrono::steady_clock::now();
  const float * last_hidden = x.data() + (batch_size - 1) * hidden;
  auto & final_hidden = scratch.final_hidden;
  final_hidden.resize(hidden);
  cpu::rms_norm_f32(last_hidden, weights.final_norm.data.data(), final_hidden.data(),
    1, hidden, dims.rms_eps, 1.0F);
  if (normalized_output != nullptr) *normalized_output = final_hidden;
  if (!compute_next_logits) return true;
  bool ok = false;
  if (greedy_sampling != nullptr && greedy_sampling->enabled &&
      greedy_sampling->token_counts != nullptr) {
    ok = greedy_q4_token(cpu_context,
      weights.embed_tokens, final_hidden, *greedy_sampling->token_counts,
      greedy_sampling->repetition_penalty, greedy_sampling->next_token,
      error_message);
    next_logits.clear();
  } else {
    ok = compute_next_logits_from_embedding(cpu_context,
      weights.embed_tokens, final_hidden, next_logits, error_message);
  }
  if (profiling != nullptr) {
    profiling->logits_ms += elapsed_ms(logits_start);
  }
  return ok;
}

} // namespace
