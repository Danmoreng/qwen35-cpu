bool gguf_shape_matches(
  const std::vector<std::uint64_t> & actual,
  const std::initializer_list<std::int64_t> expected) {
  if (actual.size() != expected.size()) {
    return false;
  }
  std::size_t index = 0;
  for (const std::int64_t dimension : expected) {
    if (dimension < 0 || actual[index] != static_cast<std::uint64_t>(dimension)) {
      return false;
    }
    ++index;
  }
  return true;
}

bool load_gguf_quantized_checked(const GgufReader& reader,const std::string& name,
    std::int64_t rows,std::int64_t cols,cpu::Q8_0Backend backend,
    TensorData& out,std::string& error,bool =true) {
  const auto* info=reader.find_tensor(name);
  if(info && info->is_q4_0()) {
    if(!gguf_shape_matches(info->shape,{rows,cols}) || rows%8 || cols%32) {
      error="Unsupported Q4_0 matrix shape: "+name;return false;
    }
    GgufTensorQ4_0 tensor;
    if(!reader.read_q4_0_tensor(name,tensor,error))return false;
    out=TensorData{};out.shape={rows,cols};out.q8_0_backend=backend;out.q4_dot4=true;
    out.q4_0_blocks=std::move(tensor.blocks);
    out.q4_0_scales.resize(out.q4_0_blocks.size());
    cpu::q4_0_scales_to_f32(out.q4_0_blocks.data(),out.q4_0_scales.data(),out.q4_0_blocks.size());
    out.packed_q4_0_blocks.resize(static_cast<std::size_t>(rows/8)*(cols/32));
    cpu::q4_dot4_pack_rows_8(out.q4_0_blocks.data(),out.packed_q4_0_blocks.data(),rows,cols/32);
    return true;
  }
  if(!info || !gguf_shape_matches(info->shape,{rows,cols}) || cols%256 ||
     (info->ggml_type!=8 && (info->ggml_type<12 || info->ggml_type>14))) {
    error="Expected Q4_0/Q4_K/Q5_K/Q6_K/Q8_0 matrix with matching dimensions: "+name;return false;
  }
  auto bytes=std::make_shared<std::vector<std::uint8_t>>();
  if(!reader.read_tensor_bytes(name,*bytes,error))return false;
  out=TensorData{};out.shape={rows,cols};out.q8_0_backend=backend;
  out.gguf_parts.push_back({static_cast<cpu::KQuantType>(info->ggml_type),
                           static_cast<std::size_t>(rows),std::move(bytes)});
  return true;
}

bool load_gguf_f32_checked(
  const GgufReader & reader,
  const std::string & tensor_name,
  const std::initializer_list<std::int64_t> expected_shape,
  const bool subtract_qwen_norm_offset,
  TensorData & out,
  std::string & error_message) {
  const GgufTensorInfo * info = reader.find_tensor(tensor_name);
  if (info == nullptr) {
    error_message = "Required GGUF tensor '" + tensor_name + "' is missing.";
    return false;
  }
  if (!info->is_f32()) {
    error_message = "GGUF tensor '" + tensor_name + "' is not F32.";
    return false;
  }
  if (!gguf_shape_matches(info->shape, expected_shape)) {
    error_message = "GGUF tensor '" + tensor_name + "' shape mismatch.";
    return false;
  }

  GgufTensorF32 tensor;
  if (!reader.read_f32_tensor(tensor_name, tensor, error_message)) {
    return false;
  }
  out.shape.clear();
  out.shape.reserve(tensor.shape.size());
  for (const std::uint64_t dimension : tensor.shape) {
    out.shape.push_back(static_cast<std::int64_t>(dimension));
  }
  out.data = std::move(tensor.data);
  out.q4_0_blocks.clear();
  out.packed_q4_0_blocks.clear();
  out.q4_0_scales.clear();
  out.q8_0_blocks.clear();
  out.q8_0_scales.clear();
  if (subtract_qwen_norm_offset) {
    for (float & value : out.data) {
      value -= 1.0f;
    }
  }
  return true;
}

bool load_model_weights_from_quantized_gguf(
  const std::string & gguf_path,
  const RuntimeDims & dims,
  const ModelProfile & profile,
  const cpu::Q8_0Backend backend,
  const int cpu_threads,
  ModelWeights & weights,
  std::string & error_message) {
  if (dims.linear_num_k_heads != dims.linear_num_v_heads) {
    error_message =
      "The direct Q8_0 GGUF path currently requires equal DeltaNet key/value head counts; "
      "inverse GGML V-head permutation is not implemented.";
    return false;
  }

  GgufReader reader;
  if (!reader.open(gguf_path, error_message)) {
    return false;
  }

  if (!load_gguf_quantized_checked(
        reader,
        "token_embd.weight",
        dims.vocab_size,
        dims.hidden,
        backend,
        weights.embed_tokens,
        error_message,
        false) ||
      !load_gguf_f32_checked(
        reader,
        "output_norm.weight",
        {dims.hidden},
        true,
        weights.final_norm,
        error_message)) {
    return false;
  }

  if (reader.find_tensor("output.weight")) {
    error_message="This engine requires tied input/output embeddings; output.weight is not supported.";
    return false;
  }
  weights.layers.resize(static_cast<std::size_t>(dims.n_layers));
  const int full_q_out = dims.n_heads * dims.head_dim * 2;
  const int full_kv_out = dims.n_kv_heads * dims.head_dim;
  const int full_o_in = dims.n_heads * dims.head_dim;

  for (int il = 0; il < dims.n_layers; ++il) {
    LayerWeights & layer = weights.layers[static_cast<std::size_t>(il)];
    const std::string base = "blk." + std::to_string(il) + ".";
    layer.is_linear = profile.fingerprint.attention_schedule[static_cast<std::size_t>(il)] == AttentionBlock::linear;

    if (!load_gguf_f32_checked(
          reader, base + "attn_norm.weight", {dims.hidden}, true, layer.input_layernorm, error_message) ||
        !load_gguf_f32_checked(
          reader,
          base + "post_attention_norm.weight",
          {dims.hidden},
          true,
          layer.post_attention_layernorm,
          error_message) ||
        !load_gguf_quantized_checked(
          reader, base + "ffn_gate.weight", dims.intermediate, dims.hidden, backend, layer.mlp_gate, error_message) ||
        !load_gguf_quantized_checked(
          reader, base + "ffn_up.weight", dims.intermediate, dims.hidden, backend, layer.mlp_up, error_message) ||
        !load_gguf_quantized_checked(
          reader, base + "ffn_down.weight", dims.hidden, dims.intermediate, backend, layer.mlp_down, error_message)) {
      return false;
    }

    if (layer.is_linear) {
      if (!load_gguf_quantized_checked(
            reader,
            base + "attn_qkv.weight",
            dims.linear_conv_channels,
            dims.hidden,
            backend,
                layer.linear.in_proj_qkv,
            error_message) ||
          !load_gguf_quantized_checked(
            reader,
            base + "attn_gate.weight",
            dims.linear_v_dim,
            dims.hidden,
            backend,
                layer.linear.in_proj_z,
            error_message) ||
          !load_gguf_quantized_checked(
            reader,
            base + "ssm_beta.weight",
            dims.linear_num_v_heads,
            dims.hidden,
            backend,
                layer.linear.in_proj_b,
            error_message) ||
          !load_gguf_quantized_checked(
            reader,
            base + "ssm_alpha.weight",
            dims.linear_num_v_heads,
            dims.hidden,
            backend,
                layer.linear.in_proj_a,
            error_message) ||
          !load_gguf_f32_checked(
            reader,
            base + "ssm_conv1d.weight",
            {dims.linear_conv_channels, dims.linear_kernel},
            false,
            layer.linear.conv1d,
            error_message) ||
          !load_gguf_quantized_checked(
            reader,
            base + "ssm_out.weight",
            dims.hidden,
            dims.linear_v_dim,
            backend,
                layer.linear.out_proj,
            error_message) ||
          !load_gguf_f32_checked(
            reader,
            base + "ssm_norm.weight",
            {dims.linear_head_v_dim},
            false,
            layer.linear.norm,
            error_message) ||
          !load_gguf_f32_checked(
            reader,
            base + "ssm_a",
            {dims.linear_num_v_heads},
            false,
            layer.linear.a_log,
            error_message) ||
          !load_gguf_f32_checked(
            reader,
            base + "ssm_dt.bias",
            {dims.linear_num_v_heads},
            false,
            layer.linear.dt_bias,
            error_message)) {
        return false;
      }
      // llama.cpp's converter stores -exp(A_log) directly as ssm_a.
      layer.linear.ssm_a = layer.linear.a_log.data;
      pack_conv1d_kernel_major(layer.linear, dims);
    } else {
      if (!load_gguf_quantized_checked(
            reader, base + "attn_q.weight", full_q_out, dims.hidden, backend, layer.full.q_proj, error_message) ||
          !load_gguf_quantized_checked(
            reader, base + "attn_k.weight", full_kv_out, dims.hidden, backend, layer.full.k_proj, error_message) ||
          !load_gguf_quantized_checked(
            reader, base + "attn_v.weight", full_kv_out, dims.hidden, backend, layer.full.v_proj, error_message) ||
          !load_gguf_quantized_checked(
            reader, base + "attn_output.weight", dims.hidden, full_o_in, backend, layer.full.o_proj, error_message) ||
          !load_gguf_f32_checked(
            reader, base + "attn_q_norm.weight", {dims.head_dim}, true, layer.full.q_norm, error_message) ||
          !load_gguf_f32_checked(
            reader, base + "attn_k_norm.weight", {dims.head_dim}, true, layer.full.k_norm, error_message)) {
        return false;
      }
    }

    if (!pack_quantized_row_concat(
          {&layer.mlp_gate, &layer.mlp_up}, layer.mlp_gate_up_cpu, error_message)) {
      return false;
    }
    if (layer.is_linear) {
      if (!pack_quantized_row_concat(
            {
              &layer.linear.in_proj_qkv,
              &layer.linear.in_proj_z,
              &layer.linear.in_proj_b,
              &layer.linear.in_proj_a,
            },
            layer.linear.in_proj_all_cpu,
            error_message)) {
        return false;
      }
    } else if (!pack_quantized_row_concat(
                 {&layer.full.q_proj, &layer.full.k_proj, &layer.full.v_proj},
                 layer.full.qkv_proj_cpu,
                 error_message)) {
      return false;
    }
  }

  const auto release=[](TensorData& tensor) {
    if(!tensor.packed_q4_0_blocks.empty()) {
      std::vector<cpu::Q4_0Block>().swap(tensor.q4_0_blocks);
      std::vector<float>().swap(tensor.q4_0_scales);
    }
  };
  release(weights.embed_tokens);
  for(auto& layer:weights.layers) {
    release(layer.mlp_gate_up_cpu);release(layer.mlp_down);
    if(layer.is_linear) {release(layer.linear.in_proj_all_cpu);release(layer.linear.out_proj);}
    else {release(layer.full.qkv_proj_cpu);release(layer.full.o_proj);}
  }
  return true;
}

