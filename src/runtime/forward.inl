bool run_forward_single_token(CpuExecutionContext *context,
    const ModelWeights &weights,const RuntimeDims &d,ModelState &state,
    int token,int position,std::vector<float> &logits,
    CpuGreedySamplingState *sampling,std::string &error) {
  if(token<0 || token>=d.vocab_size) {error="Token outside vocabulary.";return false;}
  auto &w=context->decode.forward;
  w.x.resize(d.hidden);
  cpu::q4_dot4_dequantize_row(weights.embed_tokens.packed_q4_0_blocks.data(),
      token,w.x.data(),static_cast<std::size_t>(d.hidden)/32);
  std::size_t linear=0,full=0;
  for(const auto &layer:weights.layers) {
    rms_norm_qwen3next(w.x,layer.input_layernorm,d.rms_eps,w.normed);
    if(layer.is_linear) {
      if(!run_linear_attention_step(context,layer,d,state.linear_states[linear++],
          w.normed,w.attn_out,error))return false;
    } else {
      const auto offset=static_cast<std::size_t>(position)*(d.rope_dim/2);
      if(!run_full_attention_step(context,layer,d,state.full_states[full++],
          w.normed,position,state.rope_cosine.data()+offset,
          state.rope_sine.data()+offset,w.attn_out,error))return false;
    }
    w.residual.resize(w.x.size());
    for(std::size_t i=0;i<w.x.size();++i)w.residual[i]=w.x[i]+w.attn_out[i];
    rms_norm_qwen3next(w.residual,layer.post_attention_layernorm,d.rms_eps,w.post_norm);
    if(!matvec_2d(context,layer.mlp_gate_up_cpu,w.post_norm,w.mlp_packed,error))return false;
    w.mlp_hidden.resize(d.intermediate);
    cpu::silu_mul_f32(w.mlp_packed.data(),w.mlp_packed.data()+d.intermediate,
        w.mlp_hidden.data(),d.intermediate,layer.mlp_down.q8_0_backend);
    if(!matvec_2d(context,layer.mlp_down,w.mlp_hidden,w.mlp_out,error))return false;
    for(std::size_t i=0;i<w.x.size();++i)w.x[i]=w.residual[i]+w.mlp_out[i];
  }
  rms_norm_qwen3next(w.x,weights.final_norm,d.rms_eps,w.final_hidden);
  if(sampling && sampling->enabled) {
    logits.clear();
    return greedy_q4_token(context,weights.embed_tokens,w.final_hidden,*sampling->token_counts,
        sampling->repetition_penalty,sampling->next_token,error);
  }
  return compute_next_logits_from_embedding(context,weights.embed_tokens,w.final_hidden,logits,error);
}
} // namespace
