// Included inside CpuEngine::Impl. The owner alone builds/publishes snapshots.
static bool same_build(const CpuRequestSpec &a, const CpuRequestSpec &b) {
  return a.register_prefix_tokens &&
         a.register_prefix_tokens == b.register_prefix_tokens &&
         a.trust_namespace == b.trust_namespace &&
         std::equal(a.prompt_tokens.begin(),
                    a.prompt_tokens.begin() + a.register_prefix_tokens,
                    b.prompt_tokens.begin());
}
bool prefix_admissible(Sequence &s) {
  const auto count = s.spec.register_prefix_tokens;
  if (!count || s.spec.prefix)
    return true;
  for (const auto &prefix : prefixes)
    if (prefix->token_count() == count &&
        prefix->impl_->trust_namespace == s.spec.trust_namespace &&
        std::equal(prefix->impl_->snapshot.prefix_tokens.begin(),
                   prefix->impl_->snapshot.prefix_tokens.end(),
                   s.spec.prompt_tokens.begin())) {
      s.spec.prefix = prefix;
      return true;
    }
  for (const auto &[id, other] : requests)
    if (other.get() != &s && other->resident && !terminal(*other) &&
        !other->spec.prefix && same_build(s.spec, other->spec))
      return false;
  return true;
}
void publish_prefix(Sequence &s, std::vector<float> hidden) {
  const auto count = s.spec.register_prefix_tokens;
  const auto &d = model->impl_->dims;
  const bool fp16 = cpu::q8_0_backend_uses_avx2(model->impl_->backend);
  std::size_t bytes =
      count * sizeof(std::int32_t) + hidden.size() * sizeof(float);
  const auto kv_tokens=config.shared_kv_pages?
    ((count+cpu::KvPagesF16::page_tokens-1)/cpu::KvPagesF16::page_tokens)*cpu::KvPagesF16::page_tokens:count;
  for (const auto &layer : model->impl_->weights.layers)
    if (layer.is_linear)
      bytes += (static_cast<std::size_t>(d.linear_kernel - 1) *
                    d.linear_conv_channels +
                static_cast<std::size_t>(d.linear_num_v_heads) *
                    d.linear_head_k_dim * d.linear_head_v_dim) *
               sizeof(float);
    else
      bytes += kv_tokens * d.n_kv_heads * d.head_dim * 2 *
               (fp16 ? sizeof(std::uint16_t) : sizeof(float))+
               (config.shared_kv_pages?count*2*sizeof(const std::uint16_t*):0);
  std::string ignored;
  bool published = false;
  if (prefix_room(bytes, ignored))
    try {
      auto snapshot = std::make_shared<CpuPrefix::Impl>();
      snapshot->model = model;
      snapshot->trust_namespace = s.spec.trust_namespace;
      snapshot->prefill_chunk_size = config.prefill_chunk_size;
      snapshot->paged_kv=config.shared_kv_pages;
      snapshot->snapshot.prefix_tokens.assign(
          s.spec.prompt_tokens.begin(), s.spec.prompt_tokens.begin() + count);
      snapshot->final_hidden = std::move(hidden);
      capture_cpu_prefix_state(s.state, count, d.n_kv_heads * d.head_dim, fp16,
                               snapshot->snapshot);
      snapshot->bytes = cpu_prefix_cache_size_bytes(snapshot->snapshot) +
                        snapshot->final_hidden.size() * sizeof(float);
      auto handle =
          std::shared_ptr<const CpuPrefix>(new CpuPrefix(std::move(snapshot)));
      prefixes.push_back(handle);
      prefix_bytes += handle->size_bytes();
      s.spec.prefix = std::move(handle);
      ++stats.prefix_builds;
      published = true;
    } catch (const std::bad_alloc &) {
      // Caching is optional; already computed private state remains valid.
    }
  if (!published) {
    ++stats.prefix_build_fallbacks;
    // Resolve followers as ordinary cold requests, rather than rebuilding a
    // snapshot that cannot fit the budget on every admission attempt.
    for (auto &[id, other] : requests)
      if (other.get() != &s && same_build(s.spec, other->spec))
        other->spec.register_prefix_tokens = 0;
    s.spec.register_prefix_tokens = 0;
  }
}
