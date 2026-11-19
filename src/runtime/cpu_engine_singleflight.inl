// Included inside CpuEngine::Impl. The owner alone builds/publishes snapshots.
static bool same_build(const CpuRequestSpec &a, const CpuRequestSpec &b) {
  return a.register_prefix_tokens &&
         a.register_prefix_tokens == b.register_prefix_tokens &&
         (a.draft_mode == CpuDraftMode::mtp) == (b.draft_mode == CpuDraftMode::mtp) &&
         a.trust_namespace == b.trust_namespace &&
         std::equal(a.prompt_tokens.begin(),
                    a.prompt_tokens.begin() + a.register_prefix_tokens,
                    b.prompt_tokens.begin());
}
bool prefix_admissible(Sequence &s) {
  const auto count = s.spec.register_prefix_tokens;
  if (!count)
    return true;
  const bool mtp = s.spec.draft_mode == CpuDraftMode::mtp;
  for (const auto &prefix : prefixes)
    if (prefix->token_count() == count &&
        prefix->impl_->trust_namespace == s.spec.trust_namespace &&
        prefix->impl_->has_mtp() == mtp &&
        std::equal(prefix->impl_->snapshot.prefix_tokens.begin(),
                   prefix->impl_->snapshot.prefix_tokens.end(),
                   s.spec.prompt_tokens.begin())) {
      s.spec.prefix = prefix;
      return true;
    }
  for (const auto &[id, other] : requests)
    if (other.get() != &s && other->resident && !terminal(*other) &&
        other->result.committed_tokens < count && same_build(s.spec, other->spec))
      return false;
  return true;
}
bool prefix_uses_fp16() const {
  return config.shared_kv_pages || config.kv_cache_bits == 16 ||
         (!config.kv_cache_bits && cpu::q8_0_backend_uses_avx2(model->impl_->backend));
}
std::size_t prefix_size_estimate(std::size_t count, bool mtp) const {
  const auto &d = model->impl_->dims;
  const auto kv_width = static_cast<std::size_t>(d.n_kv_heads) * d.head_dim;
  std::size_t bytes = count * sizeof(std::int32_t) + d.hidden * sizeof(float);
  if (mtp)
    bytes += count * kv_width * 2 * sizeof(std::uint16_t);
  const auto page_tokens = cpu::KvPagesF16::page_tokens;
  const auto kv_tokens = config.shared_kv_pages
      ? ((count + page_tokens - 1) / page_tokens) * page_tokens : count;
  const auto kv_element_bytes = prefix_uses_fp16() ? sizeof(std::uint16_t) : sizeof(float);
  for (const auto &layer : model->impl_->weights.layers)
    if (layer.is_linear)
      bytes += (static_cast<std::size_t>(d.linear_kernel - 1) *
                    d.linear_conv_channels +
                static_cast<std::size_t>(d.linear_num_v_heads) *
                    d.linear_head_k_dim * d.linear_head_v_dim) *
               sizeof(float);
    else
      bytes += kv_tokens * kv_width * 2 * kv_element_bytes +
               (config.shared_kv_pages ? count * 2 * sizeof(const std::uint16_t *) : 0);
  return bytes;
}
std::shared_ptr<const CpuPrefix> capture_prefix(const Sequence &s, std::size_t count,
                                               std::vector<float> hidden) {
  const auto &d = model->impl_->dims;
  const auto kv_width = static_cast<std::size_t>(d.n_kv_heads) * d.head_dim;
  auto prefix = std::make_shared<CpuPrefix::Impl>();
  prefix->model = model;
  prefix->trust_namespace = s.spec.trust_namespace;
  prefix->prefill_chunk_size = config.prefill_chunk_size;
  prefix->paged_kv = config.shared_kv_pages;
  prefix->snapshot.prefix_tokens.assign(s.spec.prompt_tokens.begin(),
                                        s.spec.prompt_tokens.begin() + count);
  prefix->final_hidden = std::move(hidden);
  capture_cpu_prefix_state(s.state, count, kv_width, prefix_uses_fp16(), prefix->snapshot);
  if (s.mtp) {
    const auto values = count * kv_width;
    const auto &attention = s.mtp->attention;
    prefix->mtp_keys.assign(attention.k_cache_f16.begin(), attention.k_cache_f16.begin() + values);
    prefix->mtp_values.assign(attention.v_cache_f16.begin(), attention.v_cache_f16.begin() + values);
  }
  prefix->bytes = cpu_prefix_cache_size_bytes(prefix->snapshot) +
                  prefix->final_hidden.size() * sizeof(float) +
                  (prefix->mtp_keys.size() + prefix->mtp_values.size()) * sizeof(std::uint16_t);
  auto handle = std::shared_ptr<const CpuPrefix>(new CpuPrefix(std::move(prefix)));
  prefixes.push_back(handle);
  prefix_bytes += handle->size_bytes();
  return handle;
}
void publish_prefix(Sequence &s, std::vector<float> hidden) {
  const auto count = s.spec.register_prefix_tokens;
  std::string ignored;
  bool published = false;
  // The restored state is already private. Allow the old entry to be evicted
  // when the cache can hold either checkpoint but cannot hold both at once.
  s.spec.prefix.reset();
  if (prefix_room(prefix_size_estimate(count, bool(s.mtp)), ignored))
    try {
      s.spec.prefix = capture_prefix(s, count, std::move(hidden));
      ++stats.prefix_builds;
      s.result.prefix_written_tokens = count - s.result.cached_prefix_tokens;
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
