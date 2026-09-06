// Included inside CpuEngine::Impl. Workers own row scratch and run the
// stateful kernel directly, without recursively dispatching the shared pool.
struct StateRowScratch {
  CpuDecodeWorkspace::Linear linear;
  CpuDecodeWorkspace::Full full;
  std::vector<float> output;
  std::string error;
  bool ok = false;
};
std::vector<StateRowScratch> state_rows;
struct StateBatchJob {
  Impl *engine;
  const LayerWeights *layer;
  Sequence *const *sequences;
  std::size_t linear_index, full_index, projected_width, core_width;
};
static void state_batch_rows(void *opaque, std::size_t begin,
                             std::size_t end) noexcept {
  const auto &job = *static_cast<StateBatchJob *>(opaque);
  auto &engine = *job.engine;
  const auto &d = engine.model->impl_->dims;
  for (std::size_t r = begin; r < end; ++r) {
    auto &scratch = engine.state_rows[r];
    auto &s = *job.sequences[r];
    const std::span<const float> projected(engine.batch.projected.data() +
                                               r * job.projected_width,
                                           job.projected_width);
    try {
      if (job.layer->is_linear)
        scratch.ok = run_linear_attention_step(
            nullptr, *job.layer, d, s.state.linear_states[job.linear_index],
            engine.batch.normed, scratch.output, scratch.error,
            projected, true, &scratch.linear);
      else {
        const auto position = s.result.committed_tokens;
        const auto offset = position * (d.rope_dim / 2);
        scratch.ok = run_full_attention_step(
            nullptr, *job.layer, d, s.state.full_states[job.full_index],
            engine.batch.normed, static_cast<int>(position),
            s.state.rope_cosine.data() + offset,
            s.state.rope_sine.data() + offset, scratch.output, scratch.error, projected, true, &scratch.full);
      }
      if (scratch.ok)
        std::copy(scratch.output.begin(), scratch.output.end(),
                  engine.batch.core.begin() + r * job.core_width);
    } catch (...) {
      // No exception may cross the executor's noexcept callback. The owner
      // marks the entire partially executed batch failed before returning.
      scratch.ok = false;
    }
  }
}
bool state_batch(const LayerWeights &layer, std::span<Sequence *> rows,
                 std::size_t linear_index, std::size_t full_index,
                 std::size_t projected_width, std::size_t core_width,
                 std::string &error) {
  state_rows.resize(rows.size());
  for (auto &scratch : state_rows) {
    scratch.ok = false;
    scratch.error.clear();
  }
  StateBatchJob job{this,       &layer,          rows.data(), linear_index,
                    full_index, projected_width, core_width};
  if (context->executor->parallel_for_rows(
          rows.size(), state_batch_rows, &job) != cpu::CpuExecutorStatus::ok) {
    error = "CPU state batch executor failed.";
    return false;
  }
  for (const auto &scratch : state_rows)
    if (!scratch.ok) {
      error =
          scratch.error.empty() ? "CPU state worker failed." : scratch.error;
      return false;
    }
  ++stats.state_batches;
  return true;
}
