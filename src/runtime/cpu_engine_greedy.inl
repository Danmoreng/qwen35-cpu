// Included inside CpuEngine::Impl. Shared weight tiles, private penalties and
// stable lowest-token tie breaking. Full logits remain the sampling reference.
struct GreedyBatchJob {
  const TensorData *weight;
  const cpu::Q8_0BlockX4 *packed;
  const cpu::Q8_0BlockX1 *tail;
  Sequence *const *sequences;
  cpu::Q4_0ArgmaxResult *results;
  std::size_t count, blocks, row_tiles, workers;
};
static void greedy_batch_rows(void *opaque, std::size_t begin,
                              std::size_t end) noexcept {
  const auto &job = *static_cast<GreedyBatchJob *>(opaque);
  constexpr std::size_t tile_rows = 64;
  alignas(64) float logits[tile_rows * 16];
  for (std::size_t worker = begin; worker < end; ++worker) {
    const std::size_t first = job.row_tiles * worker / job.workers;
    const std::size_t last = job.row_tiles * (worker + 1) / job.workers;
    auto *best = job.results + worker * job.count;
    for (std::size_t tile = first; tile < last; tile += tile_rows / 8) {
      const std::size_t rows = std::min(tile_rows, (last - tile) * 8);
      const auto *matrix =
          job.weight->packed_q4_0_blocks.data() + tile * job.blocks;
      for (std::size_t r = 0; r < job.count;) {
        const std::size_t remaining = job.count - r;
        const std::size_t vectors = remaining >= 16  ? 16
                                    : remaining >= 4 ? 4
                                                     : 1;
        if (vectors == 1) {
          (job.weight->q4_dot4 ? cpu::q4_dot4_matvec
                               : cpu::q4_0_packed_matvec_prepared_q8_0)(
              matrix, job.tail + (r - job.count / 4 * 4) * job.blocks, logits,
              rows, job.blocks, job.weight->q8_0_backend);
        } else {
          (job.weight->q4_dot4 ? cpu::q4_dot4_matmul
                               : cpu::q4_0_packed_matmul_q8_0)(
              matrix, job.packed + r / 4 * job.blocks, logits, rows, vectors,
              job.blocks, tile_rows, job.weight->q8_0_backend);
        }
        for (std::size_t v = 0; v < vectors; ++v) {
          const auto &s = *job.sequences[r + v];
          for (std::size_t k = 0; k < rows; ++k) {
            float value = logits[v * tile_rows + k];
            const auto token = tile * 8 + k;
            if (s.counts[token] > 0 && s.spec.sampling.repetition_penalty > 1) {
              value = value > 0 ? value / s.spec.sampling.repetition_penalty
                                : value * s.spec.sampling.repetition_penalty;
            }
            if (value > best[r + v].value ||
                (value == best[r + v].value && token < best[r + v].index))
              best[r + v] = {value, token};
          }
        }
        r += vectors;
      }
    }
  }
}
bool greedy_batch(std::span<Sequence *> rows, std::string &error) {
  const auto &weight = model->impl_->weights.embed_tokens;
  const std::size_t width = model->impl_->dims.hidden;
  const std::size_t blocks = width / 32;
  const std::size_t count = rows.size(), packed_count = count / 4 * 4;
  auto &packed = context->packed_q8_0_batch;
  packed.resize(packed_count / 4 * blocks);
  if (packed_count) {
    if (weight.uses_q4_h128_transform) {
      if (!cpu::q4_h128_prepare_activations_4(
              batch.final_hidden.data(), packed.data(), packed_count, width,
              weight.q4_h128_sign_seed, weight.q8_0_backend,
              weight.q4_h128_signs.data())) {
        error = "Batched greedy H128 preparation failed.";
        return false;
      }
    } else
      cpu::q8_0_quantize_vectors_4(batch.final_hidden.data(), packed.data(),
                                   packed_count, blocks, weight.q8_0_backend);
  }
  auto &tail = context->prepared_q4_input;
  tail.resize((count - packed_count) * blocks);
  for (std::size_t r = packed_count; r < count; ++r)
    if (!prepare_q4_decode_activation(
            weight, batch.final_hidden.data() + r * width, width,
            tail.data() + (r - packed_count) * blocks, error))
      return false;
  const std::size_t workers = context->executor->thread_count();
  batch.greedy_results.assign(
      workers * count,
      cpu::Q4_0ArgmaxResult{-std::numeric_limits<float>::infinity(), 0});
  GreedyBatchJob job{&weight,
                     packed.data(),
                     tail.data(),
                     rows.data(),
                     batch.greedy_results.data(),
                     count,
                     blocks,
                     static_cast<std::size_t>(model->impl_->dims.vocab_size) /
                         8,
                     workers};
  if (context->executor->parallel_for_rows(workers, greedy_batch_rows, &job) !=
      cpu::CpuExecutorStatus::ok) {
    error = "Batched greedy executor failed.";
    return false;
  }
  batch.greedy_tokens.resize(count);
  for (std::size_t r = 0; r < count; ++r) {
    auto best = batch.greedy_results[r];
    for (std::size_t worker = 1; worker < workers; ++worker) {
      const auto candidate = batch.greedy_results[worker * count + r];
      if (candidate.value > best.value ||
          (candidate.value == best.value && candidate.index < best.index))
        best = candidate;
    }
    batch.greedy_tokens[r] = static_cast<int>(best.index);
  }
  return true;
}
