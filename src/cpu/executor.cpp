#include "qwen35x/cpu/executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#endif

namespace qwen35x::cpu {

namespace {

inline void cpu_relax() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
  _mm_pause();
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
  __asm__ __volatile__("pause" ::: "memory");
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

struct RowRange {
  std::size_t begin = 0;
  std::size_t end = 0;
};

[[nodiscard]] RowRange static_row_range(
  const std::size_t row_count,
  const std::size_t partition_count,
  const std::size_t partition_index) noexcept {
  const std::size_t rows_per_partition = row_count / partition_count;
  const std::size_t remainder = row_count % partition_count;
  const std::size_t extra_before = std::min(partition_index, remainder);
  const std::size_t begin = partition_index * rows_per_partition + extra_before;
  return RowRange{
    begin,
    begin + rows_per_partition + (partition_index < remainder ? 1U : 0U),
  };
}

class AtomicFlagGuard final {
public:
  explicit AtomicFlagGuard(std::atomic_flag & flag) noexcept : flag_(flag) {}
  ~AtomicFlagGuard() {
    flag_.clear(std::memory_order_release);
  }

  AtomicFlagGuard(const AtomicFlagGuard &) = delete;
  AtomicFlagGuard & operator=(const AtomicFlagGuard &) = delete;

private:
  std::atomic_flag & flag_;
};

struct Q8MatvecJob {
  const Q8_0Block * matrix = nullptr;
  const Q8_0Block * vector = nullptr;
  float * output = nullptr;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
};

struct Q8MatmulJob {
  const Q8_0Block * matrix = nullptr;
  const Q8_0Block * vectors = nullptr;
  float * output = nullptr;
  std::size_t total_row_count = 0;
  std::size_t vector_count = 0;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
  const float * vector_scales = nullptr;
  const float * matrix_scales = nullptr;
};

struct Q4MatvecJob {
  const Q4_0Block * matrix = nullptr;
  const Q8_0Block * vector = nullptr;
  float * output = nullptr;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
};

struct Q4MatmulJob {
  const Q4_0Block * matrix = nullptr;
  const Q8_0Block * vectors = nullptr;
  float * output = nullptr;
  std::size_t total_row_count = 0;
  std::size_t vector_count = 0;
  std::size_t blocks_per_row = 0;
  Q8_0Backend backend = Q8_0Backend::auto_select;
  const float * vector_scales = nullptr;
  const float * matrix_scales = nullptr;
};

void run_q8_matvec_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q8MatvecJob *>(opaque_context);
  const Q8_0Block * row_matrix = job.matrix;
  if (job.blocks_per_row != 0) {
    row_matrix += row_begin * job.blocks_per_row;
  }
  qwen35x::cpu::q8_0_matvec(
    row_matrix,
    job.vector,
    job.output + row_begin,
    row_end - row_begin,
    job.blocks_per_row,
    job.backend);
}

void run_q8_matmul_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q8MatmulJob *>(opaque_context);
  const std::size_t local_rows = row_end - row_begin;
  qwen35x::cpu::q8_0_matmul(
    job.matrix + row_begin * job.blocks_per_row,
    job.vectors,
    job.output + row_begin,
    local_rows,
    job.vector_count,
    job.blocks_per_row,
    job.total_row_count,
    job.backend,
    job.vector_scales,
    job.matrix_scales != nullptr
      ? job.matrix_scales + row_begin * job.blocks_per_row
      : nullptr);
}

void run_q4_matvec_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q4MatvecJob *>(opaque_context);
  const Q4_0Block * row_matrix = job.matrix;
  if (job.blocks_per_row != 0) {
    row_matrix += row_begin * job.blocks_per_row;
  }
  qwen35x::cpu::q4_0_matvec_q8_0(
    row_matrix, job.vector, job.output + row_begin,
    row_end - row_begin, job.blocks_per_row, job.backend);
}

void run_q4_matmul_rows(
  void * opaque_context,
  const std::size_t row_begin,
  const std::size_t row_end) noexcept {
  auto & job = *static_cast<Q4MatmulJob *>(opaque_context);
  qwen35x::cpu::q4_0_matmul_q8_0(
    job.matrix + row_begin * job.blocks_per_row,
    job.vectors,
    job.output + row_begin,
    row_end - row_begin,
    job.vector_count,
    job.blocks_per_row,
    job.total_row_count,
    job.backend,
    job.vector_scales,
    job.matrix_scales != nullptr
      ? job.matrix_scales + row_begin * job.blocks_per_row
      : nullptr);
}

} // namespace

class CpuExecutor::Impl {
  // Only the participating worker's mailbox is published. Inactive workers
  // never inspect mutable payload from a job they do not have to acknowledge.
  struct alignas(64) WorkerSlot {
    std::atomic<std::uint64_t> generation{0};
    CpuRowRangeTask task = nullptr;
    void * context = nullptr;
    std::size_t rows = 0;
    std::size_t participants = 0;
  };
public:
  Impl(std::size_t threads, std::size_t threshold, std::size_t spins) noexcept
    : thread_count_(threads), min_parallel_rows_(threshold), spin_count_(spins) {}
  ~Impl() { shutdown(); }
  Impl(const Impl &) = delete;
  Impl & operator=(const Impl &) = delete;

  bool start(std::error_code & error) noexcept {
    error.clear();
    try {
      slots_ = std::make_unique<WorkerSlot[]>(thread_count_ - 1);
      workers_.reserve(thread_count_ - 1);
      for (std::size_t index = 1; index < thread_count_; ++index)
        workers_.emplace_back([this, index] { worker_loop(index); });
      return true;
    } catch (const std::system_error & exception) { error = exception.code(); }
      catch (const std::bad_alloc &) { error = std::make_error_code(std::errc::not_enough_memory); }
      catch (...) { error = std::make_error_code(std::errc::resource_unavailable_try_again); }
    shutdown();
    return false;
  }
  void shutdown() noexcept {
    stopping_.store(true, std::memory_order_release);
    for (std::size_t index = 0; index < workers_.size(); ++index) {
      slots_[index].generation.fetch_add(1, std::memory_order_release);
      slots_[index].generation.notify_one();
    }
    for (auto & worker : workers_) if (worker.joinable()) worker.join();
    workers_.clear();
  }
  std::size_t thread_count() const noexcept { return thread_count_; }
  std::size_t worker_thread_count() const noexcept { return thread_count_ - 1; }
  std::size_t min_parallel_rows() const noexcept { return min_parallel_rows_; }

  CpuExecutorStatus parallel_for_rows(std::size_t rows, CpuRowRangeTask task, void * context,
                                      CpuExecutorTiming * timing = nullptr) noexcept {
    return timing ? parallel_for_rows_impl<true>(rows, task, context, timing)
                  : parallel_for_rows_impl<false>(rows, task, context, nullptr);
  }
  template<bool Profile>
  CpuExecutorStatus parallel_for_rows_impl(std::size_t rows, CpuRowRangeTask task, void * context,
                                         CpuExecutorTiming * timing) noexcept {
    using Clock = std::chrono::steady_clock;
    Clock::time_point started{}, dispatched{}, worked{};
    if constexpr(Profile) { *timing = {}; started = Clock::now(); }
    const auto finish = [&]() {
      if constexpr(Profile) {
        timing->dispatch_ms = std::chrono::duration<double, std::milli>(dispatched-started).count();
        timing->caller_ms = std::chrono::duration<double, std::milli>(worked-dispatched).count();
        timing->wait_ms = std::chrono::duration<double, std::milli>(Clock::now()-worked).count();
      }
      return CpuExecutorStatus::ok;
    };
    if (rows == 0) return CpuExecutorStatus::ok;
    if (task == nullptr) return CpuExecutorStatus::invalid_argument;
    if (job_in_progress_.test_and_set(std::memory_order_acquire)) return CpuExecutorStatus::busy;
    const AtomicFlagGuard guard(job_in_progress_);
    const std::size_t participants = std::min(rows, thread_count_);
    if constexpr(Profile) timing->participants = participants;
    if (participants == 1 || rows < min_parallel_rows_) {
      if constexpr(Profile) { timing->participants = 1; dispatched = Clock::now(); }
      task(context, 0, rows);
      if constexpr(Profile) worked = Clock::now();
      return finish();
    }
    const std::size_t workers = participants - 1;
    completed_workers_.store(0, std::memory_order_relaxed);
    // The preceding completion acquire guarantees the previous payload is no
    // longer being read. Each mailbox release publishes this job's payload.
    for (std::size_t index = 0; index < workers; ++index) {
      auto & slot = slots_[index];
      slot.task = task; slot.context = context; slot.rows = rows; slot.participants = participants;
      slot.generation.fetch_add(1, std::memory_order_release);
      slot.generation.notify_one();
    }
    const auto range = static_row_range(rows, participants, 0);
    if constexpr(Profile) dispatched = Clock::now();
    task(context, range.begin, range.end);
    if constexpr(Profile) worked = Clock::now();
    for (std::size_t spin = 0; spin < spin_count_; ++spin) {
      if (completed_workers_.load(std::memory_order_acquire) == workers) return finish();
      cpu_relax();
    }
    auto completed = completed_workers_.load(std::memory_order_acquire);
    while (completed != workers) {
      completed_workers_.wait(completed, std::memory_order_acquire);
      completed = completed_workers_.load(std::memory_order_acquire);
    }
    return finish();
  }
private:
  void worker_loop(std::size_t index) noexcept {
    auto & slot = slots_[index-1];
    std::uint64_t observed = 0;
    while (true) {
      auto generation = observed;
      for (std::size_t spin = 0; spin < spin_count_; ++spin) {
        if (stopping_.load(std::memory_order_acquire)) return;
        generation = slot.generation.load(std::memory_order_acquire);
        if (generation != observed) break;
        cpu_relax();
      }
      generation = slot.generation.load(std::memory_order_acquire);
      while (generation == observed) {
        slot.generation.wait(observed, std::memory_order_acquire);
        generation = slot.generation.load(std::memory_order_acquire);
      }
      if (stopping_.load(std::memory_order_acquire)) return;
      observed = generation;
      const auto task = slot.task;
      void * const context = slot.context;
      const auto participants = slot.participants;
      const auto range = static_row_range(slot.rows, participants, index);
      task(context, range.begin, range.end);
      if (completed_workers_.fetch_add(1, std::memory_order_release) + 1 == participants - 1)
        completed_workers_.notify_one();
    }
  }
  const std::size_t thread_count_, min_parallel_rows_, spin_count_;
  std::unique_ptr<WorkerSlot[]> slots_;
  std::vector<std::thread> workers_;
  std::atomic<bool> stopping_{false};
  alignas(64) std::atomic<std::size_t> completed_workers_{0};
  std::atomic_flag job_in_progress_ = ATOMIC_FLAG_INIT;
};

const char * cpu_executor_status_name(const CpuExecutorStatus status) noexcept {
  switch (status) {
    case CpuExecutorStatus::ok:
      return "ok";
    case CpuExecutorStatus::invalid_argument:
      return "invalid_argument";
    case CpuExecutorStatus::busy:
      return "busy";
  }
  return "unknown";
}

std::unique_ptr<CpuExecutor> CpuExecutor::create(
  CpuExecutorConfig config,
  std::error_code & error_code) noexcept {
  error_code.clear();
  if (config.thread_count == 0) {
    config.thread_count = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  }

  try {
    auto impl = std::make_unique<Impl>(config.thread_count, config.min_parallel_rows, config.spin_count);
    if (!impl->start(error_code)) {
      return nullptr;
    }
    return std::unique_ptr<CpuExecutor>(new CpuExecutor(std::move(impl)));
  } catch (const std::bad_alloc &) {
    error_code = std::make_error_code(std::errc::not_enough_memory);
  } catch (...) {
    error_code = std::make_error_code(std::errc::resource_unavailable_try_again);
  }
  return nullptr;
}

CpuExecutor::CpuExecutor(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

CpuExecutor::~CpuExecutor() = default;

std::size_t CpuExecutor::thread_count() const noexcept {
  return impl_->thread_count();
}

std::size_t CpuExecutor::worker_thread_count() const noexcept {
  return impl_->worker_thread_count();
}

std::size_t CpuExecutor::min_parallel_rows() const noexcept {
  return impl_->min_parallel_rows();
}

CpuExecutorStatus CpuExecutor::parallel_for_rows(
  const std::size_t row_count,
  const CpuRowRangeTask task,
  void * context,
  CpuExecutorTiming * timing) noexcept {
  return impl_->parallel_for_rows(row_count, task, context, timing);
}

CpuExecutorStatus CpuExecutor::q8_0_matvec(
  const Q8_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
  if (row_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512 &&
       backend != Q8_0Backend::avx512_vnni) ||
      output == nullptr ||
      (blocks_per_row != 0 && (matrix == nullptr || vector == nullptr)) ||
      (blocks_per_row != 0 &&
       row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row)) {
    return CpuExecutorStatus::invalid_argument;
  }

  Q8MatvecJob job{
    matrix,
    vector,
    output,
    blocks_per_row,
    q8_0_resolve_backend(backend),
  };
  return parallel_for_rows(row_count, run_q8_matvec_rows, &job);
}

CpuExecutorStatus CpuExecutor::q8_0_matmul(
  const Q8_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  if (row_count == 0 || vector_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512 &&
       backend != Q8_0Backend::avx512_vnni) ||
      matrix == nullptr || vectors == nullptr || output == nullptr ||
      blocks_per_row == 0 ||
      row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / row_count) {
    return CpuExecutorStatus::invalid_argument;
  }
  Q8MatmulJob job{
    matrix,
    vectors,
    output,
    row_count,
    vector_count,
    blocks_per_row,
    q8_0_resolve_backend(backend),
    vector_scales,
    matrix_scales,
  };
  return parallel_for_rows(row_count, run_q8_matmul_rows, &job);
}

CpuExecutorStatus CpuExecutor::q4_0_matvec_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vector,
  float * output,
  const std::size_t row_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend) noexcept {
  if (row_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512 &&
       backend != Q8_0Backend::avx512_vnni) ||
      output == nullptr ||
      (blocks_per_row != 0 && (matrix == nullptr || vector == nullptr)) ||
      (blocks_per_row != 0 &&
       row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row)) {
    return CpuExecutorStatus::invalid_argument;
  }
  Q4MatvecJob job{
    matrix, vector, output, blocks_per_row, q8_0_resolve_backend(backend),
  };
  return parallel_for_rows(row_count, run_q4_matvec_rows, &job);
}

CpuExecutorStatus CpuExecutor::q4_0_matmul_q8_0(
  const Q4_0Block * matrix,
  const Q8_0Block * vectors,
  float * output,
  const std::size_t row_count,
  const std::size_t vector_count,
  const std::size_t blocks_per_row,
  const Q8_0Backend backend,
  const float * vector_scales,
  const float * matrix_scales) noexcept {
  if (row_count == 0 || vector_count == 0) {
    return CpuExecutorStatus::ok;
  }
  if ((backend != Q8_0Backend::auto_select &&
       backend != Q8_0Backend::scalar &&
       backend != Q8_0Backend::avx2 &&
       backend != Q8_0Backend::avx_vnni &&
       backend != Q8_0Backend::avx512 &&
       backend != Q8_0Backend::avx512_vnni) ||
      matrix == nullptr || vectors == nullptr || output == nullptr ||
      blocks_per_row == 0 ||
      row_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / blocks_per_row ||
      vector_count > std::numeric_limits<std::size_t>::max() / row_count) {
    return CpuExecutorStatus::invalid_argument;
  }
  Q4MatmulJob job{
    matrix, vectors, output, row_count, vector_count, blocks_per_row,
    q8_0_resolve_backend(backend), vector_scales, matrix_scales,
  };
  return parallel_for_rows(row_count, run_q4_matmul_rows, &job);
}

} // namespace qwen35x::cpu
