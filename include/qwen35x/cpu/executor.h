#pragma once

#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q8_0.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <system_error>

namespace qwen35x::cpu {

struct CpuExecutorConfig {
  // Zero selects std::thread::hardware_concurrency(), falling back to one.
  // The resolved count includes the calling/main thread.
  std::size_t thread_count = 0;
  std::size_t min_parallel_rows = 256;
  // Workers poll briefly between adjacent inference kernels before sleeping.
  // Zero disables polling. The default mirrors llama.cpp's 50% poll level.
  std::size_t spin_count = 1024U * 128U * 50U;
};

// Diagnostic caller-side intervals. wait_ms is elapsed time after the caller's
// partition completes, not worker CPU time or necessarily wasted CPU work.
struct CpuExecutorTiming {
  std::size_t participants = 0;
  double dispatch_ms = 0, caller_ms = 0, wait_ms = 0;
};

enum class CpuExecutorStatus : std::uint8_t {
  ok = 0,
  invalid_argument = 1,
  busy = 2,
};

[[nodiscard]] const char * cpu_executor_status_name(CpuExecutorStatus status) noexcept;

// The task is called once per non-empty, contiguous static row partition.
// noexcept prevents an exception from escaping a persistent worker thread.
using CpuRowRangeTask = void (*)(
  void * context,
  std::size_t row_begin,
  std::size_t row_end) noexcept;

class CpuExecutor final {
public:
  // Thread and allocation failures are reported through error_code. No worker
  // is leaked when only part of the pool could be created.
  [[nodiscard]] static std::unique_ptr<CpuExecutor> create(
    CpuExecutorConfig config,
    std::error_code & error_code) noexcept;

  // As with any C++ object, destruction requires that no member call is still
  // in flight. In particular, do not destroy the executor from one of its own
  // row callbacks. Under that lifetime contract all persistent workers are
  // stopped and joined before destruction returns.
  ~CpuExecutor();

  CpuExecutor(const CpuExecutor &) = delete;
  CpuExecutor & operator=(const CpuExecutor &) = delete;
  CpuExecutor(CpuExecutor &&) = delete;
  CpuExecutor & operator=(CpuExecutor &&) = delete;

  [[nodiscard]] std::size_t thread_count() const noexcept;
  [[nodiscard]] std::size_t worker_thread_count() const noexcept;
  [[nodiscard]] std::size_t min_parallel_rows() const noexcept;

  // Jobs are synchronous. Concurrent or recursively submitted jobs return
  // busy instead of deadlocking. Rows below min_parallel_rows run entirely on
  // the calling thread.
  [[nodiscard]] CpuExecutorStatus parallel_for_rows(
    std::size_t row_count,
    CpuRowRangeTask task,
    void * context,
    CpuExecutorTiming * timing = nullptr) noexcept;

  // Parallel row wrapper around the runtime-dispatched Q8_0 matvec primitive.
  [[nodiscard]] CpuExecutorStatus q8_0_matvec(
    const Q8_0Block * matrix,
    const Q8_0Block * vector,
    float * output,
    std::size_t row_count,
    std::size_t blocks_per_row,
    Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

  // Parallel row wrapper around q8_0_matmul(). Output remains token-major.
  [[nodiscard]] CpuExecutorStatus q8_0_matmul(
    const Q8_0Block * matrix,
    const Q8_0Block * vectors,
    float * output,
    std::size_t row_count,
    std::size_t vector_count,
    std::size_t blocks_per_row,
    Q8_0Backend backend = Q8_0Backend::auto_select,
    const float * vector_scales = nullptr,
    const float * matrix_scales = nullptr) noexcept;

  [[nodiscard]] CpuExecutorStatus q4_0_matvec_q8_0(
    const Q4_0Block * matrix,
    const Q8_0Block * vector,
    float * output,
    std::size_t row_count,
    std::size_t blocks_per_row,
    Q8_0Backend backend = Q8_0Backend::auto_select) noexcept;

  [[nodiscard]] CpuExecutorStatus q4_0_matmul_q8_0(
    const Q4_0Block * matrix,
    const Q8_0Block * vectors,
    float * output,
    std::size_t row_count,
    std::size_t vector_count,
    std::size_t blocks_per_row,
    Q8_0Backend backend = Q8_0Backend::auto_select,
    const float * vector_scales = nullptr,
    const float * matrix_scales = nullptr) noexcept;

private:
  class Impl;

  explicit CpuExecutor(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

} // namespace qwen35x::cpu
