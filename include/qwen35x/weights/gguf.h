#pragma once

#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q8_0.h"
#include "qwen35x/common/model_profile.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen35x {

enum class GgmlTensorType : std::uint32_t {
  f32 = 0,
  q4_0 = 2,
  q8_0 = 8,
  q4_k = 12, q5_k = 13, q6_k = 14,
};

struct GgufTensorInfo {
  std::string name;

  // GGUF stores dimensions in GGML order (the contiguous dimension first).
  // shape is reversed into conventional logical/row-major order, while
  // ggml_shape preserves the dimensions exactly as serialized.
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> ggml_shape;

  std::uint32_t ggml_type = 0;
  std::uint64_t element_count = 0;
  std::uint64_t relative_offset = 0;
  std::uint64_t data_offset = 0;
  std::uint64_t data_size = 0;

  bool is_f32() const noexcept {
    return ggml_type == static_cast<std::uint32_t>(GgmlTensorType::f32);
  }

  bool is_q8_0() const noexcept {
    return ggml_type == static_cast<std::uint32_t>(GgmlTensorType::q8_0);
  }

  bool is_q4_0() const noexcept {
    return ggml_type == static_cast<std::uint32_t>(GgmlTensorType::q4_0);
  }
};

struct GgufTensorF32 {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<float> data;
};

struct GgufTensorQ8_0 {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<cpu::Q8_0Block> blocks;
};

struct GgufTensorQ4_0 {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<cpu::Q4_0Block> blocks;
};

// Validated GGUF-v3 reader for the Qwen3.5 CPU path. It indexes the file
// without retaining tensor payloads in memory. Payload reads are owned and
// independent, so callers can choose which tensors to keep resident.
//
// Tensor payload types other than F32, Q4_0, Q8_0, Q4_K, Q5_K and Q6_K are rejected at
// open time. Every GGUF metadata scalar/array/string type defined by v3 is
// is validated. Required Qwen3.5 scalars, rope sections and tokenizer count
// are retained for validate_profile; unrelated values are safely skipped.
class GgufReader {
public:
  bool open(const std::string & gguf_file, std::string & error_message);
  // Semantic checks for the supported profile, separate from structural parsing.
  // Token count/type is checked; trusted tokenizer file hashes remain necessary
  // to establish full external tokenizer identity.
  bool validate_profile(const ModelProfile &, std::string & error_message) const;
  void close() noexcept;

  bool is_open() const noexcept {
    return is_open_;
  }

  const std::string & path() const noexcept {
    return path_;
  }

  std::uint32_t version() const noexcept {
    return version_;
  }

  std::uint64_t file_size() const noexcept {
    return file_size_;
  }

  std::uint64_t metadata_count() const noexcept {
    return metadata_count_;
  }

  std::uint32_t alignment() const noexcept {
    return alignment_;
  }

  std::uint64_t data_offset() const noexcept {
    return data_offset_;
  }

  const std::vector<GgufTensorInfo> & tensors() const noexcept {
    return tensors_;
  }

  const GgufTensorInfo * find_tensor(std::string_view tensor_name) const;

  bool read_tensor_bytes(
    std::string_view tensor_name,
    std::vector<std::uint8_t> & out_data,
    std::string & error_message) const;

  bool read_f32_tensor(
    std::string_view tensor_name,
    GgufTensorF32 & out_tensor,
    std::string & error_message) const;

  bool read_q8_0_tensor(
    std::string_view tensor_name,
    GgufTensorQ8_0 & out_tensor,
    std::string & error_message) const;

  bool read_q4_0_tensor(
    std::string_view tensor_name,
    GgufTensorQ4_0 & out_tensor,
    std::string & error_message) const;

private:
  std::unordered_map<std::string, double> model_numbers_;
  std::unordered_map<std::string, std::string> model_strings_;
  std::vector<std::uint32_t> rope_sections_;
  std::uint64_t tokenizer_count_ = 0;
  bool is_open_ = false;
  std::string path_;
  std::uint32_t version_ = 0;
  std::uint64_t file_size_ = 0;
  std::uint64_t metadata_count_ = 0;
  std::uint32_t alignment_ = 32;
  std::uint64_t data_offset_ = 0;
  std::vector<GgufTensorInfo> tensors_;
  std::unordered_map<std::string, std::size_t> tensor_index_;
};

} // namespace qwen35x
