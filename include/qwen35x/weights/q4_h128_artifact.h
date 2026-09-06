#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace qwen35x {

enum class Q4H128TensorEncoding : std::uint32_t {
  f32 = 0,
  q4_0 = 1,
  q4_h128 = 2,
  // CPU-ready Q4_0BlockX8: eight FP16 scales, then 128 interleaved
  // offset-binary nibble bytes. Complete 8-row / 32-column tiles only.
  q4_0_cpu_x8 = 3,
  q4_h128_cpu_x8 = 4,
  q4_0_cpu_dot4 = 5,
  q4_h128_cpu_dot4 = 6,
};

inline bool q4_h128_encoding_transformed(Q4H128TensorEncoding encoding) noexcept {
  return encoding == Q4H128TensorEncoding::q4_h128 ||
    encoding == Q4H128TensorEncoding::q4_h128_cpu_x8 ||
    encoding == Q4H128TensorEncoding::q4_h128_cpu_dot4;
}

inline bool q4_h128_encoding_dot4(Q4H128TensorEncoding encoding) noexcept {
  return encoding == Q4H128TensorEncoding::q4_0_cpu_dot4 ||
    encoding == Q4H128TensorEncoding::q4_h128_cpu_dot4;
}

inline bool q4_h128_encoding_cpu_packed(Q4H128TensorEncoding encoding) noexcept {
  return encoding == Q4H128TensorEncoding::q4_0_cpu_x8 ||
    encoding == Q4H128TensorEncoding::q4_h128_cpu_x8 || q4_h128_encoding_dot4(encoding);
}

struct Q4H128ArtifactMetadata {
  std::uint32_t num_hidden_layers = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t vocabulary_size = 0;
  std::uint64_t sign_seed = 0;
};

struct Q4H128TensorInfo {
  std::string name;
  std::vector<std::uint64_t> shape;
  Q4H128TensorEncoding encoding = Q4H128TensorEncoding::f32;
  std::uint32_t transform_size = 0;
  std::uint32_t scale_group = 0;
  std::uint64_t sign_seed = 0;
  std::uint64_t element_count = 0;
  std::uint64_t data_offset = 0;
  std::uint64_t data_size = 0;
  std::uint64_t checksum = 0;
};

class Q4H128ArtifactWriter {
public:
  bool open(
    const std::string & path,
    const Q4H128ArtifactMetadata & metadata,
    const std::vector<Q4H128TensorInfo> & tensors,
    std::string & error_message);

  bool write_tensor(
    std::string_view tensor_name,
    const void * data,
    std::size_t byte_count,
    std::string & error_message);

  bool finalize(std::string & error_message);
  void close() noexcept;

private:
  bool write_header_and_directory(bool include_checksums, std::string & error_message);

  std::string path_;
  std::fstream stream_;
  Q4H128ArtifactMetadata metadata_;
  std::vector<Q4H128TensorInfo> tensors_;
  std::unordered_map<std::string, std::size_t> tensor_index_;
  std::vector<bool> written_;
  std::uint64_t directory_bytes_ = 0;
  std::uint64_t data_offset_ = 0;
  bool finalized_ = false;
};

class Q4H128ArtifactReader {
public:
  bool open(const std::string & path, std::string & error_message);
  void close() noexcept;

  bool is_open() const noexcept { return stream_.is_open(); }
  const Q4H128ArtifactMetadata & metadata() const noexcept { return metadata_; }
  const std::vector<Q4H128TensorInfo> & tensors() const noexcept { return tensors_; }
  const Q4H128TensorInfo * find_tensor(std::string_view name) const;

  bool read_tensor_bytes(
    std::string_view tensor_name,
    std::vector<std::uint8_t> & output,
    std::string & error_message);

  // Reads directly into the final runtime buffer, with the same checksum
  // validation as read_tensor_bytes. The caller must discard it on failure.
  bool read_tensor_into(
    std::string_view tensor_name,
    void * output,
    std::size_t byte_count,
    std::string & error_message);

private:
  std::string path_;
  std::ifstream stream_;
  std::uint64_t file_size_ = 0;
  Q4H128ArtifactMetadata metadata_;
  std::vector<Q4H128TensorInfo> tensors_;
  std::unordered_map<std::string, std::size_t> tensor_index_;
};

[[nodiscard]] std::uint64_t q4_h128_payload_size(
  Q4H128TensorEncoding encoding,
  const std::vector<std::uint64_t> & shape,
  std::string & error_message);

} // namespace qwen35x
