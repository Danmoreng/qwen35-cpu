#include "qwen35x/weights/q4_h128_artifact.h"

#include "qwen35x/cpu/q4_0.h"
#include "qwen35x/cpu/q4_h128.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <limits>
#include <type_traits>

namespace qwen35x {
namespace {

constexpr std::array<char, 8> kMagic{'Q', '3', '5', 'H', '1', '2', '8', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kAlignment = 64;
constexpr std::uint32_t kHeaderSize = 96;
constexpr std::uint64_t kFnvOffset = UINT64_C(14695981039346656037);
constexpr std::uint64_t kFnvPrime = UINT64_C(1099511628211);

template <typename T>
bool write_value(std::ostream & stream, const T value) {
  static_assert(std::is_integral_v<T>);
  stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
  return static_cast<bool>(stream);
}

template <typename T>
bool read_value(std::istream & stream, T & value) {
  static_assert(std::is_integral_v<T>);
  stream.read(reinterpret_cast<char *>(&value), sizeof(value));
  return static_cast<bool>(stream);
}

[[nodiscard]] bool checked_add(
  const std::uint64_t lhs,
  const std::uint64_t rhs,
  std::uint64_t & output) noexcept {
  if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs) {
    return false;
  }
  output = lhs + rhs;
  return true;
}

[[nodiscard]] bool checked_multiply(
  const std::uint64_t lhs,
  const std::uint64_t rhs,
  std::uint64_t & output) noexcept {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return false;
  }
  output = lhs * rhs;
  return true;
}

[[nodiscard]] bool align_up(
  const std::uint64_t value,
  const std::uint64_t alignment,
  std::uint64_t & output) noexcept {
  const std::uint64_t remainder = value % alignment;
  if (remainder == 0) {
    output = value;
    return true;
  }
  return checked_add(value, alignment - remainder, output);
}

[[nodiscard]] std::uint64_t checksum_bytes(
  const void * data,
  const std::size_t byte_count) noexcept {
  const auto * bytes = static_cast<const std::uint8_t *>(data);
  std::uint64_t hash = kFnvOffset;
  for (std::size_t index = 0; index < byte_count; ++index) {
    hash ^= bytes[index];
    hash *= kFnvPrime;
  }
  return hash;
}

[[nodiscard]] bool validate_name_and_shape(
  const Q4H128TensorInfo & tensor,
  std::string & error_message) {
  if (tensor.name.empty() || tensor.name.size() > 4096) {
    error_message = "Q4_H128 tensor name is empty or too long.";
    return false;
  }
  if (tensor.shape.empty() || tensor.shape.size() > 8) {
    error_message = "Q4_H128 tensor rank must be between one and eight: " + tensor.name;
    return false;
  }
  for (const std::uint64_t dimension : tensor.shape) {
    if (dimension == 0) {
      error_message = "Q4_H128 tensor has a zero dimension: " + tensor.name;
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::uint64_t directory_entry_size(const Q4H128TensorInfo & tensor) {
  return 64 + static_cast<std::uint64_t>(tensor.name.size()) +
    static_cast<std::uint64_t>(tensor.shape.size()) * sizeof(std::uint64_t);
}

bool write_zero_padding(std::ostream & stream, std::uint64_t bytes) {
  constexpr std::array<char, 64> zeros{};
  while (bytes != 0) {
    const std::size_t chunk = static_cast<std::size_t>(
      std::min<std::uint64_t>(bytes, zeros.size()));
    stream.write(zeros.data(), static_cast<std::streamsize>(chunk));
    if (!stream) {
      return false;
    }
    bytes -= chunk;
  }
  return true;
}

} // namespace

std::uint64_t q4_h128_payload_size(
  const Q4H128TensorEncoding encoding,
  const std::vector<std::uint64_t> & shape,
  std::string & error_message) {
  if (shape.empty() || shape.size() > 8) {
    error_message = "Q4_H128 payload shape has an unsupported rank.";
    return 0;
  }
  std::uint64_t elements = 1;
  for (const std::uint64_t dimension : shape) {
    if (dimension == 0 || !checked_multiply(elements, dimension, elements)) {
      error_message = "Q4_H128 payload shape is zero or overflows.";
      return 0;
    }
  }
  switch (encoding) {
    case Q4H128TensorEncoding::f32: {
      std::uint64_t bytes = 0;
      if (!checked_multiply(elements, sizeof(float), bytes)) {
        error_message = "F32 Q4_H128 payload byte count overflows.";
        return 0;
      }
      return bytes;
    }
    case Q4H128TensorEncoding::q4_0_cpu_dot4:
    case Q4H128TensorEncoding::q4_h128_cpu_dot4:
    case Q4H128TensorEncoding::q4_0_cpu_x8:
    case Q4H128TensorEncoding::q4_h128_cpu_x8: {
      const std::uint64_t columns = q4_h128_encoding_transformed(encoding)
        ? cpu::q4_h128_transform_size : cpu::q4_0_values_per_block;
      if (shape.size() != 2 || shape[0] % cpu::q4_0_packed_rows != 0 ||
          shape[1] % columns != 0) {
        error_message = "CPU-packed Q4 tensor requires complete 8-row input-aligned tiles.";
        return 0;
      }
      return (elements / cpu::q4_0_values_per_block) * sizeof(cpu::Q4_0Block);
    }
    case Q4H128TensorEncoding::q4_0:
      if (elements % cpu::q4_0_values_per_block != 0) {
        error_message = "Q4_0 payload element count is not divisible by 32.";
        return 0;
      }
      return (elements / cpu::q4_0_values_per_block) * sizeof(cpu::Q4_0Block);
    case Q4H128TensorEncoding::q4_h128:
      if (shape.back() % cpu::q4_h128_transform_size != 0) {
        error_message = "Q4_H128 input dimension is not divisible by 128.";
        return 0;
      }
      return (elements / cpu::q4_0_values_per_block) * sizeof(cpu::Q4_0Block);
  }
  error_message = "Unknown Q4_H128 tensor encoding.";
  return 0;
}

bool Q4H128ArtifactWriter::open(
  const std::string & path,
  const Q4H128ArtifactMetadata & metadata,
  const std::vector<Q4H128TensorInfo> & tensors,
  std::string & error_message) {
  close();
  if (tensors.empty() || metadata.hidden_size == 0 || metadata.vocabulary_size == 0) {
    error_message = "Q4_H128 artifact metadata is incomplete.";
    return false;
  }
  path_ = path;
  metadata_ = metadata;
  tensors_ = tensors;
  tensor_index_.clear();
  written_.assign(tensors.size(), false);
  directory_bytes_ = 0;
  for (std::size_t index = 0; index < tensors_.size(); ++index) {
    Q4H128TensorInfo & tensor = tensors_[index];
    if (!validate_name_and_shape(tensor, error_message) ||
        !tensor_index_.emplace(tensor.name, index).second) {
      if (error_message.empty()) {
        error_message = "Duplicate Q4_H128 tensor name: " + tensor.name;
      }
      close();
      return false;
    }
    tensor.element_count = 1;
    for (const std::uint64_t dimension : tensor.shape) {
      if (!checked_multiply(tensor.element_count, dimension, tensor.element_count)) {
        error_message = "Q4_H128 tensor element count overflows: " + tensor.name;
        close();
        return false;
      }
    }
    tensor.data_size = q4_h128_payload_size(tensor.encoding, tensor.shape, error_message);
    if (tensor.data_size == 0) {
      error_message = "Invalid tensor '" + tensor.name + "': " + error_message;
      close();
      return false;
    }
    if (q4_h128_encoding_transformed(tensor.encoding)) {
      if (tensor.transform_size != cpu::q4_h128_transform_size ||
          tensor.scale_group != cpu::q4_0_values_per_block ||
          tensor.sign_seed != metadata.sign_seed) {
        error_message = "Q4_H128 tensor transform metadata is inconsistent: " + tensor.name;
        close();
        return false;
      }
    } else if (tensor.transform_size != 0 || tensor.sign_seed != 0 ||
               (q4_h128_encoding_cpu_packed(tensor.encoding) &&
                tensor.scale_group != cpu::q4_0_values_per_block)) {
      error_message = "Untransformed tensor has invalid quantization metadata: " + tensor.name;
      close();
      return false;
    }
    if (!checked_add(directory_bytes_, directory_entry_size(tensor), directory_bytes_)) {
      error_message = "Q4_H128 directory size overflows.";
      close();
      return false;
    }
  }

  std::uint64_t directory_end = 0;
  if (!checked_add(kHeaderSize, directory_bytes_, directory_end) ||
      !align_up(directory_end, kAlignment, data_offset_)) {
    error_message = "Q4_H128 directory offset overflows.";
    close();
    return false;
  }
  std::uint64_t cursor = data_offset_;
  for (Q4H128TensorInfo & tensor : tensors_) {
    if (!align_up(cursor, kAlignment, cursor)) {
      error_message = "Q4_H128 payload offset overflows.";
      close();
      return false;
    }
    tensor.data_offset = cursor;
    if (!checked_add(cursor, tensor.data_size, cursor)) {
      error_message = "Q4_H128 artifact size overflows.";
      close();
      return false;
    }
  }

  stream_.open(path, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
  if (!stream_) {
    error_message = "Could not create Q4_H128 artifact: " + path;
    close();
    return false;
  }
  if (!write_header_and_directory(false, error_message)) {
    close();
    return false;
  }
  stream_.seekp(static_cast<std::streamoff>(data_offset_), std::ios::beg);
  if (!stream_ || !write_zero_padding(stream_, cursor - data_offset_)) {
    error_message = "Could not allocate Q4_H128 artifact payload area.";
    close();
    return false;
  }
  stream_.flush();
  return true;
}

bool Q4H128ArtifactWriter::write_header_and_directory(
  const bool include_checksums,
  std::string & error_message) {
  stream_.seekp(0, std::ios::beg);
  stream_.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  const std::uint64_t tensor_count = tensors_.size();
  bool ok = write_value(stream_, kVersion) && write_value(stream_, kEndianMarker) &&
    write_value(stream_, kHeaderSize) && write_value(stream_, kAlignment) &&
    write_value(stream_, tensor_count) && write_value(stream_, std::uint64_t{kHeaderSize}) &&
    write_value(stream_, directory_bytes_) && write_value(stream_, data_offset_) &&
    write_value(stream_, metadata_.sign_seed) &&
    write_value(stream_, metadata_.num_hidden_layers) &&
    write_value(stream_, metadata_.hidden_size) &&
    write_value(stream_, metadata_.intermediate_size) &&
    write_value(stream_, metadata_.vocabulary_size);
  const std::streamoff used = stream_.tellp();
  if (!ok || used < 0 || static_cast<std::uint64_t>(used) > kHeaderSize ||
      !write_zero_padding(stream_, kHeaderSize - static_cast<std::uint64_t>(used))) {
    error_message = "Failed to write Q4_H128 artifact header.";
    return false;
  }
  for (const Q4H128TensorInfo & tensor : tensors_) {
    const std::uint32_t name_size = static_cast<std::uint32_t>(tensor.name.size());
    const std::uint32_t rank = static_cast<std::uint32_t>(tensor.shape.size());
    ok = write_value(stream_, name_size) && write_value(stream_, rank) &&
      write_value(stream_, static_cast<std::uint32_t>(tensor.encoding)) &&
      write_value(stream_, tensor.transform_size) && write_value(stream_, tensor.scale_group) &&
      write_value(stream_, std::uint32_t{0}) && write_value(stream_, tensor.sign_seed) &&
      write_value(stream_, tensor.element_count) && write_value(stream_, tensor.data_offset) &&
      write_value(stream_, tensor.data_size) &&
      write_value(stream_, include_checksums ? tensor.checksum : std::uint64_t{0});
    stream_.write(tensor.name.data(), static_cast<std::streamsize>(tensor.name.size()));
    for (const std::uint64_t dimension : tensor.shape) {
      ok = write_value(stream_, dimension) && ok;
    }
    if (!ok || !stream_) {
      error_message = "Failed to write Q4_H128 tensor directory.";
      return false;
    }
  }
  return true;
}

bool Q4H128ArtifactWriter::write_tensor(
  const std::string_view tensor_name,
  const void * data,
  const std::size_t byte_count,
  std::string & error_message) {
  const auto found = tensor_index_.find(std::string(tensor_name));
  if (!stream_ || found == tensor_index_.end()) {
    error_message = "Q4_H128 writer is closed or tensor is unknown.";
    return false;
  }
  const std::size_t index = found->second;
  Q4H128TensorInfo & tensor = tensors_[index];
  if (written_[index] || data == nullptr || byte_count != tensor.data_size) {
    error_message = "Q4_H128 tensor write is duplicate or has the wrong size: " + tensor.name;
    return false;
  }
  stream_.seekp(static_cast<std::streamoff>(tensor.data_offset), std::ios::beg);
  stream_.write(static_cast<const char *>(data), static_cast<std::streamsize>(byte_count));
  if (!stream_) {
    error_message = "Failed to write Q4_H128 tensor payload: " + tensor.name;
    return false;
  }
  tensor.checksum = checksum_bytes(data, byte_count);
  written_[index] = true;
  return true;
}

bool Q4H128ArtifactWriter::finalize(std::string & error_message) {
  if (!stream_ || std::find(written_.begin(), written_.end(), false) != written_.end()) {
    error_message = "Cannot finalize Q4_H128 artifact before every tensor is written.";
    return false;
  }
  if (!write_header_and_directory(true, error_message)) {
    return false;
  }
  stream_.flush();
  if (!stream_) {
    error_message = "Failed to flush Q4_H128 artifact.";
    return false;
  }
  finalized_ = true;
  stream_.close();
  return true;
}

void Q4H128ArtifactWriter::close() noexcept {
  if (stream_.is_open()) {
    stream_.close();
  }
  path_.clear();
  tensors_.clear();
  tensor_index_.clear();
  written_.clear();
  directory_bytes_ = 0;
  data_offset_ = 0;
  finalized_ = false;
}

bool Q4H128ArtifactReader::open(const std::string & path, std::string & error_message) {
  close();
  stream_.open(path, std::ios::binary);
  if (!stream_) {
    error_message = "Could not open Q4_H128 artifact: " + path;
    return false;
  }
  stream_.seekg(0, std::ios::end);
  const std::streamoff end = stream_.tellg();
  if (end < static_cast<std::streamoff>(kHeaderSize)) {
    error_message = "Q4_H128 artifact is truncated.";
    close();
    return false;
  }
  file_size_ = static_cast<std::uint64_t>(end);
  stream_.seekg(0, std::ios::beg);

  std::array<char, 8> magic{};
  std::uint32_t version = 0;
  std::uint32_t endian = 0;
  std::uint32_t header_size = 0;
  std::uint32_t alignment = 0;
  std::uint64_t tensor_count = 0;
  std::uint64_t directory_offset = 0;
  std::uint64_t directory_bytes = 0;
  std::uint64_t data_offset = 0;
  stream_.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  bool ok = read_value(stream_, version) && read_value(stream_, endian) &&
    read_value(stream_, header_size) && read_value(stream_, alignment) &&
    read_value(stream_, tensor_count) && read_value(stream_, directory_offset) &&
    read_value(stream_, directory_bytes) && read_value(stream_, data_offset) &&
    read_value(stream_, metadata_.sign_seed) &&
    read_value(stream_, metadata_.num_hidden_layers) &&
    read_value(stream_, metadata_.hidden_size) &&
    read_value(stream_, metadata_.intermediate_size) &&
    read_value(stream_, metadata_.vocabulary_size);
  std::uint64_t directory_end = 0;
  if (!ok || magic != kMagic || version != kVersion || endian != kEndianMarker ||
      header_size != kHeaderSize || alignment != kAlignment || tensor_count == 0 ||
      tensor_count > 100000 || directory_offset != kHeaderSize ||
      !checked_add(directory_offset, directory_bytes, directory_end) ||
      directory_end > data_offset || data_offset > file_size_ ||
      metadata_.hidden_size == 0 || metadata_.vocabulary_size == 0) {
    error_message = "Invalid Q4_H128 artifact header.";
    close();
    return false;
  }

  stream_.seekg(static_cast<std::streamoff>(directory_offset), std::ios::beg);
  tensors_.reserve(static_cast<std::size_t>(tensor_count));
  for (std::uint64_t index = 0; index < tensor_count; ++index) {
    Q4H128TensorInfo tensor;
    std::uint32_t name_size = 0;
    std::uint32_t rank = 0;
    std::uint32_t encoding = 0;
    std::uint32_t reserved = 0;
    ok = read_value(stream_, name_size) && read_value(stream_, rank) &&
      read_value(stream_, encoding) && read_value(stream_, tensor.transform_size) &&
      read_value(stream_, tensor.scale_group) && read_value(stream_, reserved) &&
      read_value(stream_, tensor.sign_seed) && read_value(stream_, tensor.element_count) &&
      read_value(stream_, tensor.data_offset) && read_value(stream_, tensor.data_size) &&
      read_value(stream_, tensor.checksum);
    if (!ok || name_size == 0 || name_size > 4096 || rank == 0 || rank > 8 ||
        encoding > static_cast<std::uint32_t>(Q4H128TensorEncoding::q4_h128_cpu_dot4) ||
        reserved != 0) {
      error_message = "Invalid Q4_H128 tensor directory entry.";
      close();
      return false;
    }
    tensor.encoding = static_cast<Q4H128TensorEncoding>(encoding);
    tensor.name.resize(name_size);
    tensor.shape.resize(rank);
    stream_.read(tensor.name.data(), static_cast<std::streamsize>(name_size));
    for (std::uint64_t & dimension : tensor.shape) {
      ok = read_value(stream_, dimension) && ok;
    }
    std::string size_error;
    const std::uint64_t expected_size = q4_h128_payload_size(
      tensor.encoding, tensor.shape, size_error);
    std::uint64_t tensor_end = 0;
    if (!ok || tensor.checksum == 0 || expected_size == 0 ||
        expected_size != tensor.data_size || tensor.data_offset < data_offset ||
        tensor.data_offset % kAlignment != 0 ||
        !checked_add(tensor.data_offset, tensor.data_size, tensor_end) ||
        tensor_end > file_size_ ||
        (q4_h128_encoding_transformed(tensor.encoding) &&
         (tensor.transform_size != cpu::q4_h128_transform_size ||
          tensor.scale_group != cpu::q4_0_values_per_block ||
          tensor.sign_seed != metadata_.sign_seed)) ||
        (!q4_h128_encoding_transformed(tensor.encoding) &&
         (tensor.transform_size != 0 || tensor.sign_seed != 0)) ||
        (q4_h128_encoding_cpu_packed(tensor.encoding) &&
         tensor.scale_group != cpu::q4_0_values_per_block) ||
        !tensor_index_.emplace(tensor.name, tensors_.size()).second) {
      error_message = "Invalid Q4_H128 tensor metadata: " + tensor.name;
      close();
      return false;
    }
    std::uint64_t recomputed_elements = 1;
    for (const std::uint64_t dimension : tensor.shape) {
      if (!checked_multiply(recomputed_elements, dimension, recomputed_elements)) {
        error_message = "Q4_H128 tensor element count overflows: " + tensor.name;
        close();
        return false;
      }
    }
    if (recomputed_elements != tensor.element_count) {
      error_message = "Q4_H128 tensor element count mismatch: " + tensor.name;
      close();
      return false;
    }
    tensors_.push_back(std::move(tensor));
  }
  const std::streamoff directory_position = stream_.tellg();
  if (directory_position < 0 ||
      static_cast<std::uint64_t>(directory_position) != directory_end) {
    error_message = "Q4_H128 tensor directory size mismatch.";
    close();
    return false;
  }

  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  ranges.reserve(tensors_.size());
  for (const Q4H128TensorInfo & tensor : tensors_) {
    ranges.emplace_back(tensor.data_offset, tensor.data_offset + tensor.data_size);
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t index = 1; index < ranges.size(); ++index) {
    if (ranges[index].first < ranges[index - 1].second) {
      error_message = "Q4_H128 tensor payloads overlap.";
      close();
      return false;
    }
  }
  path_ = path;
  return true;
}

const Q4H128TensorInfo * Q4H128ArtifactReader::find_tensor(
  const std::string_view name) const {
  const auto found = tensor_index_.find(std::string(name));
  return found == tensor_index_.end() ? nullptr : &tensors_[found->second];
}

bool Q4H128ArtifactReader::read_tensor_bytes(
  const std::string_view tensor_name,
  std::vector<std::uint8_t> & output,
  std::string & error_message) {
  const Q4H128TensorInfo * tensor = find_tensor(tensor_name);
  if (!stream_ || tensor == nullptr ||
      tensor->data_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    error_message = "Q4_H128 reader is closed, tensor is unknown, or tensor is too large.";
    return false;
  }
  output.resize(static_cast<std::size_t>(tensor->data_size));
  if (!read_tensor_into(tensor_name, output.data(), output.size(), error_message)) {
    output.clear();
    return false;
  }
  return true;
}

bool Q4H128ArtifactReader::read_tensor_into(
  const std::string_view tensor_name,
  void * output,
  const std::size_t byte_count,
  std::string & error_message) {
  const Q4H128TensorInfo * tensor = find_tensor(tensor_name);
  if (!is_open() || tensor == nullptr || output == nullptr ||
      tensor->data_size != byte_count ||
      byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    error_message = "Q4_H128 direct read has an invalid destination or byte count.";
    return false;
  }
  stream_.clear();
  stream_.seekg(static_cast<std::streamoff>(tensor->data_offset), std::ios::beg);
  stream_.read(static_cast<char *>(output), static_cast<std::streamsize>(byte_count));
  if (!stream_ || checksum_bytes(output, byte_count) != tensor->checksum) {
    error_message = "Q4_H128 tensor read or checksum validation failed: " + tensor->name;
    return false;
  }
  return true;
}

void Q4H128ArtifactReader::close() noexcept {
  if (stream_.is_open()) {
    stream_.close();
  }
  path_.clear();
  file_size_ = 0;
  metadata_ = {};
  tensors_.clear();
  tensor_index_.clear();
}

} // namespace qwen35x
