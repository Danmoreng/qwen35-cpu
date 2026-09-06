#include "qwen35x/weights/gguf.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace qwen35x {

namespace {

constexpr std::uint32_t gguf_version = 3;
constexpr std::uint32_t default_alignment = 32;
constexpr std::uint64_t q4_0_elements_per_block = 32;
constexpr std::uint64_t q4_0_bytes_per_block = 18;
constexpr std::uint64_t q8_0_elements_per_block = 32;
constexpr std::uint64_t q8_0_bytes_per_block = 34;
constexpr std::uint64_t max_serialized_string_bytes = 64ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t max_metadata_entries = 1'000'000;
constexpr std::uint64_t max_tensor_entries = 1'000'000;

static_assert(std::is_trivially_copyable_v<cpu::Q8_0Block>);
static_assert(std::is_trivially_copyable_v<cpu::Q4_0Block>);

enum class GgufValueType : std::uint32_t {
  uint8 = 0,
  int8 = 1,
  uint16 = 2,
  int16 = 3,
  uint32 = 4,
  int32 = 5,
  float32 = 6,
  boolean = 7,
  string = 8,
  array = 9,
  uint64 = 10,
  int64 = 11,
  float64 = 12,
};

bool checked_add_u64(const std::uint64_t a, const std::uint64_t b, std::uint64_t & out) {
  if (a > std::numeric_limits<std::uint64_t>::max() - b) {
    return false;
  }
  out = a + b;
  return true;
}

bool checked_mul_u64(const std::uint64_t a, const std::uint64_t b, std::uint64_t & out) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
    return false;
  }
  out = a * b;
  return true;
}

bool align_up_u64(
  const std::uint64_t value,
  const std::uint32_t alignment,
  std::uint64_t & out) {
  if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
    return false;
  }
  const std::uint64_t mask = static_cast<std::uint64_t>(alignment - 1U);
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  out = (value + mask) & ~mask;
  return true;
}

class FileCursor {
public:
  FileCursor(std::ifstream & stream, const std::uint64_t file_size)
    : stream_(stream), file_size_(file_size) {}

  std::uint64_t offset() const noexcept {
    return offset_;
  }

  std::uint64_t remaining() const noexcept {
    return offset_ <= file_size_ ? file_size_ - offset_ : 0;
  }

  bool read_bytes(
    void * destination,
    const std::uint64_t byte_count,
    const std::string & what,
    std::string & error_message) {
    if (byte_count > remaining()) {
      error_message = "GGUF is truncated while reading " + what + " at byte " + std::to_string(offset_) + ".";
      return false;
    }
    if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
      error_message = "GGUF read size exceeds the stream implementation limit for " + what + ".";
      return false;
    }
    if (byte_count != 0) {
      stream_.read(static_cast<char *>(destination), static_cast<std::streamsize>(byte_count));
      if (!stream_ || stream_.gcount() != static_cast<std::streamsize>(byte_count)) {
        error_message = "Could not read " + what + " from GGUF at byte " + std::to_string(offset_) + ".";
        return false;
      }
    }
    offset_ += byte_count;
    return true;
  }

  bool skip(
    const std::uint64_t byte_count,
    const std::string & what,
    std::string & error_message) {
    if (byte_count > remaining()) {
      error_message = "GGUF is truncated while skipping " + what + " at byte " + std::to_string(offset_) + ".";
      return false;
    }
    if (byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
      error_message = "GGUF skip size exceeds the stream implementation limit for " + what + ".";
      return false;
    }
    if (byte_count != 0) {
      // Tokenizer metadata contains hundreds of thousands of short strings.
      // Reading small skips preserves the stream buffer; seekg for every token
      // would otherwise turn metadata indexing into hundreds of thousands of
      // system calls. Large numeric arrays are still skipped with one seek.
      constexpr std::uint64_t buffered_skip_limit = 4096;
      if (byte_count <= buffered_skip_limit) {
        std::array<char, buffered_skip_limit> discarded;
        stream_.read(discarded.data(), static_cast<std::streamsize>(byte_count));
        if (!stream_ || stream_.gcount() != static_cast<std::streamsize>(byte_count)) {
          error_message = "Could not skip " + what + " in GGUF at byte " + std::to_string(offset_) + ".";
          return false;
        }
      } else {
        stream_.seekg(static_cast<std::streamoff>(byte_count), std::ios::cur);
        if (!stream_) {
          error_message = "Could not skip " + what + " in GGUF at byte " + std::to_string(offset_) + ".";
          return false;
        }
      }
    }
    offset_ += byte_count;
    return true;
  }

  bool read_u32(std::uint32_t & out, const std::string & what, std::string & error_message) {
    std::array<std::uint8_t, 4> bytes{};
    if (!read_bytes(bytes.data(), bytes.size(), what, error_message)) {
      return false;
    }
    out = static_cast<std::uint32_t>(bytes[0]) |
          (static_cast<std::uint32_t>(bytes[1]) << 8U) |
          (static_cast<std::uint32_t>(bytes[2]) << 16U) |
          (static_cast<std::uint32_t>(bytes[3]) << 24U);
    return true;
  }

  bool read_u64(std::uint64_t & out, const std::string & what, std::string & error_message) {
    std::array<std::uint8_t, 8> bytes{};
    if (!read_bytes(bytes.data(), bytes.size(), what, error_message)) {
      return false;
    }
    out = 0;
    for (std::uint32_t i = 0; i < 8; ++i) {
      out |= static_cast<std::uint64_t>(bytes[i]) << (i * 8U);
    }
    return true;
  }

  bool read_string(
    std::string & out,
    const std::string & what,
    std::string & error_message) {
    std::uint64_t length = 0;
    if (!read_u64(length, what + " length", error_message)) {
      return false;
    }
    if (length > max_serialized_string_bytes) {
      error_message = what + " is unreasonably large (" + std::to_string(length) + " bytes).";
      return false;
    }
    if (length > remaining() || length > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      error_message = "Invalid " + what + " length at byte " + std::to_string(offset_ - sizeof(std::uint64_t)) + ".";
      return false;
    }
    out.resize(static_cast<std::size_t>(length));
    return read_bytes(out.data(), length, what, error_message);
  }

  bool skip_string(const std::string & what, std::string & error_message) {
    std::uint64_t length = 0;
    if (!read_u64(length, what + " length", error_message)) {
      return false;
    }
    if (length > max_serialized_string_bytes) {
      error_message = what + " is unreasonably large (" + std::to_string(length) + " bytes).";
      return false;
    }
    return skip(length, what, error_message);
  }

private:
  std::ifstream & stream_;
  std::uint64_t file_size_ = 0;
  std::uint64_t offset_ = 0;
};

bool value_type_width(const GgufValueType type, std::uint64_t & out_width) {
  switch (type) {
    case GgufValueType::uint8:
    case GgufValueType::int8:
    case GgufValueType::boolean:
      out_width = 1;
      return true;
    case GgufValueType::uint16:
    case GgufValueType::int16:
      out_width = 2;
      return true;
    case GgufValueType::uint32:
    case GgufValueType::int32:
    case GgufValueType::float32:
      out_width = 4;
      return true;
    case GgufValueType::uint64:
    case GgufValueType::int64:
    case GgufValueType::float64:
      out_width = 8;
      return true;
    case GgufValueType::string:
    case GgufValueType::array:
      return false;
  }
  return false;
}

bool valid_value_type(const std::uint32_t raw_type) {
  return raw_type <= static_cast<std::uint32_t>(GgufValueType::float64);
}

bool skip_array_payload(
  FileCursor & cursor,
  const GgufValueType element_type,
  const std::uint64_t element_count,
  const std::string & key,
  std::string & error_message) {
  if (element_type == GgufValueType::array) {
    error_message = "GGUF metadata array '" + key + "' has the forbidden nested-array element type.";
    return false;
  }

  std::uint64_t width = 0;
  if (value_type_width(element_type, width)) {
    std::uint64_t byte_count = 0;
    if (!checked_mul_u64(element_count, width, byte_count)) {
      error_message = "GGUF metadata array '" + key + "' byte size overflows uint64.";
      return false;
    }
    return cursor.skip(byte_count, "metadata array '" + key + "'", error_message);
  }

  if (element_type != GgufValueType::string) {
    error_message = "GGUF metadata array '" + key + "' has an unsupported element type.";
    return false;
  }

  // Every serialized string consumes at least its uint64 length prefix. This
  // bound prevents a malicious count from causing an effectively unbounded loop.
  if (element_count > cursor.remaining() / sizeof(std::uint64_t)) {
    error_message = "GGUF metadata string array '" + key + "' cannot fit in the remaining file.";
    return false;
  }
  for (std::uint64_t i = 0; i < element_count; ++i) {
    if (!cursor.skip_string(
          "metadata string array '" + key + "' element " + std::to_string(i),
          error_message)) {
      return false;
    }
  }
  return true;
}

bool skip_metadata_value(
  FileCursor & cursor,
  const GgufValueType type,
  const std::string & key,
  std::string & error_message) {
  std::uint64_t width = 0;
  if (value_type_width(type, width)) {
    return cursor.skip(width, "metadata value '" + key + "'", error_message);
  }
  if (type == GgufValueType::string) {
    return cursor.skip_string("metadata string '" + key + "'", error_message);
  }
  if (type != GgufValueType::array) {
    error_message = "GGUF metadata key '" + key + "' has an unsupported value type.";
    return false;
  }

  std::uint32_t raw_element_type = 0;
  std::uint64_t element_count = 0;
  if (!cursor.read_u32(raw_element_type, "metadata array element type", error_message) ||
      !cursor.read_u64(element_count, "metadata array element count", error_message)) {
    return false;
  }
  if (!valid_value_type(raw_element_type)) {
    error_message = "GGUF metadata array '" + key + "' has invalid element type " +
                    std::to_string(raw_element_type) + ".";
    return false;
  }
  return skip_array_payload(
    cursor,
    static_cast<GgufValueType>(raw_element_type),
    element_count,
    key,
    error_message);
}

bool compute_tensor_layout(GgufTensorInfo & info, std::string & error_message) {
  std::uint64_t elements = 1;
  for (const std::uint64_t dimension : info.ggml_shape) {
    if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      error_message = "Tensor '" + info.name + "' has a dimension larger than INT64_MAX.";
      return false;
    }
    if (!checked_mul_u64(elements, dimension, elements)) {
      error_message = "Tensor '" + info.name + "' element count overflows uint64.";
      return false;
    }
  }
  info.element_count = elements;

  if (info.is_f32()) {
    if (!checked_mul_u64(elements, sizeof(float), info.data_size)) {
      error_message = "F32 tensor '" + info.name + "' byte size overflows uint64.";
      return false;
    }
    return true;
  }

  if (info.ggml_type >= 12 && info.ggml_type <= 14) {
    if (info.ggml_shape.empty() || info.ggml_shape.front() % 256 != 0) {
      error_message = "K-quant tensor has a contiguous dimension not divisible by 256: " + info.name;
      return false;
    }
    const std::uint64_t bytes = info.ggml_type == 12 ? 144 : info.ggml_type == 13 ? 176 : 210;
    if (!checked_mul_u64(elements / 256, bytes, info.data_size)) {
      error_message = "K-quant tensor byte size overflow: " + info.name;
      return false;
    }
    return true;
  }

  if (info.is_q8_0()) {
    if (info.ggml_shape.empty()) {
      error_message = "Q8_0 tensor '" + info.name + "' has no contiguous dimension.";
      return false;
    }
    if (info.ggml_shape.front() % q8_0_elements_per_block != 0) {
      error_message = "Q8_0 tensor '" + info.name + "' has a contiguous dimension that is not divisible by 32.";
      return false;
    }
    const std::uint64_t block_count = elements / q8_0_elements_per_block;
    if (!checked_mul_u64(block_count, q8_0_bytes_per_block, info.data_size)) {
      error_message = "Q8_0 tensor '" + info.name + "' byte size overflows uint64.";
      return false;
    }
    return true;
  }

  if (info.is_q4_0()) {
    if (info.ggml_shape.empty()) {
      error_message = "Q4_0 tensor '" + info.name + "' has no contiguous dimension.";
      return false;
    }
    if (info.ggml_shape.front() % q4_0_elements_per_block != 0) {
      error_message = "Q4_0 tensor '" + info.name +
        "' has a contiguous dimension that is not divisible by 32.";
      return false;
    }
    const std::uint64_t block_count = elements / q4_0_elements_per_block;
    if (!checked_mul_u64(block_count, q4_0_bytes_per_block, info.data_size)) {
      error_message = "Q4_0 tensor '" + info.name + "' byte size overflows uint64.";
      return false;
    }
    return true;
  }

  error_message = "Tensor '" + info.name + "' uses unsupported GGML payload type " +
                  std::to_string(info.ggml_type) +
                  " (supported: F32, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K).";
  return false;
}

bool get_file_size(std::ifstream & stream, std::uint64_t & out_size, std::string & error_message) {
  stream.seekg(0, std::ios::end);
  const std::streampos end = stream.tellg();
  if (end < 0) {
    error_message = "Could not determine GGUF file size.";
    return false;
  }
  const auto end_offset = static_cast<std::streamoff>(end);
  out_size = static_cast<std::uint64_t>(end_offset);
  stream.seekg(0, std::ios::beg);
  if (!stream) {
    error_message = "Could not seek to the beginning of the GGUF file.";
    return false;
  }
  return true;
}

bool read_file_range(
  const std::string & path,
  const std::uint64_t file_offset,
  void * destination,
  const std::uint64_t byte_count,
  std::string & error_message) {
  if (file_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
      byte_count > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max())) {
    error_message = "GGUF tensor range exceeds the stream implementation limit.";
    return false;
  }

  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream) {
    error_message = "Could not reopen GGUF file: " + path;
    return false;
  }
  stream.seekg(static_cast<std::streamoff>(file_offset), std::ios::beg);
  if (!stream) {
    error_message = "Could not seek to GGUF tensor payload at byte " + std::to_string(file_offset) + ".";
    return false;
  }
  if (byte_count != 0) {
    stream.read(static_cast<char *>(destination), static_cast<std::streamsize>(byte_count));
    if (!stream || stream.gcount() != static_cast<std::streamsize>(byte_count)) {
      error_message = "Could not read complete GGUF tensor payload at byte " + std::to_string(file_offset) + ".";
      return false;
    }
  }
  return true;
}

std::string tensor_not_found_message(const std::string_view tensor_name) {
  return "Tensor '" + std::string(tensor_name) + "' was not found in the GGUF index.";
}

} // namespace

void GgufReader::close() noexcept {
  is_open_ = false;
  path_.clear();
  version_ = 0;
  file_size_ = 0;
  metadata_count_ = 0;
  alignment_ = default_alignment;
  data_offset_ = 0;
  tensors_.clear();
  tensor_index_.clear();
}

bool GgufReader::open(const std::string & gguf_file, std::string & error_message) {
  close();
  error_message.clear();

  try {
    std::ifstream stream(gguf_file, std::ios::in | std::ios::binary);
    if (!stream) {
      error_message = "Could not open GGUF file: " + gguf_file;
      return false;
    }

    std::uint64_t parsed_file_size = 0;
    if (!get_file_size(stream, parsed_file_size, error_message)) {
      return false;
    }
    FileCursor cursor(stream, parsed_file_size);

    std::array<char, 4> magic{};
    std::uint32_t parsed_version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t metadata_count = 0;
    if (!cursor.read_bytes(magic.data(), magic.size(), "GGUF magic", error_message) ||
        !cursor.read_u32(parsed_version, "GGUF version", error_message) ||
        !cursor.read_u64(tensor_count, "GGUF tensor count", error_message) ||
        !cursor.read_u64(metadata_count, "GGUF metadata count", error_message)) {
      return false;
    }
    if (magic != std::array<char, 4>{'G', 'G', 'U', 'F'}) {
      error_message = "File does not start with GGUF magic.";
      return false;
    }
    if (parsed_version != gguf_version) {
      error_message = "Unsupported GGUF version " + std::to_string(parsed_version) + "; expected version 3.";
      return false;
    }

    // Even an empty-name tensor info occupies 24 bytes. This is a cheap bound
    // against corrupt counts before reserving or entering a large loop.
    if (tensor_count > max_tensor_entries ||
        tensor_count > parsed_file_size / 24U ||
        tensor_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(GgufTensorInfo))) {
      error_message = "GGUF tensor count cannot fit in the file or in memory.";
      return false;
    }
    // A KV record needs at least an 8-byte key length, a 4-byte type and one
    // byte of scalar payload. Arrays and strings only make it larger.
    if (metadata_count > max_metadata_entries ||
        metadata_count > parsed_file_size / 13U ||
        metadata_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      error_message = "GGUF metadata count cannot fit in the file or in memory.";
      return false;
    }

    std::uint32_t parsed_alignment = default_alignment;
    std::unordered_set<std::string> metadata_keys;
    metadata_keys.reserve(static_cast<std::size_t>(metadata_count));
    for (std::uint64_t i = 0; i < metadata_count; ++i) {
      std::string key;
      std::uint32_t raw_type = 0;
      if (!cursor.read_string(key, "metadata key " + std::to_string(i), error_message) ||
          !cursor.read_u32(raw_type, "metadata value type for '" + key + "'", error_message)) {
        return false;
      }
      if (key.empty()) {
        error_message = "GGUF metadata key " + std::to_string(i) + " is empty.";
        return false;
      }
      if (!metadata_keys.emplace(key).second) {
        error_message = "Duplicate GGUF metadata key '" + key + "'.";
        return false;
      }
      if (!valid_value_type(raw_type)) {
        error_message = "GGUF metadata key '" + key + "' has invalid value type " + std::to_string(raw_type) + ".";
        return false;
      }

      const auto type = static_cast<GgufValueType>(raw_type);
      if (key == "general.alignment") {
        if (type != GgufValueType::uint32) {
          error_message = "GGUF general.alignment must have UINT32 metadata type.";
          return false;
        }
        if (!cursor.read_u32(parsed_alignment, "general.alignment", error_message)) {
          return false;
        }
      } else if (!skip_metadata_value(cursor, type, key, error_message)) {
        return false;
      }
    }

    if (parsed_alignment == 0 || (parsed_alignment & (parsed_alignment - 1U)) != 0) {
      error_message = "GGUF alignment " + std::to_string(parsed_alignment) + " is not a nonzero power of two.";
      return false;
    }

    std::vector<GgufTensorInfo> parsed_tensors;
    std::unordered_map<std::string, std::size_t> parsed_index;
    parsed_tensors.reserve(static_cast<std::size_t>(tensor_count));
    parsed_index.reserve(static_cast<std::size_t>(tensor_count));

    std::uint64_t expected_relative_offset = 0;
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
      GgufTensorInfo info;
      std::uint32_t dimension_count = 0;
      if (!cursor.read_string(info.name, "tensor name " + std::to_string(i), error_message) ||
          !cursor.read_u32(dimension_count, "dimension count for tensor '" + info.name + "'", error_message)) {
        return false;
      }
      if (info.name.empty()) {
        error_message = "GGUF tensor " + std::to_string(i) + " has an empty name.";
        return false;
      }
      if (dimension_count > 4) {
        error_message = "Tensor '" + info.name + "' has more than four dimensions.";
        return false;
      }
      info.ggml_shape.resize(dimension_count);
      for (std::uint32_t dimension = 0; dimension < dimension_count; ++dimension) {
        if (!cursor.read_u64(
              info.ggml_shape[dimension],
              "dimension " + std::to_string(dimension) + " for tensor '" + info.name + "'",
              error_message)) {
          return false;
        }
      }
      info.shape.assign(info.ggml_shape.rbegin(), info.ggml_shape.rend());
      if (!cursor.read_u32(info.ggml_type, "GGML type for tensor '" + info.name + "'", error_message) ||
          !cursor.read_u64(info.relative_offset, "data offset for tensor '" + info.name + "'", error_message)) {
        return false;
      }
      if (!compute_tensor_layout(info, error_message)) {
        return false;
      }
      if (info.relative_offset % parsed_alignment != 0) {
        error_message = "Tensor '" + info.name + "' relative offset is not GGUF-aligned.";
        return false;
      }
      if (info.relative_offset != expected_relative_offset) {
        error_message = "Tensor '" + info.name + "' has relative offset " +
                        std::to_string(info.relative_offset) + ", expected " +
                        std::to_string(expected_relative_offset) + ".";
        return false;
      }

      std::uint64_t payload_end = 0;
      if (!checked_add_u64(info.relative_offset, info.data_size, payload_end) ||
          !align_up_u64(payload_end, parsed_alignment, expected_relative_offset)) {
        error_message = "Tensor '" + info.name + "' padded data range overflows uint64.";
        return false;
      }

      const std::size_t tensor_index = parsed_tensors.size();
      if (!parsed_index.emplace(info.name, tensor_index).second) {
        error_message = "Duplicate GGUF tensor name '" + info.name + "'.";
        return false;
      }
      parsed_tensors.push_back(std::move(info));
    }

    std::uint64_t parsed_data_offset = cursor.offset();
    if (tensor_count != 0 && !align_up_u64(parsed_data_offset, parsed_alignment, parsed_data_offset)) {
      error_message = "GGUF tensor data offset overflows uint64 during alignment.";
      return false;
    }
    if (parsed_data_offset > parsed_file_size) {
      error_message = "GGUF tensor data section starts beyond the end of the file.";
      return false;
    }

    for (GgufTensorInfo & info : parsed_tensors) {
      if (!checked_add_u64(parsed_data_offset, info.relative_offset, info.data_offset)) {
        error_message = "Absolute data offset for tensor '" + info.name + "' overflows uint64.";
        return false;
      }
      std::uint64_t tensor_end = 0;
      if (!checked_add_u64(info.data_offset, info.data_size, tensor_end) || tensor_end > parsed_file_size) {
        error_message = "Tensor '" + info.name + "' payload extends beyond the GGUF file.";
        return false;
      }
    }

    path_ = gguf_file;
    version_ = parsed_version;
    file_size_ = parsed_file_size;
    metadata_count_ = metadata_count;
    alignment_ = parsed_alignment;
    data_offset_ = parsed_data_offset;
    tensors_ = std::move(parsed_tensors);
    tensor_index_ = std::move(parsed_index);
    is_open_ = true;
    return true;
  } catch (const std::bad_alloc &) {
    close();
    error_message = "Out of memory while parsing GGUF index.";
    return false;
  } catch (const std::length_error & exception) {
    close();
    error_message = std::string("Invalid oversized GGUF field: ") + exception.what();
    return false;
  } catch (const std::exception & exception) {
    close();
    error_message = std::string("Unexpected error while parsing GGUF: ") + exception.what();
    return false;
  }
}

const GgufTensorInfo * GgufReader::find_tensor(const std::string_view tensor_name) const {
  const auto found = tensor_index_.find(std::string(tensor_name));
  if (found == tensor_index_.end()) {
    return nullptr;
  }
  return &tensors_[found->second];
}

bool GgufReader::read_tensor_bytes(
  const std::string_view tensor_name,
  std::vector<std::uint8_t> & out_data,
  std::string & error_message) const {
  out_data.clear();
  error_message.clear();
  if (!is_open_) {
    error_message = "No GGUF file is open.";
    return false;
  }
  const GgufTensorInfo * info = find_tensor(tensor_name);
  if (info == nullptr) {
    error_message = tensor_not_found_message(tensor_name);
    return false;
  }
  if (info->data_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    error_message = "Tensor '" + info->name + "' is too large for an owned byte buffer.";
    return false;
  }
  try {
    out_data.resize(static_cast<std::size_t>(info->data_size));
  } catch (const std::bad_alloc &) {
    error_message = "Out of memory while allocating tensor '" + info->name + "'.";
    return false;
  } catch (const std::length_error &) {
    error_message = "Tensor '" + info->name + "' is too large for an owned byte buffer.";
    return false;
  }
  if (!read_file_range(path_, info->data_offset, out_data.data(), info->data_size, error_message)) {
    out_data.clear();
    return false;
  }
  return true;
}

bool GgufReader::read_f32_tensor(
  const std::string_view tensor_name,
  GgufTensorF32 & out_tensor,
  std::string & error_message) const {
  out_tensor = {};
  error_message.clear();
  if (!is_open_) {
    error_message = "No GGUF file is open.";
    return false;
  }
  const GgufTensorInfo * info = find_tensor(tensor_name);
  if (info == nullptr) {
    error_message = tensor_not_found_message(tensor_name);
    return false;
  }
  if (!info->is_f32()) {
    error_message = "Tensor '" + info->name + "' is not F32 (GGML type " + std::to_string(info->ggml_type) + ").";
    return false;
  }
  if (info->element_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    error_message = "F32 tensor '" + info->name + "' is too large for an owned vector.";
    return false;
  }

  try {
    out_tensor.name = info->name;
    out_tensor.shape = info->shape;
    out_tensor.data.resize(static_cast<std::size_t>(info->element_count));
  } catch (const std::bad_alloc &) {
    out_tensor = {};
    error_message = "Out of memory while allocating F32 tensor '" + info->name + "'.";
    return false;
  } catch (const std::length_error &) {
    out_tensor = {};
    error_message = "F32 tensor '" + info->name + "' is too large for an owned vector.";
    return false;
  }

  if (!read_file_range(path_, info->data_offset, out_tensor.data.data(), info->data_size, error_message)) {
    out_tensor = {};
    return false;
  }

  if constexpr (std::endian::native == std::endian::big) {
    for (float & value : out_tensor.data) {
      std::array<std::uint8_t, sizeof(float)> bytes{};
      std::memcpy(bytes.data(), &value, sizeof(float));
      std::reverse(bytes.begin(), bytes.end());
      std::memcpy(&value, bytes.data(), sizeof(float));
    }
  }
  return true;
}

bool GgufReader::read_q8_0_tensor(
  const std::string_view tensor_name,
  GgufTensorQ8_0 & out_tensor,
  std::string & error_message) const {
  out_tensor = {};
  error_message.clear();
  if (!is_open_) {
    error_message = "No GGUF file is open.";
    return false;
  }
  const GgufTensorInfo * info = find_tensor(tensor_name);
  if (info == nullptr) {
    error_message = tensor_not_found_message(tensor_name);
    return false;
  }
  if (!info->is_q8_0()) {
    error_message = "Tensor '" + info->name + "' is not Q8_0 (GGML type " + std::to_string(info->ggml_type) + ").";
    return false;
  }
  const std::uint64_t block_count = info->data_size / sizeof(cpu::Q8_0Block);
  if (block_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    error_message = "Q8_0 tensor '" + info->name + "' is too large for an owned vector.";
    return false;
  }

  try {
    out_tensor.name = info->name;
    out_tensor.shape = info->shape;
    out_tensor.blocks.resize(static_cast<std::size_t>(block_count));
  } catch (const std::bad_alloc &) {
    out_tensor = {};
    error_message = "Out of memory while allocating Q8_0 tensor '" + info->name + "'.";
    return false;
  } catch (const std::length_error &) {
    out_tensor = {};
    error_message = "Q8_0 tensor '" + info->name + "' is too large for an owned vector.";
    return false;
  }

  if (!read_file_range(path_, info->data_offset, out_tensor.blocks.data(), info->data_size, error_message)) {
    out_tensor = {};
    return false;
  }

  if constexpr (std::endian::native == std::endian::big) {
    for (cpu::Q8_0Block & block : out_tensor.blocks) {
      block.d = static_cast<std::uint16_t>((block.d >> 8U) | (block.d << 8U));
    }
  }
  return true;
}

bool GgufReader::read_q4_0_tensor(
  const std::string_view tensor_name,
  GgufTensorQ4_0 & out_tensor,
  std::string & error_message) const {
  out_tensor = {};
  error_message.clear();
  if (!is_open_) {
    error_message = "No GGUF file is open.";
    return false;
  }
  const GgufTensorInfo * info = find_tensor(tensor_name);
  if (info == nullptr) {
    error_message = tensor_not_found_message(tensor_name);
    return false;
  }
  if (!info->is_q4_0()) {
    error_message = "Tensor '" + info->name + "' is not Q4_0 (GGML type " +
      std::to_string(info->ggml_type) + ").";
    return false;
  }
  const std::uint64_t block_count = info->data_size / sizeof(cpu::Q4_0Block);
  if (block_count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    error_message = "Q4_0 tensor '" + info->name + "' is too large for an owned vector.";
    return false;
  }
  try {
    out_tensor.name = info->name;
    out_tensor.shape = info->shape;
    out_tensor.blocks.resize(static_cast<std::size_t>(block_count));
  } catch (const std::bad_alloc &) {
    out_tensor = {};
    error_message = "Out of memory while allocating Q4_0 tensor '" + info->name + "'.";
    return false;
  } catch (const std::length_error &) {
    out_tensor = {};
    error_message = "Q4_0 tensor '" + info->name + "' is too large for an owned vector.";
    return false;
  }
  if (!read_file_range(
        path_, info->data_offset, out_tensor.blocks.data(), info->data_size, error_message)) {
    out_tensor = {};
    return false;
  }
  if constexpr (std::endian::native == std::endian::big) {
    for (cpu::Q4_0Block & block : out_tensor.blocks) {
      block.d = static_cast<std::uint16_t>((block.d >> 8U) | (block.d << 8U));
    }
  }
  return true;
}

} // namespace qwen35x
