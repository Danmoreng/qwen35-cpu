#include "qwen35x/weights/safetensors.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

namespace qwen35x {

namespace {

std::optional<std::string> read_text_file(const std::filesystem::path & path, std::string & error_message) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    error_message = "Could not open file: " + path.string();
    return std::nullopt;
  }

  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::vector<std::int64_t> parse_int64_list(const std::string & list_text) {
  std::vector<std::int64_t> values;
  std::stringstream ss(list_text);
  std::string token;
  while (std::getline(ss, token, ',')) {
    const auto begin = token.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      continue;
    }
    const auto end = token.find_last_not_of(" \t\r\n");
    values.push_back(std::stoll(token.substr(begin, end - begin + 1)));
  }
  return values;
}

bool find_first_safetensors_file(const std::filesystem::path & model_dir, std::string & out_file) {
  std::vector<std::filesystem::path> matches;
  for (const auto & entry : std::filesystem::directory_iterator(model_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.find(".safetensors") != std::string::npos) {
      matches.push_back(entry.path());
    }
  }
  std::sort(matches.begin(), matches.end());
  if (matches.empty()) {
    return false;
  }
  out_file = matches.front().string();
  return true;
}

bool find_json_key(const std::string & json, const std::string & key, std::size_t & key_pos) {
  const std::string marker = "\"" + key + "\"";
  key_pos = json.find(marker);
  return key_pos != std::string::npos;
}

std::optional<std::string> extract_json_string_value_for_key(const std::string & json, const std::string & key) {
  std::size_t key_pos = 0;
  if (!find_json_key(json, key, key_pos)) {
    return std::nullopt;
  }

  const std::size_t colon_pos = json.find(':', key_pos);
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t open_quote = json.find('"', colon_pos + 1);
  if (open_quote == std::string::npos) {
    return std::nullopt;
  }

  const std::size_t close_quote = json.find('"', open_quote + 1);
  if (close_quote == std::string::npos) {
    return std::nullopt;
  }

  return json.substr(open_quote + 1, close_quote - open_quote - 1);
}

std::optional<std::string> extract_enclosed(
  const std::string & text,
  std::size_t start_pos,
  char open_char,
  char close_char) {
  if (start_pos >= text.size() || text[start_pos] != open_char) {
    return std::nullopt;
  }

  bool in_string = false;
  bool escaped = false;
  int depth = 0;
  for (std::size_t i = start_pos; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }

    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == open_char) {
      ++depth;
      continue;
    }
    if (c == close_char) {
      --depth;
      if (depth == 0) {
        return text.substr(start_pos, i - start_pos + 1);
      }
    }
  }

  return std::nullopt;
}

std::optional<std::string> extract_json_object_for_key(const std::string & json, const std::string & key) {
  std::size_t key_pos = 0;
  if (!find_json_key(json, key, key_pos)) {
    return std::nullopt;
  }

  const std::size_t colon_pos = json.find(':', key_pos);
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t open_pos = json.find('{', colon_pos + 1);
  if (open_pos == std::string::npos) {
    return std::nullopt;
  }

  return extract_enclosed(json, open_pos, '{', '}');
}

std::optional<std::string> extract_json_array_for_key(const std::string & json, const std::string & key) {
  std::size_t key_pos = 0;
  if (!find_json_key(json, key, key_pos)) {
    return std::nullopt;
  }

  const std::size_t colon_pos = json.find(':', key_pos);
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t open_pos = json.find('[', colon_pos + 1);
  if (open_pos == std::string::npos) {
    return std::nullopt;
  }

  return extract_enclosed(json, open_pos, '[', ']');
}

std::size_t dtype_size_bytes(const std::string & dtype) {
  if (dtype == "BF16") {
    return 2;
  }
  if (dtype == "F16") {
    return 2;
  }
  if (dtype == "F32") {
    return 4;
  }
  return 0;
}

bool compute_tensor_elements(const std::vector<std::int64_t> & shape, std::size_t & out_elements, std::string & error_message) {
  std::size_t elements = 1;
  for (const std::int64_t dim : shape) {
    if (dim <= 0) {
      error_message = "Invalid tensor shape dimension.";
      return false;
    }
    if (elements > (std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))) {
      error_message = "Tensor element count overflows size_t.";
      return false;
    }
    elements *= static_cast<std::size_t>(dim);
  }
  out_elements = elements;
  return true;
}

float bf16_to_float(std::uint16_t bf16) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bf16) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

float f16_to_float(std::uint16_t f16) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(f16) >> 15) & 0x1;
  const std::uint32_t exp = (static_cast<std::uint32_t>(f16) >> 10) & 0x1F;
  const std::uint32_t mant = static_cast<std::uint32_t>(f16) & 0x3FF;

  std::uint32_t out_bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      out_bits = sign << 31;
    } else {
      std::uint32_t mant_shifted = mant;
      int e = -1;
      do {
        ++e;
        mant_shifted <<= 1;
      } while ((mant_shifted & 0x400U) == 0U);
      mant_shifted &= 0x3FFU;
      const std::uint32_t exp32 = static_cast<std::uint32_t>(127 - 15 - e);
      out_bits = (sign << 31) | (exp32 << 23) | (mant_shifted << 13);
    }
  } else if (exp == 0x1F) {
    out_bits = (sign << 31) | 0x7F800000U | (mant << 13);
  } else {
    const std::uint32_t exp32 = exp + (127 - 15);
    out_bits = (sign << 31) | (exp32 << 23) | (mant << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &out_bits, sizeof(float));
  return out;
}

} // namespace

bool SafetensorLoader::resolve_tensor_file(
  const std::string & model_dir,
  const std::string & tensor_name,
  std::string & out_file,
  std::string & error_message) {
  namespace fs = std::filesystem;
  const fs::path root(model_dir);
  if (!fs::exists(root) || !fs::is_directory(root)) {
    error_message = "Model directory does not exist: " + model_dir;
    return false;
  }

  const fs::path index_path = root / "model.safetensors.index.json";
  if (!fs::exists(index_path) || !fs::is_regular_file(index_path)) {
    if (find_first_safetensors_file(root, out_file)) {
      return true;
    }
    error_message = "No safetensors index or shard files found in: " + model_dir;
    return false;
  }

  const auto index_text = read_text_file(index_path, error_message);
  if (!index_text) {
    return false;
  }

  const auto shard = extract_json_string_value_for_key(*index_text, tensor_name);
  if (!shard) {
    error_message = "Tensor '" + tensor_name + "' not found in model.safetensors.index.json";
    return false;
  }

  const fs::path shard_path = root / *shard;
  if (!fs::exists(shard_path) || !fs::is_regular_file(shard_path)) {
    error_message = "Indexed shard file does not exist: " + shard_path.string();
    return false;
  }

  out_file = shard_path.string();
  return true;
}

bool SafetensorLoader::load_tensor_info(
  const std::string & safetensors_file,
  const std::string & tensor_name,
  SafetensorTensorInfo & out_info,
  std::string & error_message) {
  std::ifstream in(safetensors_file, std::ios::binary);
  if (!in) {
    error_message = "Could not open safetensors shard: " + safetensors_file;
    return false;
  }

  std::uint64_t header_len = 0;
  in.read(reinterpret_cast<char *>(&header_len), sizeof(header_len));
  if (!in || header_len == 0) {
    error_message = "Invalid safetensors header length in: " + safetensors_file;
    return false;
  }

  std::string header_json;
  header_json.resize(static_cast<std::size_t>(header_len));
  in.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!in) {
    error_message = "Failed to read safetensors header JSON from: " + safetensors_file;
    return false;
  }

  const auto tensor_obj = extract_json_object_for_key(header_json, tensor_name);
  if (!tensor_obj) {
    error_message = "Tensor '" + tensor_name + "' not found in safetensors header.";
    return false;
  }

  const auto dtype = extract_json_string_value_for_key(*tensor_obj, "dtype");
  const auto shape = extract_json_array_for_key(*tensor_obj, "shape");
  const auto offsets = extract_json_array_for_key(*tensor_obj, "data_offsets");
  if (!dtype || !shape || !offsets) {
    error_message = "Tensor metadata is incomplete for '" + tensor_name + "'";
    return false;
  }

  const auto shape_vals = parse_int64_list(shape->substr(1, shape->size() - 2));
  const auto offset_vals = parse_int64_list(offsets->substr(1, offsets->size() - 2));
  if (offset_vals.size() != 2 || offset_vals[0] < 0 || offset_vals[1] <= offset_vals[0]) {
    error_message = "Invalid data_offsets for tensor '" + tensor_name + "'";
    return false;
  }

  out_info.name = tensor_name;
  out_info.dtype = *dtype;
  out_info.shape = shape_vals;
  out_info.data_start = 8 + header_len + static_cast<std::uint64_t>(offset_vals[0]);
  out_info.data_end = 8 + header_len + static_cast<std::uint64_t>(offset_vals[1]);
  return true;
}

bool SafetensorLoader::read_bf16_tensor(
  const std::string & safetensors_file,
  const SafetensorTensorInfo & info,
  std::vector<std::uint16_t> & out_data,
  std::string & error_message) {
  if (info.dtype != "BF16") {
    error_message = "Tensor '" + info.name + "' has dtype " + info.dtype + " (expected BF16).";
    return false;
  }

  const std::uint64_t byte_count = info.data_end - info.data_start;
  if ((byte_count % 2) != 0) {
    error_message = "Tensor '" + info.name + "' BF16 byte count is not divisible by 2.";
    return false;
  }

  out_data.resize(static_cast<std::size_t>(byte_count / 2));

  std::ifstream in(safetensors_file, std::ios::binary);
  if (!in) {
    error_message = "Could not open safetensors shard: " + safetensors_file;
    return false;
  }

  in.seekg(static_cast<std::streamoff>(info.data_start), std::ios::beg);
  if (!in) {
    error_message = "Failed to seek to tensor data in: " + safetensors_file;
    return false;
  }

  in.read(reinterpret_cast<char *>(out_data.data()), static_cast<std::streamsize>(byte_count));
  if (!in) {
    error_message = "Failed to read tensor data from: " + safetensors_file;
    return false;
  }

  return true;
}

bool SafetensorLoader::read_tensor_f32(
  const std::string & model_dir,
  const std::string & tensor_name,
  SafetensorTensorF32 & out_tensor,
  std::string & error_message) {
  std::string tensor_file;
  if (!resolve_tensor_file(model_dir, tensor_name, tensor_file, error_message)) {
    return false;
  }

  SafetensorTensorInfo info;
  if (!load_tensor_info(tensor_file, tensor_name, info, error_message)) {
    return false;
  }

  const std::size_t dtype_bytes = dtype_size_bytes(info.dtype);
  if (dtype_bytes == 0) {
    error_message = "Unsupported safetensors dtype '" + info.dtype + "' for tensor '" + tensor_name + "'";
    return false;
  }

  std::size_t elements = 0;
  if (!compute_tensor_elements(info.shape, elements, error_message)) {
    error_message = "Invalid shape for tensor '" + tensor_name + "': " + error_message;
    return false;
  }

  const std::uint64_t byte_count = info.data_end - info.data_start;
  const std::uint64_t expected_bytes = static_cast<std::uint64_t>(elements) * static_cast<std::uint64_t>(dtype_bytes);
  if (byte_count != expected_bytes) {
    error_message = "Tensor '" + tensor_name + "' byte size mismatch.";
    return false;
  }

  std::vector<std::uint8_t> raw(static_cast<std::size_t>(byte_count));
  std::ifstream in(tensor_file, std::ios::binary);
  if (!in) {
    error_message = "Could not open safetensors shard: " + tensor_file;
    return false;
  }
  in.seekg(static_cast<std::streamoff>(info.data_start), std::ios::beg);
  if (!in) {
    error_message = "Failed to seek tensor '" + tensor_name + "' in shard.";
    return false;
  }
  in.read(reinterpret_cast<char *>(raw.data()), static_cast<std::streamsize>(raw.size()));
  if (!in) {
    error_message = "Failed to read tensor '" + tensor_name + "' from shard.";
    return false;
  }

  out_tensor.name = tensor_name;
  out_tensor.dtype = info.dtype;
  out_tensor.shape = info.shape;
  out_tensor.data.resize(elements);

  if (info.dtype == "F32") {
    std::memcpy(out_tensor.data.data(), raw.data(), raw.size());
    return true;
  }

  const auto * raw_u16 = reinterpret_cast<const std::uint16_t *>(raw.data());
  for (std::size_t i = 0; i < elements; ++i) {
    if (info.dtype == "BF16") {
      out_tensor.data[i] = bf16_to_float(raw_u16[i]);
    } else if (info.dtype == "F16") {
      out_tensor.data[i] = f16_to_float(raw_u16[i]);
    }
  }

  return true;
}

} // namespace qwen35x
