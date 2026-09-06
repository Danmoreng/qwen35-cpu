#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen35x {

struct SafetensorTensorInfo {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::uint64_t data_start = 0;
  std::uint64_t data_end = 0;
};

struct SafetensorTensorF32 {
  std::string name;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<float> data;
};

class SafetensorLoader {
public:
  static bool resolve_tensor_file(
    const std::string & model_dir,
    const std::string & tensor_name,
    std::string & out_file,
    std::string & error_message);

  static bool load_tensor_info(
    const std::string & safetensors_file,
    const std::string & tensor_name,
    SafetensorTensorInfo & out_info,
    std::string & error_message);

  static bool read_bf16_tensor(
    const std::string & safetensors_file,
    const SafetensorTensorInfo & info,
    std::vector<std::uint16_t> & out_data,
    std::string & error_message);

  static bool read_tensor_f32(
    const std::string & model_dir,
    const std::string & tensor_name,
    SafetensorTensorF32 & out_tensor,
    std::string & error_message);
};

} // namespace qwen35x
