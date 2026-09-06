#pragma once
#include "qwen35x/common/model_profile.h"
#include <optional>
#include <string>
namespace qwen35x {
class ProfileLoader {
public:
  static std::optional<ModelProfile> load_from_json(const std::string&,std::string&);
  static std::optional<ModelProfile> load_from_hf_directory(const std::string&,std::string&);
};
}
