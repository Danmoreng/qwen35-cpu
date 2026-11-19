#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>

namespace qwen35x::server {
// One checkpoint per server. Configure before starting HTTP or inference workers.
void load_chat_template(const std::filesystem::path &directory);
std::string render_chat_template(const nlohmann::ordered_json &messages,
    const nlohmann::ordered_json &tools, const nlohmann::ordered_json &kwargs, bool generation);
}
