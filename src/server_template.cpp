#include "server_template.h"
#include "qwen35-template.h"
#include "jinja/parser.h"
#include <fstream>
#include <mutex>
#include <optional>

namespace qwen35x::server {
namespace {
using Json = nlohmann::ordered_json;
struct Template {
  std::string source;
  jinja::program program;
  Json tokens;
  Template(std::string text, Json special)
      : source(std::move(text)), program(jinja::parse_from_tokens(jinja::lexer().tokenize(source))),
        tokens(std::move(special)) {}
};
std::optional<Template> loaded, tool_template;
bool configured = false;
const Template &active_template(bool tools) {
  if (tools && tool_template) return *tool_template;
  if (configured && !loaded) throw std::runtime_error("The checkpoint has no default chat template");
  if (loaded) return *loaded;
  static const Template fallback(qwen35_default_template, Json::object());
  return fallback;
}
}
void load_chat_template(const std::filesystem::path &directory) {
  Json config = Json::object();
  if (std::ifstream in(directory / "tokenizer_config.json"); in) in >> config;
  Json special = Json::object();
  auto add_token = [&](const std::string &name, const Json &token) {
    if (token.is_string()) special[name] = token;
    else if (token.is_object() && token.contains("content")) special[name] = token.at("content");
  };
  Json legacy_tokens;
  if (std::ifstream in(directory / "special_tokens_map.json"); in) in >> legacy_tokens;
  for (const auto &[name, token] : legacy_tokens.items()) add_token(name, token);
  for (const auto &[name, token] : config.items()) {
    if (name.ends_with("_token")) add_token(name, token);
    if ((name == "extra_special_tokens" || name == "model_specific_special_tokens") && token.is_object())
      for (const auto &[key, value] : token.items()) add_token(key, value);
  }
  Json sources = Json::object();
  if (config.contains("chat_template") && !config["chat_template"].is_null()) {
    const auto &entry = config.at("chat_template");
    if (entry.is_string()) sources["default"] = entry;
    else if (entry.is_object()) sources = entry;
    else if (entry.is_array()) {
      for (const auto &item : entry) sources[item.at("name").get<std::string>()] = item.at("template");
    } else throw std::runtime_error("Invalid tokenizer_config.json chat_template");
  }
  Json files = Json::object();
  auto load_file = [&](const std::string &name, const std::filesystem::path &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not read chat template: " + path.string());
    files[name] = std::string(std::istreambuf_iterator<char>(file), {});
    if (file.bad()) throw std::runtime_error("Failed reading chat template: " + path.string());
  };
  const auto default_path = directory / "chat_template.jinja";
  if (std::filesystem::exists(default_path)) load_file("default", default_path);
  const auto named_dir = directory / "chat_templates";
  if (std::filesystem::is_directory(named_dir)) {
    for (const auto &entry : std::filesystem::directory_iterator(named_dir))
      if (entry.is_regular_file() && entry.path().extension() == ".jinja")
        load_file(entry.path().stem().string(), entry.path());
  }
  // Transformers selects the standalone file family when any template files exist.
  if (!files.empty()) sources = std::move(files);
  configured = true;
  auto compile = [&](const char *name, std::optional<Template> &target) {
    target.reset();
    if (!sources.contains(name)) return;
    const auto source = sources.at(name).get<std::string>();
    if (source.empty()) throw std::runtime_error(std::string("Empty model chat template: ") + name);
    target.emplace(source, special);
  };
  if (sources.empty()) sources["default"] = qwen35_default_template;
  compile("default", loaded);
  compile("tool_use", tool_template);
}
std::string render_chat_template(const Json &messages, const Json &tools,
                                 const Json &kwargs, bool generation) {
  const auto &tmpl = active_template(!tools.empty());
  auto vars = tmpl.tokens;
  vars.update(kwargs);
  vars["messages"] = messages;
  vars["add_generation_prompt"] = generation;
  vars["tools"] = tools.empty() ? Json(nullptr) : tools;
  if (!vars.contains("documents")) vars["documents"] = nullptr;
  jinja::context ctx(tmpl.source);
  jinja::global_from_json(ctx, vars, false);
  // Serialize evaluation of the shared upstream AST. Tokenization and decode
  // proceed outside this lock.
  static std::mutex render_mutex;
  std::lock_guard lock(render_mutex);
  auto result = jinja::runtime(ctx).execute(tmpl.program);
  return jinja::runtime::gather_string_parts(result)->as_string().str();
}
}
