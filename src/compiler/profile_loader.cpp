#include "qwen35x/compiler/compiler.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>

namespace qwen35x {

namespace {

std::optional<std::string> read_text(const std::string & path, std::string & error_message) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    error_message = "Could not open profile file: " + path;
    return std::nullopt;
  }

  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::optional<std::string> capture_string(const std::string & json, const std::string & key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (std::regex_search(json, match, pattern) && match.size() == 2) {
    return match[1].str();
  }
  return std::nullopt;
}

std::optional<int> capture_int(const std::string & json, const std::string & key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (std::regex_search(json, match, pattern) && match.size() == 2) {
    return std::stoi(match[1].str());
  }
  return std::nullopt;
}

std::optional<std::uint64_t> capture_uint64(const std::string & json, const std::string & key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch match;
  if (std::regex_search(json, match, pattern) && match.size() == 2) {
    return static_cast<std::uint64_t>(std::stoull(match[1].str()));
  }
  return std::nullopt;
}

std::optional<float> capture_float(const std::string & json, const std::string & key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::smatch match;
  if (std::regex_search(json, match, pattern) && match.size() == 2) {
    return std::stof(match[1].str());
  }
  return std::nullopt;
}

std::optional<bool> capture_bool(const std::string & json, const std::string & key) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (std::regex_search(json, match, pattern) && match.size() == 2) {
    return match[1].str() == "true";
  }
  return std::nullopt;
}

std::string to_lower_copy(const std::string & text) {
  std::string out = text;
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

AttentionBlock parse_block(const std::string & token) {
  const std::string lower = to_lower_copy(token);
  return lower.find("full") != std::string::npos ? AttentionBlock::full : AttentionBlock::linear;
}

std::vector<AttentionBlock> parse_pattern(const std::string & pattern, int repeats) {
  std::vector<AttentionBlock> base;
  std::stringstream ss(pattern);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      base.push_back(parse_block(item));
    }
  }

  if (base.empty()) {
    base = {AttentionBlock::linear, AttentionBlock::linear, AttentionBlock::linear, AttentionBlock::full};
  }

  std::vector<AttentionBlock> full;
  full.reserve(base.size() * std::max(1, repeats));
  for (int i = 0; i < std::max(1, repeats); ++i) {
    full.insert(full.end(), base.begin(), base.end());
  }
  return full;
}

std::optional<std::string> extract_enclosed(const std::string & text, std::size_t open_pos, char open_char, char close_char) {
  if (open_pos >= text.size() || text[open_pos] != open_char) {
    return std::nullopt;
  }

  std::size_t depth = 0;
  bool in_string = false;
  bool escaped = false;

  for (std::size_t i = open_pos; i < text.size(); ++i) {
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
    } else if (c == close_char) {
      if (depth == 0) {
        return std::nullopt;
      }
      --depth;
      if (depth == 0) {
        return text.substr(open_pos, i - open_pos + 1);
      }
    }
  }

  return std::nullopt;
}

std::optional<std::string> extract_object_for_key(const std::string & json, const std::string & key) {
  const std::string marker = "\"" + key + "\"";
  std::size_t search_pos = 0;

  while (true) {
    const std::size_t key_pos = json.find(marker, search_pos);
    if (key_pos == std::string::npos) {
      return std::nullopt;
    }

    const std::size_t colon_pos = json.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
      return std::nullopt;
    }

    const std::size_t open_pos = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (open_pos != std::string::npos && json[open_pos] == '{') {
      return extract_enclosed(json, open_pos, '{', '}');
    }

    search_pos = key_pos + marker.size();
  }
}

std::optional<std::string> extract_array_for_key(const std::string & json, const std::string & key) {
  const std::string marker = "\"" + key + "\"";
  std::size_t search_pos = 0;

  while (true) {
    const std::size_t key_pos = json.find(marker, search_pos);
    if (key_pos == std::string::npos) {
      return std::nullopt;
    }

    const std::size_t colon_pos = json.find(':', key_pos + marker.size());
    if (colon_pos == std::string::npos) {
      return std::nullopt;
    }

    const std::size_t open_pos = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (open_pos != std::string::npos && json[open_pos] == '[') {
      return extract_enclosed(json, open_pos, '[', ']');
    }

    search_pos = key_pos + marker.size();
  }
}

std::vector<std::string> parse_string_array(const std::string & json_array) {
  std::vector<std::string> out;
  const std::regex token("\"([^\"]+)\"");
  auto begin = std::sregex_iterator(json_array.begin(), json_array.end(), token);
  auto end = std::sregex_iterator();

  for (auto it = begin; it != end; ++it) {
    if (it->size() == 2) {
      out.push_back((*it)[1].str());
    }
  }
  return out;
}

std::string infer_variant(const std::string & model_id) {
  const std::regex variant_regex("([0-9]+(?:\\.[0-9]+)?b)", std::regex_constants::icase);
  std::smatch match;
  if (std::regex_search(model_id, match, variant_regex) && match.size() == 2) {
    return to_lower_copy(match[1].str());
  }
  return "unknown";
}

std::string normalize_family(const std::string & model_type) {
  const std::string lower = to_lower_copy(model_type);
  if (lower.find("qwen3_5") != std::string::npos || lower.find("qwen3.5") != std::string::npos) {
    return "qwen3.5";
  }
  return lower.empty() ? "unknown" : lower;
}

std::vector<std::string> parse_index_shards(const std::string & index_json) {
  std::set<std::string> unique;
  const std::regex shard_regex(":\\s*\"([^\"]+\\.safetensors)\"");

  auto begin = std::sregex_iterator(index_json.begin(), index_json.end(), shard_regex);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    if (it->size() == 2) {
      unique.insert((*it)[1].str());
    }
  }

  return {unique.begin(), unique.end()};
}

std::vector<std::string> find_local_safetensor_files(const std::filesystem::path & model_dir) {
  std::vector<std::string> shards;
  for (const auto & entry : std::filesystem::directory_iterator(model_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (filename.find(".safetensors") != std::string::npos) {
      shards.push_back(filename);
    }
  }
  std::sort(shards.begin(), shards.end());
  shards.erase(std::unique(shards.begin(), shards.end()), shards.end());
  return shards;
}

std::uint64_t sum_existing_shard_sizes(const std::filesystem::path & model_dir, const std::vector<std::string> & shards) {
  std::uint64_t total = 0;
  for (const auto & shard : shards) {
    const std::filesystem::path shard_path = model_dir / shard;
    if (std::filesystem::exists(shard_path) && std::filesystem::is_regular_file(shard_path)) {
      total += std::filesystem::file_size(shard_path);
    }
  }
  return total;
}

} // namespace

std::optional<ModelProfile> ProfileLoader::load_from_json(const std::string & path, std::string & error_message) {
  const auto json = read_text(path, error_message);
  if (!json) {
    return std::nullopt;
  }

  ModelProfile profile;
  profile.model_id = capture_string(*json, "model_id").value_or("Qwen/Qwen3.5-0.8B");
  profile.family = capture_string(*json, "family").value_or("qwen3.5");
  profile.variant = capture_string(*json, "variant").value_or("0.8b");

  profile.text.num_hidden_layers = capture_int(*json, "num_hidden_layers").value_or(24);
  profile.text.hidden_size = capture_int(*json, "hidden_size").value_or(1024);
  profile.text.intermediate_size = capture_int(*json, "intermediate_size").value_or(3584);
  profile.text.num_attention_heads = capture_int(*json, "num_attention_heads").value_or(8);
  profile.text.num_key_value_heads = capture_int(*json, "num_key_value_heads").value_or(2);
  profile.text.head_dim = capture_int(*json, "head_dim").value_or(256);
  profile.text.vocab_size = capture_int(*json, "vocab_size").value_or(248320);
  profile.text.max_position_embeddings = capture_int(*json, "max_position_embeddings").value_or(32768);
  profile.text.rms_norm_eps = capture_float(*json, "rms_norm_eps").value_or(1.0e-6f);
  profile.text.rope_theta = capture_float(*json, "rope_theta").value_or(10000000.0f);
  profile.text.partial_rotary_factor = capture_float(*json, "partial_rotary_factor").value_or(0.25f);
  profile.text.full_attention_interval = capture_int(*json, "full_attention_interval").value_or(4);
  profile.text.linear_conv_kernel_dim = capture_int(*json, "linear_conv_kernel_dim").value_or(4);
  profile.text.linear_num_key_heads = capture_int(*json, "linear_num_key_heads").value_or(16);
  profile.text.linear_num_value_heads = capture_int(*json, "linear_num_value_heads").value_or(16);
  profile.text.linear_key_head_dim = capture_int(*json, "linear_key_head_dim").value_or(128);
  profile.text.linear_value_head_dim = capture_int(*json, "linear_value_head_dim").value_or(128);
  profile.text.tie_word_embeddings = capture_bool(*json, "tie_word_embeddings").value_or(true);

  profile.fingerprint.num_hidden_layers = profile.text.num_hidden_layers;
  profile.fingerprint.hidden_size = profile.text.hidden_size;
  profile.fingerprint.num_attention_heads = profile.text.num_attention_heads;
  profile.fingerprint.num_key_value_heads = profile.text.num_key_value_heads;

  const auto pattern = capture_string(*json, "layer_pattern").value_or("linear,linear,linear,full");
  const int repeats = capture_int(*json, "pattern_repeats").value_or(6);
  profile.fingerprint.attention_schedule = parse_pattern(pattern, repeats);

  if (profile.fingerprint.attention_schedule.size() != static_cast<std::size_t>(profile.fingerprint.num_hidden_layers)) {
    error_message = "Invalid profile: attention schedule length must match num_hidden_layers.";
    return std::nullopt;
  }

  return profile;
}

std::optional<ModelProfile> ProfileLoader::load_from_hf_directory(const std::string & model_dir, std::string & error_message) {
  namespace fs = std::filesystem;

  const fs::path root(model_dir);
  if (!fs::exists(root) || !fs::is_directory(root)) {
    error_message = "HF model directory does not exist: " + model_dir;
    return std::nullopt;
  }

  const fs::path config_path = root / "config.json";
  const fs::path index_path = root / "model.safetensors.index.json";

  const auto config_json = read_text(config_path.string(), error_message);
  if (!config_json) {
    error_message = "Could not read HF config.json from: " + config_path.string();
    return std::nullopt;
  }

  const std::string text_config = extract_object_for_key(*config_json, "text_config").value_or(*config_json);
  const std::string root_model_type = capture_string(*config_json, "model_type").value_or("");
  const std::string text_model_type = capture_string(text_config, "model_type").value_or(root_model_type);

  ModelProfile profile;
  profile.model_id = capture_string(*config_json, "_name_or_path").value_or(root.filename().string());
  profile.family = normalize_family(text_model_type);
  profile.variant = infer_variant(profile.model_id + " " + root.filename().string());

  profile.text.num_hidden_layers =
    capture_int(text_config, "num_hidden_layers").value_or(capture_int(*config_json, "num_hidden_layers").value_or(0));
  profile.text.hidden_size =
    capture_int(text_config, "hidden_size").value_or(capture_int(*config_json, "hidden_size").value_or(0));
  profile.text.intermediate_size =
    capture_int(text_config, "intermediate_size").value_or(capture_int(*config_json, "intermediate_size").value_or(0));
  profile.text.num_attention_heads =
    capture_int(text_config, "num_attention_heads").value_or(capture_int(*config_json, "num_attention_heads").value_or(0));
  profile.text.num_key_value_heads = capture_int(text_config, "num_key_value_heads")
                                              .value_or(capture_int(*config_json, "num_key_value_heads").value_or(0));
  profile.text.head_dim = capture_int(text_config, "head_dim").value_or(0);
  profile.text.vocab_size = capture_int(text_config, "vocab_size").value_or(capture_int(*config_json, "vocab_size").value_or(0));
  profile.text.max_position_embeddings =
    capture_int(text_config, "max_position_embeddings").value_or(capture_int(*config_json, "max_position_embeddings").value_or(0));
  profile.text.rms_norm_eps = capture_float(text_config, "rms_norm_eps").value_or(1.0e-6f);
  profile.text.tie_word_embeddings = capture_bool(text_config, "tie_word_embeddings")
                                       .value_or(capture_bool(*config_json, "tie_word_embeddings").value_or(true));
  profile.text.full_attention_interval = capture_int(text_config, "full_attention_interval").value_or(0);
  profile.text.linear_conv_kernel_dim = capture_int(text_config, "linear_conv_kernel_dim").value_or(0);
  profile.text.linear_num_key_heads = capture_int(text_config, "linear_num_key_heads").value_or(0);
  profile.text.linear_num_value_heads = capture_int(text_config, "linear_num_value_heads").value_or(0);
  profile.text.linear_key_head_dim = capture_int(text_config, "linear_key_head_dim").value_or(0);
  profile.text.linear_value_head_dim = capture_int(text_config, "linear_value_head_dim").value_or(0);

  const auto rope_parameters = extract_object_for_key(text_config, "rope_parameters");
  if (rope_parameters) {
    profile.text.rope_theta = capture_float(*rope_parameters, "rope_theta").value_or(10000000.0f);
    profile.text.partial_rotary_factor = capture_float(*rope_parameters, "partial_rotary_factor").value_or(0.25f);
  } else {
    profile.text.rope_theta = 10000000.0f;
    profile.text.partial_rotary_factor = 0.25f;
  }

  if (profile.text.head_dim <= 0 && profile.text.num_attention_heads > 0) {
    profile.text.head_dim = profile.text.hidden_size / profile.text.num_attention_heads;
  }

  profile.fingerprint.num_hidden_layers = profile.text.num_hidden_layers;
  profile.fingerprint.hidden_size = profile.text.hidden_size;
  profile.fingerprint.num_attention_heads = profile.text.num_attention_heads;
  profile.fingerprint.num_key_value_heads = profile.text.num_key_value_heads;

  if (profile.text.num_hidden_layers <= 0 || profile.text.hidden_size <= 0 ||
      profile.text.num_attention_heads <= 0 || profile.text.num_key_value_heads <= 0) {
    error_message = "HF config is missing required text model dimensions.";
    return std::nullopt;
  }

  if (const auto layer_types_json = extract_array_for_key(text_config, "layer_types")) {
    const auto layer_types = parse_string_array(*layer_types_json);
    for (const auto & type : layer_types) {
      profile.fingerprint.attention_schedule.push_back(parse_block(type));
    }
  }

  if (profile.fingerprint.attention_schedule.empty()) {
    const int interval = profile.text.full_attention_interval > 0 ? profile.text.full_attention_interval : 4;
    profile.fingerprint.attention_schedule.reserve(static_cast<std::size_t>(profile.text.num_hidden_layers));
    for (int i = 0; i < profile.text.num_hidden_layers; ++i) {
      const bool is_full = interval > 0 && ((i + 1) % interval == 0);
      profile.fingerprint.attention_schedule.push_back(is_full ? AttentionBlock::full : AttentionBlock::linear);
    }
  }

  if (profile.fingerprint.attention_schedule.size() != static_cast<std::size_t>(profile.text.num_hidden_layers)) {
    error_message = "HF config layer_types length does not match num_hidden_layers.";
    return std::nullopt;
  }

  if (fs::exists(index_path) && fs::is_regular_file(index_path)) {
    const auto index_json = read_text(index_path.string(), error_message);
    if (!index_json) {
      error_message = "Could not read HF safetensors index from: " + index_path.string();
      return std::nullopt;
    }
    profile.weights.total_size_bytes = capture_uint64(*index_json, "total_size").value_or(0);
    profile.weights.shard_files = parse_index_shards(*index_json);
  } else {
    profile.weights.shard_files = find_local_safetensor_files(root);
  }

  // Inference needs the config and the packed artifact, not BF16 source shards.
  // The converter validates source shard availability when opening safetensors.

  const std::uint64_t local_bytes = sum_existing_shard_sizes(root, profile.weights.shard_files);
  if (profile.weights.total_size_bytes == 0) {
    profile.weights.total_size_bytes = local_bytes;
  }

  return profile;
}

std::string to_string(AttentionBlock block) {
  switch (block) {
    case AttentionBlock::linear:
      return "linear";
    case AttentionBlock::full:
      return "full";
  }
  return "unknown";
}

std::string fingerprint_summary(const ModelFingerprint & fingerprint) {
  std::ostringstream out;
  out << "layers=" << fingerprint.num_hidden_layers
      << ", hidden=" << fingerprint.hidden_size
      << ", q_heads=" << fingerprint.num_attention_heads
      << ", kv_heads=" << fingerprint.num_key_value_heads
      << ", schedule=";

  const std::size_t preview = std::min<std::size_t>(fingerprint.attention_schedule.size(), 8);
  for (std::size_t i = 0; i < preview; ++i) {
    if (i > 0) {
      out << ",";
    }
    out << to_string(fingerprint.attention_schedule[i]);
  }
  if (fingerprint.attention_schedule.size() > preview) {
    out << "...";
  }
  return out.str();
}

} // namespace qwen35x
