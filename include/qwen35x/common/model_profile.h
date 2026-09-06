#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace qwen35x {

enum class AttentionBlock {
  linear,
  full,
};

struct TextModelConfig {
  int num_hidden_layers = 0;
  int hidden_size = 0;
  int intermediate_size = 0;

  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;

  int linear_conv_kernel_dim = 0;
  int linear_num_key_heads = 0;
  int linear_num_value_heads = 0;
  int linear_key_head_dim = 0;
  int linear_value_head_dim = 0;

  int vocab_size = 0;
  int max_position_embeddings = 0;
  float rms_norm_eps = 1.0e-6f;
  float rope_theta = 10000000.0f;
  float partial_rotary_factor = 0.25f;
  int full_attention_interval = 0;
  bool tie_word_embeddings = true;
};

struct ModelFingerprint {
  int num_hidden_layers = 0;
  int hidden_size = 0;
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  std::vector<AttentionBlock> attention_schedule;
};

struct ModelWeightsManifest {
  std::uint64_t total_size_bytes = 0;
  std::vector<std::string> shard_files;
};

struct ModelProfile {
  std::string model_id;
  std::string family;
  std::string variant;
  TextModelConfig text;
  ModelFingerprint fingerprint;
  ModelWeightsManifest weights;
};

std::string to_string(AttentionBlock block);
std::string fingerprint_summary(const ModelFingerprint & fingerprint);

} // namespace qwen35x
