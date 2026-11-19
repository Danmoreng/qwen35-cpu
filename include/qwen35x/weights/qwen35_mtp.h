#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace qwen35x {

// The single next-token prediction block from the official Qwen3.5-4B.
struct Qwen35MtpTensor {
  std::string_view name;
  std::uint64_t rows, columns = 0;
};

inline constexpr std::array qwen35_4b_mtp_tensors{
  Qwen35MtpTensor{"mtp.fc.weight", 2560, 5120},
  Qwen35MtpTensor{"mtp.pre_fc_norm_embedding.weight", 2560},
  Qwen35MtpTensor{"mtp.pre_fc_norm_hidden.weight", 2560},
  Qwen35MtpTensor{"mtp.layers.0.input_layernorm.weight", 2560},
  Qwen35MtpTensor{"mtp.layers.0.self_attn.q_proj.weight", 8192, 2560},
  Qwen35MtpTensor{"mtp.layers.0.self_attn.k_proj.weight", 1024, 2560},
  Qwen35MtpTensor{"mtp.layers.0.self_attn.v_proj.weight", 1024, 2560},
  Qwen35MtpTensor{"mtp.layers.0.self_attn.q_norm.weight", 256},
  Qwen35MtpTensor{"mtp.layers.0.self_attn.k_norm.weight", 256},
  Qwen35MtpTensor{"mtp.layers.0.self_attn.o_proj.weight", 2560, 4096},
  Qwen35MtpTensor{"mtp.layers.0.post_attention_layernorm.weight", 2560},
  Qwen35MtpTensor{"mtp.layers.0.mlp.gate_proj.weight", 9216, 2560},
  Qwen35MtpTensor{"mtp.layers.0.mlp.up_proj.weight", 9216, 2560},
  Qwen35MtpTensor{"mtp.layers.0.mlp.down_proj.weight", 2560, 9216},
  Qwen35MtpTensor{"mtp.norm.weight", 2560},
};

} // namespace qwen35x
