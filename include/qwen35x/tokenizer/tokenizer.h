#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen35x {

class QwenTokenizer {
public:
  static bool load_from_hf_directory(
    const std::string & model_dir,
    QwenTokenizer & out_tokenizer,
    std::string & error_message);

  bool encode(const std::string & text, std::vector<std::int32_t> & out_tokens, std::string & error_message) const;
  bool decode(const std::vector<std::int32_t> & token_ids, std::string & out_text, std::string & error_message) const;

  std::optional<std::int32_t> token_to_id(const std::string & token) const;
  std::optional<std::string> id_to_token(std::int32_t token_id) const;
  std::size_t vocab_size() const;

private:
  struct StringPairHash {
    std::size_t operator()(const std::pair<std::string, std::string> & value) const noexcept;
  };

  struct Utf8Span {
    std::uint32_t codepoint = 0;
    std::size_t begin = 0;
    std::size_t end = 0;
  };

  bool encode_text_segment(
    const std::string & text,
    std::vector<std::int32_t> & out_tokens,
    std::string & error_message) const;
  bool apply_bpe(const std::string & transformed_piece, std::vector<std::string> & out_tokens) const;
  void split_text_for_bpe(const std::string & text, std::vector<std::string> & out_pieces) const;
  std::vector<Utf8Span> decode_utf8_with_spans(const std::string & text) const;

  static bool read_text_file(const std::string & path, std::string & out_text, std::string & error_message);
  static bool parse_vocab_json(
    const std::string & json_text,
    std::unordered_map<std::string, std::int32_t> & out_vocab,
    std::vector<std::string> & out_id_to_token,
    std::string & error_message);
  static bool parse_merges_txt(
    const std::string & text,
    std::unordered_map<std::pair<std::string, std::string>, int, StringPairHash> & out_ranks,
    std::string & error_message);
  static bool parse_added_tokens_decoder_from_config(
    const std::string & config_json,
    std::unordered_map<std::string, std::int32_t> & out_token_to_id,
    std::unordered_map<std::int32_t, std::string> & out_id_to_token,
    std::string & error_message);
  static std::string encode_utf8_codepoint(std::uint32_t codepoint);
  static bool decode_json_string(
    const std::string & text,
    std::size_t & cursor,
    std::string & out,
    std::string & error_message);
  static bool skip_json_value(const std::string & text, std::size_t & cursor, std::string & error_message);
  static void build_byte_level_maps(
    std::vector<std::string> & out_byte_to_unicode,
    std::unordered_map<std::uint32_t, std::uint8_t> & out_unicode_to_byte);
  static bool looks_like_added_token(const std::string & token);

  std::unordered_map<std::string, std::int32_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::pair<std::string, std::string>, int, StringPairHash> merge_ranks_;
  mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;

  std::unordered_map<std::string, std::int32_t> added_tokens_;
  std::unordered_map<std::int32_t, std::string> added_token_id_to_content_;
  std::unordered_map<unsigned char, std::vector<std::string>> added_tokens_by_first_byte_;
  std::int32_t max_token_id_ = -1;

  std::vector<std::string> byte_to_unicode_;
  std::unordered_map<std::uint32_t, std::uint8_t> unicode_to_byte_;
};

} // namespace qwen35x
