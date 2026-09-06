#include "qwen35x/tokenizer/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace qwen35x {

namespace {

void skip_ws(const std::string & text, std::size_t & cursor) {
  while (cursor < text.size() && std::isspace(static_cast<unsigned char>(text[cursor])) != 0) {
    ++cursor;
  }
}

bool is_newline(const std::uint32_t cp) {
  return cp == static_cast<std::uint32_t>('\n') || cp == static_cast<std::uint32_t>('\r');
}

bool is_space_no_newline(const std::uint32_t cp) {
  if (is_newline(cp)) {
    return false;
  }
  if (cp < 128) {
    return std::isspace(static_cast<unsigned char>(cp)) != 0;
  }
  return cp == 0x00A0u || cp == 0x1680u || cp == 0x2000u || cp == 0x2001u || cp == 0x2002u || cp == 0x2003u ||
         cp == 0x2004u || cp == 0x2005u || cp == 0x2006u || cp == 0x2007u || cp == 0x2008u || cp == 0x2009u ||
         cp == 0x200Au || cp == 0x202Fu || cp == 0x205Fu || cp == 0x3000u;
}

bool is_space_any(const std::uint32_t cp) {
  return is_newline(cp) || is_space_no_newline(cp);
}

bool is_mark(const std::uint32_t cp) {
  return (cp >= 0x0300u && cp <= 0x036Fu) || (cp >= 0x1AB0u && cp <= 0x1AFFu) || (cp >= 0x1DC0u && cp <= 0x1DFFu) ||
         (cp >= 0x20D0u && cp <= 0x20FFu) || (cp >= 0xFE20u && cp <= 0xFE2Fu);
}

bool is_digit(const std::uint32_t cp) {
  return cp >= static_cast<std::uint32_t>('0') && cp <= static_cast<std::uint32_t>('9');
}

bool is_letter(const std::uint32_t cp) {
  if (cp < 128) {
    return std::isalpha(static_cast<unsigned char>(cp)) != 0;
  }
  return !is_mark(cp) && !is_digit(cp) && !is_space_any(cp);
}

bool is_letter_or_mark(const std::uint32_t cp) {
  return is_letter(cp) || is_mark(cp);
}

std::uint32_t parse_hex4(const std::string & text, const std::size_t cursor) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    const char c = text[cursor + i];
    value <<= 4;
    if (c >= '0' && c <= '9') {
      value |= static_cast<std::uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      value |= static_cast<std::uint32_t>(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      value |= static_cast<std::uint32_t>(c - 'A' + 10);
    }
  }
  return value;
}

} // namespace

std::size_t QwenTokenizer::StringPairHash::operator()(const std::pair<std::string, std::string> & value) const noexcept {
  const std::size_t h1 = std::hash<std::string>{}(value.first);
  const std::size_t h2 = std::hash<std::string>{}(value.second);
  return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6U) + (h1 >> 2U));
}

bool QwenTokenizer::read_text_file(const std::string & path, std::string & out_text, std::string & error_message) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    error_message = "Could not open file: " + path;
    return false;
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  out_text = buffer.str();
  return true;
}

std::string QwenTokenizer::encode_utf8_codepoint(const std::uint32_t codepoint) {
  std::string out;
  if (codepoint <= 0x7Fu) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FFu) {
    out.push_back(static_cast<char>(0xC0u | (codepoint >> 6)));
    out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
  } else if (codepoint <= 0xFFFFu) {
    out.push_back(static_cast<char>(0xE0u | (codepoint >> 12)));
    out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
    out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
  } else {
    out.push_back(static_cast<char>(0xF0u | (codepoint >> 18)));
    out.push_back(static_cast<char>(0x80u | ((codepoint >> 12) & 0x3Fu)));
    out.push_back(static_cast<char>(0x80u | ((codepoint >> 6) & 0x3Fu)));
    out.push_back(static_cast<char>(0x80u | (codepoint & 0x3Fu)));
  }
  return out;
}

bool QwenTokenizer::decode_json_string(
  const std::string & text,
  std::size_t & cursor,
  std::string & out,
  std::string & error_message) {
  if (cursor >= text.size() || text[cursor] != '"') {
    error_message = "Invalid JSON string: missing opening quote.";
    return false;
  }
  ++cursor;
  out.clear();

  while (cursor < text.size()) {
    const char c = text[cursor++];
    if (c == '"') {
      return true;
    }
    if (c != '\\') {
      out.push_back(c);
      continue;
    }
    if (cursor >= text.size()) {
      error_message = "Invalid JSON escape sequence.";
      return false;
    }

    const char esc = text[cursor++];
    switch (esc) {
      case '"':
      case '\\':
      case '/':
        out.push_back(esc);
        break;
      case 'b':
        out.push_back('\b');
        break;
      case 'f':
        out.push_back('\f');
        break;
      case 'n':
        out.push_back('\n');
        break;
      case 'r':
        out.push_back('\r');
        break;
      case 't':
        out.push_back('\t');
        break;
      case 'u': {
        if (cursor + 4 > text.size()) {
          error_message = "Invalid JSON unicode escape sequence.";
          return false;
        }
        std::uint32_t cp = parse_hex4(text, cursor);
        cursor += 4;

        if (cp >= 0xD800u && cp <= 0xDBFFu) {
          if (cursor + 6 <= text.size() && text[cursor] == '\\' && text[cursor + 1] == 'u') {
            cursor += 2;
            const std::uint32_t low = parse_hex4(text, cursor);
            cursor += 4;
            if (low >= 0xDC00u && low <= 0xDFFFu) {
              cp = 0x10000u + ((cp - 0xD800u) << 10) + (low - 0xDC00u);
            } else {
              error_message = "Invalid JSON unicode surrogate pair.";
              return false;
            }
          } else {
            error_message = "Invalid JSON unicode surrogate pair.";
            return false;
          }
        }

        out += encode_utf8_codepoint(cp);
        break;
      }
      default:
        error_message = "Invalid JSON escape sequence.";
        return false;
    }
  }

  error_message = "Invalid JSON string: missing closing quote.";
  return false;
}

bool QwenTokenizer::skip_json_value(const std::string & text, std::size_t & cursor, std::string & error_message) {
  skip_ws(text, cursor);
  if (cursor >= text.size()) {
    error_message = "Invalid JSON: unexpected EOF.";
    return false;
  }

  if (text[cursor] == '"') {
    std::string ignored;
    return decode_json_string(text, cursor, ignored, error_message);
  }

  if (text[cursor] == '{') {
    ++cursor;
    while (true) {
      skip_ws(text, cursor);
      if (cursor >= text.size()) {
        error_message = "Invalid JSON object: unexpected EOF.";
        return false;
      }
      if (text[cursor] == '}') {
        ++cursor;
        return true;
      }

      std::string key;
      if (!decode_json_string(text, cursor, key, error_message)) {
        return false;
      }

      skip_ws(text, cursor);
      if (cursor >= text.size() || text[cursor] != ':') {
        error_message = "Invalid JSON object: expected ':'.";
        return false;
      }
      ++cursor;

      if (!skip_json_value(text, cursor, error_message)) {
        return false;
      }

      skip_ws(text, cursor);
      if (cursor < text.size() && text[cursor] == ',') {
        ++cursor;
        continue;
      }
      if (cursor < text.size() && text[cursor] == '}') {
        ++cursor;
        return true;
      }
      error_message = "Invalid JSON object: expected ',' or '}'.";
      return false;
    }
  }

  if (text[cursor] == '[') {
    ++cursor;
    while (true) {
      skip_ws(text, cursor);
      if (cursor >= text.size()) {
        error_message = "Invalid JSON array: unexpected EOF.";
        return false;
      }
      if (text[cursor] == ']') {
        ++cursor;
        return true;
      }

      if (!skip_json_value(text, cursor, error_message)) {
        return false;
      }

      skip_ws(text, cursor);
      if (cursor < text.size() && text[cursor] == ',') {
        ++cursor;
        continue;
      }
      if (cursor < text.size() && text[cursor] == ']') {
        ++cursor;
        return true;
      }
      error_message = "Invalid JSON array: expected ',' or ']'.";
      return false;
    }
  }

  if (text.compare(cursor, 4, "true") == 0) {
    cursor += 4;
    return true;
  }
  if (text.compare(cursor, 5, "false") == 0) {
    cursor += 5;
    return true;
  }
  if (text.compare(cursor, 4, "null") == 0) {
    cursor += 4;
    return true;
  }

  std::size_t start = cursor;
  if (text[cursor] == '-') {
    ++cursor;
  }
  bool have_digits = false;
  while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor])) != 0) {
    have_digits = true;
    ++cursor;
  }
  if (cursor < text.size() && text[cursor] == '.') {
    ++cursor;
    while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor])) != 0) {
      have_digits = true;
      ++cursor;
    }
  }
  if (cursor < text.size() && (text[cursor] == 'e' || text[cursor] == 'E')) {
    ++cursor;
    if (cursor < text.size() && (text[cursor] == '+' || text[cursor] == '-')) {
      ++cursor;
    }
    bool have_exp_digits = false;
    while (cursor < text.size() && std::isdigit(static_cast<unsigned char>(text[cursor])) != 0) {
      have_exp_digits = true;
      ++cursor;
    }
    if (!have_exp_digits) {
      error_message = "Invalid JSON number exponent.";
      return false;
    }
  }
  if (!have_digits || cursor == start) {
    error_message = "Invalid JSON value.";
    return false;
  }
  return true;
}

bool QwenTokenizer::parse_vocab_json(
  const std::string & json_text,
  std::unordered_map<std::string, std::int32_t> & out_vocab,
  std::vector<std::string> & out_id_to_token,
  std::string & error_message) {
  std::size_t cursor = 0;
  skip_ws(json_text, cursor);
  if (cursor >= json_text.size() || json_text[cursor] != '{') {
    error_message = "Invalid vocab.json: expected object.";
    return false;
  }
  ++cursor;
  out_vocab.clear();
  out_id_to_token.clear();

  while (true) {
    skip_ws(json_text, cursor);
    if (cursor >= json_text.size()) {
      error_message = "Invalid vocab.json: unexpected EOF.";
      return false;
    }
    if (json_text[cursor] == '}') {
      ++cursor;
      break;
    }

    std::string token;
    if (!decode_json_string(json_text, cursor, token, error_message)) {
      error_message = "Invalid vocab.json key: " + error_message;
      return false;
    }

    skip_ws(json_text, cursor);
    if (cursor >= json_text.size() || json_text[cursor] != ':') {
      error_message = "Invalid vocab.json: expected ':' after key.";
      return false;
    }
    ++cursor;

    skip_ws(json_text, cursor);
    if (cursor >= json_text.size()) {
      error_message = "Invalid vocab.json: missing value.";
      return false;
    }

    bool negative = false;
    if (json_text[cursor] == '-') {
      negative = true;
      ++cursor;
    }
    if (cursor >= json_text.size() || !std::isdigit(static_cast<unsigned char>(json_text[cursor]))) {
      error_message = "Invalid vocab.json: expected integer token id.";
      return false;
    }

    long long id = 0;
    while (cursor < json_text.size() && std::isdigit(static_cast<unsigned char>(json_text[cursor])) != 0) {
      const int digit = json_text[cursor] - '0';
      if (id > (std::numeric_limits<long long>::max() - digit) / 10) {
        error_message = "Invalid vocab.json: token id overflow.";
        return false;
      }
      id = id * 10 + digit;
      ++cursor;
    }
    if (negative) {
      id = -id;
    }
    if (id < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
        id > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
      error_message = "Invalid vocab.json: token id out of int32 range.";
      return false;
    }

    const auto id_i32 = static_cast<std::int32_t>(id);
    out_vocab[token] = id_i32;
    if (id >= 0) {
      const auto idx = static_cast<std::size_t>(id);
      if (idx >= out_id_to_token.size()) {
        out_id_to_token.resize(idx + 1);
      }
      out_id_to_token[idx] = token;
    }

    skip_ws(json_text, cursor);
    if (cursor < json_text.size() && json_text[cursor] == ',') {
      ++cursor;
      continue;
    }
    if (cursor < json_text.size() && json_text[cursor] == '}') {
      ++cursor;
      break;
    }
    if (cursor >= json_text.size()) {
      error_message = "Invalid vocab.json: unexpected EOF.";
      return false;
    }
    error_message = "Invalid vocab.json: expected ',' or '}'.";
    return false;
  }

  return true;
}

bool QwenTokenizer::parse_merges_txt(
  const std::string & text,
  std::unordered_map<std::pair<std::string, std::string>, int, StringPairHash> & out_ranks,
  std::string & error_message) {
  out_ranks.clear();

  std::istringstream stream(text);
  std::string line;
  int rank = 0;
  while (std::getline(stream, line)) {
    const auto begin = line.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
      continue;
    }
    const auto end = line.find_last_not_of(" \t\r\n");
    const std::string trimmed = line.substr(begin, end - begin + 1);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    std::istringstream pair_stream(trimmed);
    std::string left;
    std::string right;
    if (!(pair_stream >> left >> right)) {
      error_message = "Invalid merges.txt entry: " + trimmed;
      return false;
    }
    out_ranks[{left, right}] = rank++;
  }

  return true;
}

bool QwenTokenizer::parse_added_tokens_decoder_from_config(
  const std::string & config_json,
  std::unordered_map<std::string, std::int32_t> & out_token_to_id,
  std::unordered_map<std::int32_t, std::string> & out_id_to_token,
  std::string & error_message) {
  out_token_to_id.clear();
  out_id_to_token.clear();

  const std::string marker = "\"added_tokens_decoder\"";
  const std::size_t marker_pos = config_json.find(marker);
  if (marker_pos == std::string::npos) {
    return true;
  }

  std::size_t cursor = marker_pos + marker.size();
  skip_ws(config_json, cursor);
  if (cursor >= config_json.size() || config_json[cursor] != ':') {
    error_message = "Invalid tokenizer_config.json: expected ':' after added_tokens_decoder.";
    return false;
  }
  ++cursor;
  skip_ws(config_json, cursor);
  if (cursor >= config_json.size() || config_json[cursor] != '{') {
    error_message = "Invalid tokenizer_config.json: added_tokens_decoder must be an object.";
    return false;
  }
  ++cursor;

  while (true) {
    skip_ws(config_json, cursor);
    if (cursor >= config_json.size()) {
      error_message = "Invalid tokenizer_config.json: unexpected EOF in added_tokens_decoder.";
      return false;
    }
    if (config_json[cursor] == '}') {
      ++cursor;
      break;
    }

    std::string id_key;
    if (!decode_json_string(config_json, cursor, id_key, error_message)) {
      return false;
    }

    skip_ws(config_json, cursor);
    if (cursor >= config_json.size() || config_json[cursor] != ':') {
      error_message = "Invalid tokenizer_config.json: expected ':' after added token id key.";
      return false;
    }
    ++cursor;
    skip_ws(config_json, cursor);
    if (cursor >= config_json.size() || config_json[cursor] != '{') {
      error_message = "Invalid tokenizer_config.json: added_tokens_decoder entry must be an object.";
      return false;
    }
    ++cursor;

    std::string content;
    bool has_content = false;
    while (true) {
      skip_ws(config_json, cursor);
      if (cursor >= config_json.size()) {
        error_message = "Invalid tokenizer_config.json: unexpected EOF in added token entry.";
        return false;
      }
      if (config_json[cursor] == '}') {
        ++cursor;
        break;
      }

      std::string field_name;
      if (!decode_json_string(config_json, cursor, field_name, error_message)) {
        return false;
      }

      skip_ws(config_json, cursor);
      if (cursor >= config_json.size() || config_json[cursor] != ':') {
        error_message = "Invalid tokenizer_config.json: expected ':' after added token field key.";
        return false;
      }
      ++cursor;
      skip_ws(config_json, cursor);

      if (field_name == "content" && cursor < config_json.size() && config_json[cursor] == '"') {
        if (!decode_json_string(config_json, cursor, content, error_message)) {
          return false;
        }
        has_content = true;
      } else {
        if (!skip_json_value(config_json, cursor, error_message)) {
          return false;
        }
      }

      skip_ws(config_json, cursor);
      if (cursor < config_json.size() && config_json[cursor] == ',') {
        ++cursor;
        continue;
      }
      if (cursor < config_json.size() && config_json[cursor] == '}') {
        ++cursor;
        break;
      }
      error_message = "Invalid tokenizer_config.json: expected ',' or '}' in added token entry.";
      return false;
    }

    if (has_content) {
      try {
        const long long parsed = std::stoll(id_key);
        if (parsed < static_cast<long long>(std::numeric_limits<std::int32_t>::min()) ||
            parsed > static_cast<long long>(std::numeric_limits<std::int32_t>::max())) {
          error_message = "Invalid tokenizer_config.json added token id out of int32 range: " + id_key;
          return false;
        }
        const std::int32_t token_id = static_cast<std::int32_t>(parsed);
        out_token_to_id[content] = token_id;
        out_id_to_token[token_id] = content;
      } catch (...) {
        error_message = "Invalid tokenizer_config.json added token id: " + id_key;
        return false;
      }
    }

    skip_ws(config_json, cursor);
    if (cursor < config_json.size() && config_json[cursor] == ',') {
      ++cursor;
      continue;
    }
    if (cursor < config_json.size() && config_json[cursor] == '}') {
      ++cursor;
      break;
    }
    error_message = "Invalid tokenizer_config.json: expected ',' or '}' in added_tokens_decoder.";
    return false;
  }

  return true;
}

void QwenTokenizer::build_byte_level_maps(
  std::vector<std::string> & out_byte_to_unicode,
  std::unordered_map<std::uint32_t, std::uint8_t> & out_unicode_to_byte) {
  out_byte_to_unicode.assign(256, std::string{});
  out_unicode_to_byte.clear();

  std::vector<std::uint32_t> bs;
  bs.reserve(256);
  for (std::uint32_t i = 33; i <= 126; ++i) {
    bs.push_back(i);
  }
  for (std::uint32_t i = 161; i <= 172; ++i) {
    bs.push_back(i);
  }
  for (std::uint32_t i = 174; i <= 255; ++i) {
    bs.push_back(i);
  }

  std::vector<std::uint32_t> cs = bs;
  std::vector<bool> used(256, false);
  for (const auto b : bs) {
    used[b] = true;
  }

  std::uint32_t extra = 0;
  for (std::uint32_t b = 0; b <= 255; ++b) {
    if (!used[b]) {
      bs.push_back(b);
      cs.push_back(256 + extra);
      ++extra;
    }
  }

  for (std::size_t i = 0; i < bs.size(); ++i) {
    const auto byte_value = static_cast<std::uint8_t>(bs[i]);
    const auto codepoint = cs[i];
    out_byte_to_unicode[byte_value] = encode_utf8_codepoint(codepoint);
    out_unicode_to_byte[codepoint] = byte_value;
  }
}

bool QwenTokenizer::looks_like_added_token(const std::string & token) {
  if (token.size() < 3 || token.front() != '<' || token.back() != '>') {
    return false;
  }
  for (const unsigned char c : token) {
    if (std::isspace(c) != 0) {
      return false;
    }
  }
  return true;
}

std::vector<QwenTokenizer::Utf8Span> QwenTokenizer::decode_utf8_with_spans(const std::string & text) const {
  std::vector<Utf8Span> out;
  std::size_t i = 0;
  while (i < text.size()) {
    const std::size_t begin = i;
    const unsigned char c0 = static_cast<unsigned char>(text[i]);

    std::uint32_t cp = 0xFFFDu;
    std::size_t len = 1;
    if ((c0 & 0x80u) == 0) {
      cp = c0;
      len = 1;
    } else if ((c0 & 0xE0u) == 0xC0u && i + 1 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
      if ((c1 & 0xC0u) == 0x80u) {
        cp = (static_cast<std::uint32_t>(c0 & 0x1Fu) << 6) | static_cast<std::uint32_t>(c1 & 0x3Fu);
        len = 2;
      }
    } else if ((c0 & 0xF0u) == 0xE0u && i + 2 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
      if ((c1 & 0xC0u) == 0x80u && (c2 & 0xC0u) == 0x80u) {
        cp = (static_cast<std::uint32_t>(c0 & 0x0Fu) << 12) | (static_cast<std::uint32_t>(c1 & 0x3Fu) << 6) |
             static_cast<std::uint32_t>(c2 & 0x3Fu);
        len = 3;
      }
    } else if ((c0 & 0xF8u) == 0xF0u && i + 3 < text.size()) {
      const unsigned char c1 = static_cast<unsigned char>(text[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(text[i + 2]);
      const unsigned char c3 = static_cast<unsigned char>(text[i + 3]);
      if ((c1 & 0xC0u) == 0x80u && (c2 & 0xC0u) == 0x80u && (c3 & 0xC0u) == 0x80u) {
        cp = (static_cast<std::uint32_t>(c0 & 0x07u) << 18) | (static_cast<std::uint32_t>(c1 & 0x3Fu) << 12) |
             (static_cast<std::uint32_t>(c2 & 0x3Fu) << 6) | static_cast<std::uint32_t>(c3 & 0x3Fu);
        len = 4;
      }
    }

    i += len;
    out.push_back(Utf8Span{cp, begin, i});
  }
  return out;
}

void QwenTokenizer::split_text_for_bpe(const std::string & text, std::vector<std::string> & out_pieces) const {
  out_pieces.clear();
  const auto spans = decode_utf8_with_spans(text);
  if (spans.empty()) {
    return;
  }

  const auto emit_piece = [&](const std::size_t begin_index, const std::size_t end_index_exclusive) {
    if (begin_index >= end_index_exclusive || end_index_exclusive > spans.size()) {
      return;
    }
    const std::size_t begin = spans[begin_index].begin;
    const std::size_t end = spans[end_index_exclusive - 1].end;
    out_pieces.push_back(text.substr(begin, end - begin));
  };

  std::size_t i = 0;
  while (i < spans.size()) {
    if (spans[i].codepoint == static_cast<std::uint32_t>('\'') && i + 1 < spans.size()) {
      static constexpr const char * contractions[] = {"re", "ve", "ll", "s", "t", "m", "d"};
      bool matched = false;
      for (const char * suffix : contractions) {
        const std::size_t len = std::char_traits<char>::length(suffix);
        if (i + 1 + len > spans.size()) {
          continue;
        }
        bool ok = true;
        for (std::size_t j = 0; j < len; ++j) {
          const auto cp = spans[i + 1 + j].codepoint;
          if (cp > 0x7Fu) {
            ok = false;
            break;
          }
          const char lc = static_cast<char>(std::tolower(static_cast<unsigned char>(cp)));
          if (lc != suffix[j]) {
            ok = false;
            break;
          }
        }
        if (ok) {
          emit_piece(i, i + 1 + len);
          i += 1 + len;
          matched = true;
          break;
        }
      }
      if (matched) {
        continue;
      }
    }

    std::size_t j = i;
    if (!is_newline(spans[i].codepoint) && !is_letter(spans[i].codepoint) && !is_digit(spans[i].codepoint)) {
      j = i + 1;
    }
    std::size_t k = j;
    while (k < spans.size() && is_letter_or_mark(spans[k].codepoint)) {
      ++k;
    }
    if (k > j) {
      emit_piece(i, k);
      i = k;
      continue;
    }

    if (is_digit(spans[i].codepoint)) {
      emit_piece(i, i + 1);
      ++i;
      continue;
    }

    j = i;
    if (spans[j].codepoint == static_cast<std::uint32_t>(' ')) {
      ++j;
    }
    k = j;
    while (k < spans.size() && !is_space_any(spans[k].codepoint) && !is_letter(spans[k].codepoint) &&
           !is_mark(spans[k].codepoint) && !is_digit(spans[k].codepoint)) {
      ++k;
    }
    if (k > j) {
      while (k < spans.size() && is_newline(spans[k].codepoint)) {
        ++k;
      }
      emit_piece(i, k);
      i = k;
      continue;
    }

    j = i;
    while (j < spans.size() && is_space_no_newline(spans[j].codepoint)) {
      ++j;
    }
    k = j;
    while (k < spans.size() && is_newline(spans[k].codepoint)) {
      ++k;
    }
    if (k > j) {
      emit_piece(i, k);
      i = k;
      continue;
    }

    if (is_space_any(spans[i].codepoint)) {
      j = i;
      while (j < spans.size() && is_space_any(spans[j].codepoint)) {
        ++j;
      }
      emit_piece(i, j);
      i = j;
      continue;
    }

    emit_piece(i, i + 1);
    ++i;
  }
}

bool QwenTokenizer::apply_bpe(const std::string & transformed_piece, std::vector<std::string> & out_tokens) const {
  auto cache_it = bpe_cache_.find(transformed_piece);
  if (cache_it != bpe_cache_.end()) {
    out_tokens = cache_it->second;
    return true;
  }

  const auto spans = decode_utf8_with_spans(transformed_piece);
  if (spans.empty()) {
    out_tokens.clear();
    return true;
  }

  std::vector<std::string> symbols;
  symbols.reserve(spans.size());
  for (const auto & span : spans) {
    symbols.push_back(transformed_piece.substr(span.begin, span.end - span.begin));
  }

  while (symbols.size() > 1) {
    int best_rank = INT_MAX;
    std::size_t best_pair_index = symbols.size();
    std::pair<std::string, std::string> best_pair;

    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const std::pair<std::string, std::string> pair{symbols[i], symbols[i + 1]};
      const auto it = merge_ranks_.find(pair);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_pair_index = i;
        best_pair = pair;
      }
    }

    if (best_pair_index == symbols.size()) {
      break;
    }

    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (std::size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == best_pair.first && symbols[i + 1] == best_pair.second) {
        merged.push_back(symbols[i] + symbols[i + 1]);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        ++i;
      }
    }
    symbols.swap(merged);
  }

  out_tokens = symbols;
  bpe_cache_.emplace(transformed_piece, out_tokens);
  return true;
}

bool QwenTokenizer::encode_text_segment(
  const std::string & text,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message) const {
  if (text.empty()) {
    return true;
  }

  std::vector<std::string> pieces;
  split_text_for_bpe(text, pieces);

  std::vector<std::string> bpe_parts;
  for (const auto & piece : pieces) {
    std::string transformed;
    transformed.reserve(piece.size() * 2);
    for (const unsigned char b : piece) {
      transformed += byte_to_unicode_[b];
    }

    bpe_parts.clear();
    if (!apply_bpe(transformed, bpe_parts)) {
      error_message = "BPE failed while encoding text.";
      return false;
    }

    for (const auto & bpe_token : bpe_parts) {
      const auto it = vocab_.find(bpe_token);
      if (it == vocab_.end()) {
        error_message = "Tokenizer vocab is missing merged token: '" + bpe_token + "'";
        return false;
      }
      out_tokens.push_back(it->second);
    }
  }

  return true;
}

bool QwenTokenizer::load_from_hf_directory(
  const std::string & model_dir,
  QwenTokenizer & out_tokenizer,
  std::string & error_message) {
  namespace fs = std::filesystem;

  const fs::path root(model_dir);
  if (!fs::exists(root) || !fs::is_directory(root)) {
    error_message = "Tokenizer model directory does not exist: " + model_dir;
    return false;
  }

  const fs::path vocab_path = root / "vocab.json";
  const fs::path merges_path = root / "merges.txt";
  const fs::path config_path = root / "tokenizer_config.json";

  std::string vocab_text;
  std::string merges_text;
  std::string config_text;
  const bool has_config = fs::exists(config_path) && fs::is_regular_file(config_path);
  if (!read_text_file(vocab_path.string(), vocab_text, error_message)) {
    return false;
  }
  if (!read_text_file(merges_path.string(), merges_text, error_message)) {
    return false;
  }
  if (has_config && !read_text_file(config_path.string(), config_text, error_message)) {
    return false;
  }

  QwenTokenizer tokenizer;
  if (!parse_vocab_json(vocab_text, tokenizer.vocab_, tokenizer.id_to_token_, error_message)) {
    return false;
  }
  if (!parse_merges_txt(merges_text, tokenizer.merge_ranks_, error_message)) {
    return false;
  }

  std::unordered_map<std::string, std::int32_t> config_added_tokens;
  std::unordered_map<std::int32_t, std::string> config_added_id_to_token;
  if (has_config &&
      !parse_added_tokens_decoder_from_config(config_text, config_added_tokens, config_added_id_to_token, error_message)) {
    return false;
  }

  build_byte_level_maps(tokenizer.byte_to_unicode_, tokenizer.unicode_to_byte_);

  tokenizer.added_tokens_.clear();
  tokenizer.added_token_id_to_content_.clear();
  for (const auto & kv : tokenizer.vocab_) {
    if (looks_like_added_token(kv.first)) {
      tokenizer.added_tokens_[kv.first] = kv.second;
      tokenizer.added_token_id_to_content_[kv.second] = kv.first;
    }
  }
  for (const auto & kv : config_added_tokens) {
    tokenizer.added_tokens_[kv.first] = kv.second;
    tokenizer.vocab_[kv.first] = kv.second;
  }
  for (const auto & kv : config_added_id_to_token) {
    tokenizer.added_token_id_to_content_[kv.first] = kv.second;
  }

  tokenizer.added_tokens_by_first_byte_.clear();
  for (const auto & kv : tokenizer.added_tokens_) {
    if (!kv.first.empty()) {
      const unsigned char first = static_cast<unsigned char>(kv.first[0]);
      tokenizer.added_tokens_by_first_byte_[first].push_back(kv.first);
    }
  }
  for (auto & kv : tokenizer.added_tokens_by_first_byte_) {
    auto & tokens = kv.second;
    std::sort(tokens.begin(), tokens.end(), [](const std::string & a, const std::string & b) {
      return a.size() > b.size();
    });
  }

  tokenizer.max_token_id_ = -1;
  for (const auto & kv : tokenizer.vocab_) {
    tokenizer.max_token_id_ = std::max(tokenizer.max_token_id_, kv.second);
  }
  for (const auto & kv : tokenizer.added_token_id_to_content_) {
    tokenizer.max_token_id_ = std::max(tokenizer.max_token_id_, kv.first);
  }

  out_tokenizer = std::move(tokenizer);
  return true;
}

bool QwenTokenizer::encode(
  const std::string & text,
  std::vector<std::int32_t> & out_tokens,
  std::string & error_message) const {
  out_tokens.clear();
  if (text.empty()) {
    return true;
  }

  std::size_t segment_begin = 0;
  std::size_t i = 0;
  while (i < text.size()) {
    std::size_t best_len = 0;
    std::int32_t best_id = -1;
    const auto it = added_tokens_by_first_byte_.find(static_cast<unsigned char>(text[i]));
    if (it != added_tokens_by_first_byte_.end()) {
      for (const auto & token : it->second) {
        if (token.size() > best_len && i + token.size() <= text.size() && text.compare(i, token.size(), token) == 0) {
          const auto id_it = added_tokens_.find(token);
          if (id_it != added_tokens_.end()) {
            best_len = token.size();
            best_id = id_it->second;
          }
        }
      }
    }

    if (best_len > 0) {
      if (i > segment_begin) {
        if (!encode_text_segment(text.substr(segment_begin, i - segment_begin), out_tokens, error_message)) {
          return false;
        }
      }
      out_tokens.push_back(best_id);
      i += best_len;
      segment_begin = i;
    } else {
      ++i;
    }
  }

  if (segment_begin < text.size()) {
    if (!encode_text_segment(text.substr(segment_begin), out_tokens, error_message)) {
      return false;
    }
  }

  return true;
}

bool QwenTokenizer::decode(
  const std::vector<std::int32_t> & token_ids,
  std::string & out_text,
  std::string & error_message) const {
  out_text.clear();
  for (const auto token_id : token_ids) {
    auto added_it = added_token_id_to_content_.find(token_id);
    if (added_it != added_token_id_to_content_.end()) {
      out_text += added_it->second;
      continue;
    }

    if (token_id < 0 || static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
      error_message = "Token id is out of tokenizer range: " + std::to_string(token_id);
      return false;
    }
    const std::string & token = id_to_token_[static_cast<std::size_t>(token_id)];
    if (token.empty()) {
      error_message = "Tokenizer does not have text for token id: " + std::to_string(token_id);
      return false;
    }

    const auto spans = decode_utf8_with_spans(token);
    for (const auto & span : spans) {
      const auto it = unicode_to_byte_.find(span.codepoint);
      if (it != unicode_to_byte_.end()) {
        out_text.push_back(static_cast<char>(it->second));
      } else {
        out_text.append(token, span.begin, span.end - span.begin);
      }
    }
  }
  return true;
}

std::optional<std::int32_t> QwenTokenizer::token_to_id(const std::string & token) const {
  const auto it = vocab_.find(token);
  if (it != vocab_.end()) {
    return it->second;
  }
  const auto added_it = added_tokens_.find(token);
  if (added_it != added_tokens_.end()) {
    return added_it->second;
  }
  return std::nullopt;
}

std::optional<std::string> QwenTokenizer::id_to_token(const std::int32_t token_id) const {
  const auto added_it = added_token_id_to_content_.find(token_id);
  if (added_it != added_token_id_to_content_.end()) {
    return added_it->second;
  }
  if (token_id < 0 || static_cast<std::size_t>(token_id) >= id_to_token_.size()) {
    return std::nullopt;
  }
  if (id_to_token_[static_cast<std::size_t>(token_id)].empty()) {
    return std::nullopt;
  }
  return id_to_token_[static_cast<std::size_t>(token_id)];
}

std::size_t QwenTokenizer::vocab_size() const {
  if (max_token_id_ < 0) {
    return 0;
  }
  return static_cast<std::size_t>(max_token_id_) + 1;
}

} // namespace qwen35x
