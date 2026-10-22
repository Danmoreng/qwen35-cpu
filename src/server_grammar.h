#pragma once
#include "server_protocol.h"
#include <bitset>
#include <set>

namespace qwen35x::server {
// Small byte grammar with persistent pushdown states. Grammar construction is
// bounded and contains no left recursion or nullable cycles. UTF-8 and JSON
// escapes are recognized before a token can reach the sampler.
class ByteGrammar {
public:
  using Stack = std::vector<int>;
  using State = std::vector<Stack>;
  using Alternatives = std::vector<Stack>;
  std::vector<Alternatives> rules;
  std::vector<std::bitset<256>> terminals;
  int start = 0;
  int rule(Alternatives alternatives = {}) {
    require(rules.size() < 8192, "Schema grammar exceeds rule budget");
    rules.push_back(std::move(alternatives)); return int(rules.size() - 1);
  }
  int bytes(unsigned lo, unsigned hi) {
    std::bitset<256> set;
    for (unsigned i = lo; i <= hi; ++i) set.set(i);
    return charset(set);
  }
  int charset(std::bitset<256> set) {
    for (std::size_t i = 0; i < terminals.size(); ++i)
      if (terminals[i] == set) return -int(i) - 1;
    terminals.push_back(set); return -int(terminals.size());
  }
  int literal(const std::string &text) {
    Stack symbols;
    for (unsigned char c : text) symbols.push_back(bytes(c, c));
    return rule({symbols});
  }
  State expand(State pending) const {
    State result;
    std::size_t steps = 0;
    while (!pending.empty()) {
      require(++steps <= 32768 && pending.size() + result.size() <= 4096,
              "Schema grammar exceeds state budget");
      auto stack = std::move(pending.back()); pending.pop_back();
      require(stack.size() <= 512, "JSON nesting exceeds grammar budget");
      if (stack.empty() || stack.back() < 0) { result.push_back(std::move(stack)); continue; }
      const int id = stack.back(); stack.pop_back();
      for (const auto &alternative : rules.at(id)) {
        auto next = stack;
        next.insert(next.end(), alternative.rbegin(), alternative.rend());
        pending.push_back(std::move(next));
      }
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }
  State initial() const { return expand({{start}}); }
  State advance(const State &state, unsigned char byte) const {
    State next;
    for (const auto &stack : state)
      if (!stack.empty() && terminals[-stack.back() - 1].test(byte)) {
        next.push_back(stack); next.back().pop_back();
      }
    return expand(std::move(next));
  }
  bool complete(const State &state) const {
    return std::any_of(state.begin(), state.end(), [](const auto &stack) { return stack.empty(); });
  }
  bool consume(State &state, std::string_view text) const {
    for (unsigned char byte : text) {
      state = advance(state, byte);
      if (state.empty()) return false;
    }
    return true;
  }
};

class JsonGrammarBuilder {
  ByteGrammar g_;
  int ws_, string_, number_, integer_, boolean_, null_;
  std::size_t nodes_ = 0;
  int ch(unsigned c) { return g_.bytes(c, c); }
  int sequence(ByteGrammar::Stack symbols) { return g_.rule({std::move(symbols)}); }
  int optional(int rule) { return g_.rule({{}, {rule}}); }
  int many(int value) {
    const auto r = g_.rule(); g_.rules[r] = {{}, {value, r}}; return r;
  }
  int schema(const Json &s, unsigned depth) {
    require(depth <= 12 && ++nodes_ <= 256, "Schema exceeds depth/node limit");
    fields(s, {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
               "enum", "const", "anyOf", "description", "title"});
    for (auto key : {"description", "title"})
      if (s.contains(key)) require(s[key].is_string(), std::string(key) + " must be text");
    if (s.contains("anyOf")) {
      fields(s, {"anyOf", "description", "title"});
      require(!s.contains("type") && s.size() <= 3 && s["anyOf"].is_array() &&
              !s["anyOf"].empty() && s["anyOf"].size() <= 16, "Invalid anyOf");
      ByteGrammar::Alternatives choices;
      for (const auto &part : s["anyOf"]) choices.push_back({schema(part, depth + 1)});
      return g_.rule(std::move(choices));
    }
    const auto type = s.at("type").get<std::string>();
    require(type == "object" || type == "array" || type == "string" || type == "number" ||
            type == "integer" || type == "boolean" || type == "null", "Unsupported schema type");
    for (auto key : {"properties", "required", "additionalProperties"})
      require(type == "object" || !s.contains(key), std::string(key) + " requires object type");
    for (auto key : {"items", "minItems", "maxItems"})
      require(type == "array" || !s.contains(key), std::string(key) + " requires array type");
    if (s.contains("enum") || s.contains("const")) {
      require(!(s.contains("enum") && s.contains("const")), "Specify enum or const, not both");
      require(type != "object" && type != "array", "Enums support scalar types only");
      const auto values = s.contains("enum") ? s["enum"] : Json::array({s["const"]});
      require(values.is_array() && !values.empty() && values.size() <= 64, "enum must contain 1..64 scalar values");
      ByteGrammar::Alternatives choices;
      for (const auto &v : values) {
        require((type == "string" && v.is_string()) || (type == "number" && v.is_number()) ||
                (type == "integer" && v.is_number_integer()) || (type == "boolean" && v.is_boolean()) ||
                (type == "null" && v.is_null()), "Enum value differs from declared type");
        require(v.dump().size() <= 1024, "Enum value exceeds byte limit");
        choices.push_back({g_.literal(v.dump())});
      }
      return g_.rule(std::move(choices));
    }
    if (type == "string") return string_;
    if (type == "number") return number_;
    if (type == "integer") return integer_;
    if (type == "boolean") return boolean_;
    if (type == "null") return null_;
    if (type == "array") {
      require(s.contains("items"), "Array schemas require items");
      for (const auto *key : {"minItems", "maxItems"})
        if (s.contains(key)) require(s[key].is_number_integer(), std::string(key) + " must be an integer");
      const auto min = integer(s, "minItems", 0), max = integer(s, "maxItems", 64);
      require(min >= 0 && max >= min && max <= 64, "Array bounds must be within 0..64");
      const auto item = schema(s["items"], depth + 1);
      int tail = g_.rule({{}});
      for (int n = int(max) - 1; n >= 0; --n) {
        ByteGrammar::Stack row;
        if (n) row = {ch(','), ws_};
        row.insert(row.end(), {item, ws_, tail});
        ByteGrammar::Alternatives choices = {row};
        if (n >= min) choices.push_back({});
        tail = g_.rule(std::move(choices));
      }
      return sequence({ch('['), ws_, tail, ch(']')});
    }
    require(s.contains("properties") && s["properties"].is_object() && s["properties"].size() <= 32 &&
            s.contains("additionalProperties") && s["additionalProperties"] == false,
            "Objects require <=32 properties and additionalProperties=false");
    std::set<std::string> required;
    if (s.contains("required")) {
      require(s["required"].is_array(), "required must be an array");
      for (const auto &value : s["required"]) {
        auto name = value.get<std::string>();
        require(s["properties"].contains(name) && required.insert(name).second, "Unknown/duplicate required property");
      }
    }
    // Two continuations track whether a comma is necessary. Output property
    // order follows the schema; optional properties may be omitted.
    int empty = g_.rule({{}}), nonempty = empty;
    const auto &properties = s["properties"];
    std::vector<std::pair<std::string, Json>> rows;
    for (const auto &[name, value] : properties.items()) rows.emplace_back(name, value);
    for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
      require(it->first.size() <= 128, "Property name exceeds 128 bytes");
      const int key = g_.literal(Json(it->first).dump()), value = schema(it->second, depth + 1);
      ByteGrammar::Alternatives first = {{key, ws_, ch(':'), ws_, value, ws_, nonempty}};
      ByteGrammar::Alternatives rest = {{ch(','), ws_, key, ws_, ch(':'), ws_, value, ws_, nonempty}};
      if (!required.contains(it->first)) { first.push_back({empty}); rest.push_back({nonempty}); }
      empty = g_.rule(std::move(first)); nonempty = g_.rule(std::move(rest));
    }
    return sequence({ch('{'), ws_, empty, ch('}')});
  }
public:
  JsonGrammarBuilder() {
    std::bitset<256> space;
    for (auto c : {' ', '\t', '\r', '\n'}) space.set(c);
    ws_ = many(g_.charset(space));
    const auto digit = g_.bytes('0', '9');
    // Keep generated numeric literals representable by ordinary JSON clients.
    int digits = g_.rule({{}});
    for (int i = 0; i < 17; ++i) digits = g_.rule({{}, {digit, digits}});
    const auto magnitude = g_.rule({{ch('0')}, {g_.bytes('1', '9'), digits}});
    integer_ = sequence({optional(ch('-')), magnitude});
    const auto fraction = sequence({ch('.'), digit, digits});
    const auto exponent = sequence({g_.rule({{ch('e')}, {ch('E')}}),
      optional(g_.rule({{ch('+')}, {ch('-')}})), digit, optional(digit)});
    number_ = sequence({integer_, optional(fraction), optional(exponent)});
    boolean_ = g_.rule({{g_.literal("true")}, {g_.literal("false")}});
    null_ = g_.literal("null");
    const auto hex = g_.rule({{digit}, {g_.bytes('a','f')}, {g_.bytes('A','F')}});
    const auto d = g_.rule({{ch('d')}, {ch('D')}});
    const auto non_surrogate = g_.rule({{g_.bytes('0','9')}, {g_.bytes('a','c')},
      {g_.bytes('A','C')}, {g_.bytes('e','f')}, {g_.bytes('E','F')}});
    const auto codepoint = g_.rule({{non_surrogate, hex, hex, hex},
      {d, g_.bytes('0','7'), hex, hex},
      {d, g_.rule({{g_.bytes('8','9')}, {g_.bytes('a','b')}, {g_.bytes('A','B')}}), hex, hex,
       ch('\\'), ch('u'), d, g_.rule({{g_.bytes('c','f')}, {g_.bytes('C','F')}}), hex, hex}});
    const auto escape = g_.rule({{g_.literal("\\\"")}, {g_.literal("\\\\")}, {g_.literal("\\/")},
      {g_.literal("\\b")}, {g_.literal("\\f")}, {g_.literal("\\n")}, {g_.literal("\\r")},
      {g_.literal("\\t")}, {ch('\\'), ch('u'), codepoint}});
    std::bitset<256> ascii;
    for (unsigned c = 32; c < 128; ++c) if (c != '"' && c != '\\') ascii.set(c);
    const auto cont = g_.bytes(0x80, 0xbf);
    const auto character = g_.rule({{g_.charset(ascii)}, {escape},
      {g_.bytes(0xc2,0xdf),cont}, {ch(0xe0),g_.bytes(0xa0,0xbf),cont},
      {g_.bytes(0xe1,0xec),cont,cont}, {ch(0xed),g_.bytes(0x80,0x9f),cont},
      {g_.bytes(0xee,0xef),cont,cont}, {ch(0xf0),g_.bytes(0x90,0xbf),cont,cont},
      {g_.bytes(0xf1,0xf3),cont,cont,cont}, {ch(0xf4),g_.bytes(0x80,0x8f),cont,cont}});
    string_ = sequence({ch('"'), many(character), ch('"')});
  }
  ByteGrammar build(const Json &s) {
    require(s.dump().size() <= 32768, "Schema exceeds 32768 bytes");
    const int root = schema(s, 0);
    g_.start = sequence({ws_, root, ws_});
    (void)g_.initial(); return std::move(g_);
  }
  ByteGrammar object() {
    // General JSON objects; nested containers are capped at eight levels.
    int value = g_.rule({{string_}, {number_}, {boolean_}, {null_}}), object = 0;
    for (unsigned depth = 0; depth < 8; ++depth) {
      const auto member = sequence({string_, ws_, ch(':'), ws_, value, ws_});
      object = sequence({ch('{'), ws_, optional(sequence({member, many(sequence({ch(','), ws_, member}))})), ch('}')});
      const auto item = sequence({value, ws_});
      const auto array = sequence({ch('['), ws_, optional(sequence({item, many(sequence({ch(','), ws_, item}))})), ch(']')});
      value = g_.rule({{string_}, {number_}, {boolean_}, {null_}, {object}, {array}});
    }
    g_.start = sequence({ws_, object, ws_}); return std::move(g_);
  }
};
} // namespace qwen35x::server
