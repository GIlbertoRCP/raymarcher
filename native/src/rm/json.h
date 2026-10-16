// Minimal JSON reader for the files the compiler writes (catalog.json and the
// metric-agreement fixture). Host only. Numbers are parsed with strtod, so the
// shortest round-trip decimals written by JavaScript come back bit-exact.
#pragma once

#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace rm {
namespace json {

struct Value {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double n = 0;
  std::string s;
  std::vector<Value> a;
  std::map<std::string, Value> o;

  const Value& operator[](const std::string& key) const {
    auto it = o.find(key);
    if (kind != Object || it == o.end()) throw std::runtime_error("JSON: missing key '" + key + "'");
    return it->second;
  }
  const Value& operator[](size_t i) const {
    if (kind != Array || i >= a.size()) throw std::runtime_error("JSON: index out of range");
    return a[i];
  }
  bool has(const std::string& key) const { return kind == Object && o.count(key) > 0; }
  size_t size() const { return kind == Array ? a.size() : o.size(); }
  double num() const {
    if (kind != Number) throw std::runtime_error("JSON: expected a number");
    return n;
  }
  const std::string& str() const {
    if (kind != String) throw std::runtime_error("JSON: expected a string");
    return s;
  }
  std::vector<double> numbers() const {
    std::vector<double> out;
    for (const Value& v : a) out.push_back(v.num());
    return out;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& text) : t_(text) {}

  Value parse() {
    Value v = value();
    ws();
    if (i_ != t_.size()) fail("trailing characters");
    return v;
  }

 private:
  const std::string& t_;
  size_t i_ = 0;

  [[noreturn]] void fail(const char* what) {
    throw std::runtime_error(std::string("JSON: ") + what + " at offset " + std::to_string(i_));
  }
  void ws() {
    while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\n' || t_[i_] == '\r' || t_[i_] == '\t')) i_++;
  }
  bool eat(char c) {
    ws();
    if (i_ < t_.size() && t_[i_] == c) {
      i_++;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c)) fail("unexpected character");
  }
  bool literal(const char* word) {
    size_t n = std::char_traits<char>::length(word);
    if (t_.compare(i_, n, word) == 0) {
      i_ += n;
      return true;
    }
    return false;
  }

  std::string string() {
    expect('"');
    std::string out;
    while (i_ < t_.size() && t_[i_] != '"') {
      char c = t_[i_++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= t_.size()) fail("bad escape");
      char e = t_[i_++];
      switch (e) {
        case 'n': out += '\n'; break;
        case 't': out += '\t'; break;
        case 'r': out += '\r'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'u': {
          if (i_ + 4 > t_.size()) fail("bad \\u escape");
          unsigned code = std::stoul(t_.substr(i_, 4), nullptr, 16);
          i_ += 4;
          // UTF-8 encode (the compiler only writes BMP characters).
          if (code < 0x80) {
            out += char(code);
          } else if (code < 0x800) {
            out += char(0xC0 | (code >> 6));
            out += char(0x80 | (code & 0x3F));
          } else {
            out += char(0xE0 | (code >> 12));
            out += char(0x80 | ((code >> 6) & 0x3F));
            out += char(0x80 | (code & 0x3F));
          }
          break;
        }
        default: out += e;
      }
    }
    if (i_ >= t_.size()) fail("unterminated string");
    i_++;
    return out;
  }

  Value value() {
    ws();
    if (i_ >= t_.size()) fail("unexpected end");
    Value v;
    char c = t_[i_];
    if (c == '{') {
      i_++;
      v.kind = Value::Object;
      if (eat('}')) return v;
      do {
        ws();
        std::string key = string();
        expect(':');
        v.o[key] = value();
      } while (eat(','));
      expect('}');
    } else if (c == '[') {
      i_++;
      v.kind = Value::Array;
      if (eat(']')) return v;
      do v.a.push_back(value());
      while (eat(','));
      expect(']');
    } else if (c == '"') {
      v.kind = Value::String;
      v.s = string();
    } else if (literal("true")) {
      v.kind = Value::Bool;
      v.b = true;
    } else if (literal("false")) {
      v.kind = Value::Bool;
    } else if (literal("null")) {
      v.kind = Value::Null;
    } else {
      char* end = nullptr;
      v.kind = Value::Number;
      v.n = std::strtod(t_.c_str() + i_, &end);
      if (end == t_.c_str() + i_) fail("unexpected character");
      i_ = size_t(end - t_.c_str());
    }
    return v;
  }
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

}  // namespace json
}  // namespace rm
