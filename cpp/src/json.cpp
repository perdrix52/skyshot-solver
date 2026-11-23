#include "skyshot/json.hpp"

#include <charconv>
#include <cstdio>
#include <cstdlib>

namespace skyshot {

double Json::as_f64() const {
  switch (type_) {
    case Type::Int: return double(i_);
    case Type::UInt: return double(u_);
    case Type::Double: return d_;
    default: throw Error("expected a number");
  }
}
std::optional<uint64_t> Json::as_u64() const {
  if (type_ == Type::UInt) return u_;
  if (type_ == Type::Int && i_ >= 0) return uint64_t(i_);
  return std::nullopt;
}
std::optional<int64_t> Json::as_i64() const {
  if (type_ == Type::Int) return i_;
  if (type_ == Type::UInt && u_ <= uint64_t(INT64_MAX)) return int64_t(u_);
  return std::nullopt;
}

template <class F>
static std::string format_shortest(F v) {
  if (!std::isfinite(v)) return "null";
  if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
  char buf[64];
  auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  std::string s(buf, r.ptr);
  bool neg = false;
  size_t p = 0;
  if (s[0] == '-') {
    neg = true;
    p = 1;
  }
  size_t e = s.find('e');
  std::string mant = s.substr(p, e - p);
  int exp10 = std::atoi(s.c_str() + e + 1);
  std::string digits;
  for (char c : mant)
    if (c != '.') digits.push_back(c);
  int n = int(digits.size());
  int kk = exp10 + 1;  // value = 0.d1d2... * 10^kk
  std::string out = neg ? "-" : "";
  if (0 < kk && kk <= 16) {
    if (n <= kk) {
      out += digits;
      out.append(size_t(kk - n), '0');
      out += ".0";
    } else {
      out += digits.substr(0, size_t(kk));
      out += '.';
      out += digits.substr(size_t(kk));
    }
  } else if (-5 < kk && kk <= 0) {
    out += "0.";
    out.append(size_t(-kk), '0');
    out += digits;
  } else {
    out += digits[0];
    if (n > 1) {
      out += '.';
      out += digits.substr(1);
    }
    out += 'e';
    out += std::to_string(kk - 1);
  }
  return out;
}
std::string format_f64(double v) { return format_shortest<double>(v); }
std::string format_f32(float v) { return format_shortest<float>(v); }

static void escape(std::string& out, const std::string& s) {
  out.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof b, "\\u%04x", c);
          out += b;
        } else {
          out.push_back(char(c));
        }
    }
  }
  out.push_back('"');
}

void Json::dump_to(std::string& out, bool pretty, int indent) const {
  auto nl = [&](int level) {
    out.push_back('\n');
    out.append(size_t(level) * 2, ' ');
  };
  switch (type_) {
    case Type::Null: out += "null"; break;
    case Type::Bool: out += b_ ? "true" : "false"; break;
    case Type::Int: out += std::to_string(i_); break;
    case Type::UInt: out += std::to_string(u_); break;
    case Type::Double: out += std::isfinite(d_) ? format_f64(d_) : "null"; break;
    case Type::String: escape(out, s_); break;
    case Type::Array:
      if (a_.empty()) {
        out += "[]";
        break;
      }
      out.push_back('[');
      for (size_t i = 0; i < a_.size(); ++i) {
        if (i) out.push_back(',');
        if (pretty) nl(indent + 1);
        a_[i].dump_to(out, pretty, indent + 1);
      }
      if (pretty) nl(indent);
      out.push_back(']');
      break;
    case Type::Object:
      if (o_.empty()) {
        out += "{}";
        break;
      }
      out.push_back('{');
      for (size_t i = 0; i < o_.size(); ++i) {
        if (i) out.push_back(',');
        if (pretty) nl(indent + 1);
        escape(out, o_[i].first);
        out += pretty ? ": " : ":";
        o_[i].second.dump_to(out, pretty, indent + 1);
      }
      if (pretty) nl(indent);
      out.push_back('}');
      break;
  }
}

std::string Json::dump(bool pretty) const {
  std::string s;
  dump_to(s, pretty, 0);
  return s;
}

namespace {
struct Parser {
  std::string_view t;
  size_t p = 0;
  [[noreturn]] void fail(const char* m) const {
    throw Error(std::string(m) + " at offset " + std::to_string(p));
  }
  void ws() {
    while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' || t[p] == '\r')) ++p;
  }
  bool lit(const char* s) {
    size_t n = std::strlen(s);
    if (t.substr(p, n) == s) {
      p += n;
      return true;
    }
    return false;
  }
  static void put_utf8(std::string& o, uint32_t c) {
    if (c < 0x80) o.push_back(char(c));
    else if (c < 0x800) {
      o.push_back(char(0xC0 | (c >> 6)));
      o.push_back(char(0x80 | (c & 63)));
    } else if (c < 0x10000) {
      o.push_back(char(0xE0 | (c >> 12)));
      o.push_back(char(0x80 | ((c >> 6) & 63)));
      o.push_back(char(0x80 | (c & 63)));
    } else {
      o.push_back(char(0xF0 | (c >> 18)));
      o.push_back(char(0x80 | ((c >> 12) & 63)));
      o.push_back(char(0x80 | ((c >> 6) & 63)));
      o.push_back(char(0x80 | (c & 63)));
    }
  }
  uint32_t hex4() {
    if (p + 4 > t.size()) fail("bad escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      char c = t[p++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else fail("bad hex digit");
    }
    return v;
  }
  std::string str() {
    ++p;  // opening quote
    std::string o;
    while (true) {
      if (p >= t.size()) fail("unterminated string");
      char c = t[p++];
      if (c == '"') break;
      if (c == '\\') {
        if (p >= t.size()) fail("bad escape");
        char e = t[p++];
        switch (e) {
          case '"': o.push_back('"'); break;
          case '\\': o.push_back('\\'); break;
          case '/': o.push_back('/'); break;
          case 'b': o.push_back('\b'); break;
          case 'f': o.push_back('\f'); break;
          case 'n': o.push_back('\n'); break;
          case 'r': o.push_back('\r'); break;
          case 't': o.push_back('\t'); break;
          case 'u': {
            uint32_t u = hex4();
            if (u >= 0xD800 && u < 0xDC00 && t.substr(p, 2) == "\\u") {
              p += 2;
              uint32_t lo = hex4();
              u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
            }
            put_utf8(o, u);
            break;
          }
          default: fail("bad escape");
        }
      } else if (static_cast<unsigned char>(c) < 0x20) {
        fail("control character in string");
      } else {
        o.push_back(c);
      }
    }
    return o;
  }
  Json number() {
    size_t s = p;
    bool is_int = true;
    if (t[p] == '-') ++p;
    if (p >= t.size() || !(t[p] >= '0' && t[p] <= '9')) fail("invalid number");
    if (t[p] == '0') ++p;
    else
      while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
    if (p < t.size() && t[p] == '.') {
      is_int = false;
      ++p;
      if (p >= t.size() || !(t[p] >= '0' && t[p] <= '9')) fail("invalid number");
      while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
    }
    if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
      is_int = false;
      ++p;
      if (p < t.size() && (t[p] == '+' || t[p] == '-')) ++p;
      if (p >= t.size() || !(t[p] >= '0' && t[p] <= '9')) fail("invalid number");
      while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
    }
    std::string_view num = t.substr(s, p - s);
    if (is_int) {
      if (num[0] == '-') {
        int64_t v;
        auto r = std::from_chars(num.data(), num.data() + num.size(), v);
        if (r.ec == std::errc()) return Json(v);
      } else {
        uint64_t v;
        auto r = std::from_chars(num.data(), num.data() + num.size(), v);
        if (r.ec == std::errc()) return Json(v);
      }
    }
    double d;
    auto r = std::from_chars(num.data(), num.data() + num.size(), d);
    if (r.ec == std::errc::result_out_of_range) fail("number out of range");
    if (r.ec != std::errc()) fail("invalid number");
    return Json(d);
  }
  Json value(int depth) {
    if (depth > 128) fail("recursion limit exceeded");
    ws();
    if (p >= t.size()) fail("unexpected end of input");
    char c = t[p];
    if (c == '{') {
      ++p;
      Json o = Json::object();
      ws();
      if (p < t.size() && t[p] == '}') {
        ++p;
        return o;
      }
      while (true) {
        ws();
        if (p >= t.size() || t[p] != '"') fail("expected string key");
        std::string k = str();
        ws();
        if (p >= t.size() || t[p] != ':') fail("expected ':'");
        ++p;
        Json v = value(depth + 1);
        o.set(k, std::move(v));
        ws();
        if (p < t.size() && t[p] == ',') {
          ++p;
          continue;
        }
        if (p < t.size() && t[p] == '}') {
          ++p;
          return o;
        }
        fail("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++p;
      Json a = Json::array();
      ws();
      if (p < t.size() && t[p] == ']') {
        ++p;
        return a;
      }
      while (true) {
        a.push(value(depth + 1));
        ws();
        if (p < t.size() && t[p] == ',') {
          ++p;
          continue;
        }
        if (p < t.size() && t[p] == ']') {
          ++p;
          return a;
        }
        fail("expected ',' or ']'");
      }
    }
    if (c == '"') return Json(str());
    if (lit("true")) return Json(true);
    if (lit("false")) return Json(false);
    if (lit("null")) return Json();
    if (c == '-' || (c >= '0' && c <= '9')) return number();
    fail("unexpected character");
  }
};
}  // namespace

Json parse_json(std::string_view text) {
  Parser ps{text};
  Json v = ps.value(0);
  ps.ws();
  if (ps.p != text.size()) ps.fail("trailing characters");
  return v;
}

}  // namespace skyshot
