#include "skyshot/common.hpp"

#include <cctype>
#include <filesystem>
#include <charconv>
#include <fstream>
#include <sstream>

namespace skyshot {

std::optional<double> parse_f64(std::string_view s) {
  if (s.empty()) return std::nullopt;
  bool neg = false;
  std::string_view body = s;
  if (body[0] == '+' || body[0] == '-') {
    neg = body[0] == '-';
    body.remove_prefix(1);
  }
  if (body.empty()) return std::nullopt;
  // Rust accepts "inf", "infinity", "nan" case-insensitively.
  std::string low = ascii_lower(body);
  if (low == "inf" || low == "infinity") return neg ? -INF : INF;
  if (low == "nan") return NaN;
  // Validate the grammar: digits [. digits] [e[+-]digits] with at least one digit in the mantissa.
  size_t i = 0, digits = 0;
  while (i < body.size() && std::isdigit(static_cast<unsigned char>(body[i]))) ++i, ++digits;
  if (i < body.size() && body[i] == '.') {
    ++i;
    while (i < body.size() && std::isdigit(static_cast<unsigned char>(body[i]))) ++i, ++digits;
  }
  if (digits == 0) return std::nullopt;
  if (i < body.size() && (body[i] == 'e' || body[i] == 'E')) {
    ++i;
    if (i < body.size() && (body[i] == '+' || body[i] == '-')) ++i;
    size_t ed = 0;
    while (i < body.size() && std::isdigit(static_cast<unsigned char>(body[i]))) ++i, ++ed;
    if (ed == 0) return std::nullopt;
  }
  if (i != body.size()) return std::nullopt;
  // from_chars does not accept a leading '.' only in some implementations; normalise.
  std::string norm;
  if (body[0] == '.') norm = "0";
  norm.append(body);
  double v = 0;
  auto r = std::from_chars(norm.data(), norm.data() + norm.size(), v);
  if (r.ec == std::errc::result_out_of_range) {
    // Overflow -> inf, underflow -> 0 (as Rust does).
    v = std::strtod(norm.c_str(), nullptr);
  } else if (r.ec != std::errc()) {
    return std::nullopt;
  }
  return neg ? -v : v;
}

std::vector<uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
  if (!f) throw Error("open " + path);
  f.seekg(0, std::ios::end);
  auto n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> out(static_cast<size_t>(n));
  if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
  if (!f && n > 0) throw Error("read " + path);
  return out;
}
std::string read_file_string(const std::string& path) {
  auto b = read_file_bytes(path);
  return std::string(b.begin(), b.end());
}

std::string trim(std::string_view s) {
  size_t a = 0, b = s.size();
  auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
  while (a < b && ws(s[a])) ++a;
  while (b > a && ws(s[b - 1])) --b;
  return std::string(s.substr(a, b - a));
}

std::string ascii_lower(std::string_view s) {
  std::string o(s);
  for (auto& c : o) c = char(std::tolower(static_cast<unsigned char>(c)));
  return o;
}

}  // namespace skyshot
