// Shared helpers that reproduce Rust numeric semantics used by the original solver.
#pragma once
#include <bit>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace skyshot {

inline constexpr double PI = 3.14159265358979323846264338327950288;
inline constexpr double TAU = 2.0 * PI;
inline constexpr double FRAC_PI_2 = PI / 2.0;
inline constexpr double FRAC_PI_4 = PI / 4.0;
inline constexpr double SQRT_2 = 1.41421356237309504880168872420969808;
inline constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
inline constexpr double INF = std::numeric_limits<double>::infinity();

// Error type used where the Rust code returns anyhow::Error.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// f64::to_radians / to_degrees use exactly these constants in Rust.
inline double to_radians(double d) { return d * (PI / 180.0); }
inline double to_degrees(double r) { return r * (180.0 / PI); }

// f64::rem_euclid
inline double rem_euclid(double a, double b) {
  double r = std::fmod(a, b);
  return r < 0.0 ? r + std::fabs(b) : r;
}
inline int64_t rem_euclid_i(int64_t a, int64_t b) {
  int64_t r = a % b;
  return r < 0 ? r + (b < 0 ? -b : b) : r;
}
inline int64_t div_euclid_i(int64_t a, int64_t b) {
  int64_t q = a / b;
  if (a % b < 0) return b > 0 ? q - 1 : q + 1;
  return q;
}

// f64::powi (repeated multiplication, as LLVM's powi intrinsic does).
inline double powi(double x, int n) {
  bool neg = n < 0;
  unsigned e = neg ? 0u - static_cast<unsigned>(n) : static_cast<unsigned>(n);
  double r = 1.0, b = x;
  while (e) {
    if (e & 1) r *= b;
    b *= b;
    e >>= 1;
  }
  return neg ? 1.0 / r : r;
}

inline double clampd(double v, double lo, double hi) {
  // f64::clamp: NaN stays NaN.
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}
// f64::max / min ignore NaN operands (return the other one).
inline double fmax_r(double a, double b) {
  if (std::isnan(a)) return b;
  if (std::isnan(b)) return a;
  return a > b ? a : b;
}
inline double fmin_r(double a, double b) {
  if (std::isnan(a)) return b;
  if (std::isnan(b)) return a;
  return a < b ? a : b;
}

inline float fmax_r(float a, float b) {
  if (std::isnan(a)) return b;
  if (std::isnan(b)) return a;
  return a > b ? a : b;
}
inline float fmin_r(float a, float b) {
  if (std::isnan(a)) return b;
  if (std::isnan(b)) return a;
  return a < b ? a : b;
}
inline float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// Rust `as` casts from float saturate and map NaN to 0.
template <class T, class F>
inline T sat_cast(F vf) {
  double v = double(vf);
  if (std::isnan(v)) return T(0);
  constexpr double lo = static_cast<double>(std::numeric_limits<T>::lowest());
  constexpr double hi = static_cast<double>(std::numeric_limits<T>::max());
  if (v <= lo) return std::numeric_limits<T>::lowest();
  if (v >= hi) return std::numeric_limits<T>::max();
  return static_cast<T>(v);
}

// f64::signum (NaN stays NaN, +-0.0 give +-1.0).
inline double signum(double v) {
  if (std::isnan(v)) return v;
  return std::signbit(v) ? -1.0 : 1.0;
}

// f64/f32::total_cmp: -1, 0, 1 following IEEE 754 totalOrder.
inline int total_cmp(double a, double b) {
  auto x = std::bit_cast<int64_t>(a);
  auto y = std::bit_cast<int64_t>(b);
  x ^= int64_t(uint64_t(x >> 63) >> 1);
  y ^= int64_t(uint64_t(y >> 63) >> 1);
  return x < y ? -1 : (x > y ? 1 : 0);
}
inline int total_cmp(float a, float b) {
  auto x = std::bit_cast<int32_t>(a);
  auto y = std::bit_cast<int32_t>(b);
  x ^= int32_t(uint32_t(x >> 31) >> 1);
  y ^= int32_t(uint32_t(y >> 31) >> 1);
  return x < y ? -1 : (x > y ? 1 : 0);
}

// Whole-file helpers; throw skyshot::Error on failure.
std::vector<uint8_t> read_file_bytes(const std::string& path);
std::string read_file_string(const std::string& path);

// Rust `str::parse::<f64>()`: no surrounding whitespace, accepts a leading '+', "inf", "nan", "1e5", ".5", "5.".
std::optional<double> parse_f64(std::string_view s);
// ASCII-whitespace trim and case helpers (Rust trim() also strips Unicode spaces; headers are ASCII).
std::string trim(std::string_view s);
std::string ascii_lower(std::string_view s);

using Vec3 = std::array<double, 3>;
using Vec2 = std::array<double, 2>;

}  // namespace skyshot
