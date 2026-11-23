// Four-star pattern codes. Port of src/patterns.rs.
#pragma once
#include <array>
#include <optional>
#include <set>

#include "types.hpp"

namespace skyshot::patterns {

struct Pattern {
  std::array<size_t, 4> ids;
  std::array<double, 4> code;
  double baseline, quality;
  std::array<double, 6> features;
};

double distance(const Star& a, const Star& b);
std::vector<Pattern> variants(const std::vector<Star>& stars, std::array<size_t, 4> ids);
std::optional<Pattern> canonical(const std::vector<Star>& stars, std::array<size_t, 4> ids);
std::vector<size_t> balanced(const std::vector<Star>& stars, uint32_t width, uint32_t height);
std::vector<Pattern> enumerate(const std::vector<Star>& stars, uint32_t width, uint32_t height);

// Lazily deepening pattern stream (tiers of 12, 24, 48 and 96 stars).
class Emitter {
 public:
  Emitter(const std::vector<Star>& stars, uint32_t width, uint32_t height);
  std::optional<Pattern> next();
  size_t tier() const { return tier_; }
  const std::vector<size_t>& ranked() const { return ranked_; }

 private:
  void expand(size_t tier);
  const std::vector<Star>* stars_;
  std::vector<size_t> ranked_;
  std::set<std::array<size_t, 4>> seen_;
  size_t tier_ = 0;
  std::vector<Pattern> pending_;
  size_t pending_pos_ = 0;
};

inline constexpr size_t BINS = size_t(1) << 20;
size_t bin(const std::array<double, 4>& code);
std::array<uint16_t, 4> quantize(const std::array<double, 4>& code);

}  // namespace skyshot::patterns
