// Sky mask (ground / foreground estimate). Port of src/skymask.rs.
#pragma once
#include <memory>

#include "image_io.hpp"
#include "json.hpp"
#include "types.hpp"

namespace skyshot {

inline constexpr uint32_t SKYMASK_MAX_CELLS = 128 * 128;
inline constexpr double LANDSCAPE_FRACTION = 0.08;

struct SkyMask {
  uint32_t cols = 0, rows = 0;
  Vec2 cell{0, 0};
  Vec2 origin{0, 0};
  std::vector<bool> blocked;

  static SkyMask open(uint32_t width, uint32_t height, uint32_t cols, uint32_t rows);
  std::optional<std::string> validate() const;
  bool is_blocked(double x, double y) const;
  size_t n_blocked() const;
  bool is_landscape() const;
  double blocked_area(double x0, double y0, double x1, double y1) const;
  std::array<double, 4> open_bbox(double width, double height) const;
  double open_fraction(double x0, double y0, double x1, double y1) const;
  SkyMask shifted(double x0, double y0) const;
  SkyMask trimmed() const;

  Json to_json() const;
  static SkyMask from_json(const Json& j);

 private:
  std::optional<size_t> index(double x, double y) const;
};

namespace skymask {
// Estimate where stars are hidden; nullptr when nothing is hidden or the estimate is implausible.
std::shared_ptr<const SkyMask> estimate(const image_io::GrayImage& gray, const StarField& field);
}  // namespace skymask

}  // namespace skyshot
