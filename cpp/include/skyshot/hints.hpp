// Optional, untrusted astrometric hints. Port of src/hints.rs.
#pragma once
#include <map>

#include "common.hpp"
#include "json.hpp"

namespace skyshot {

struct SolveHint {
  std::optional<Vec2> position;
  std::optional<double> scale_arcsec;

  bool operator==(const SolveHint& o) const { return position == o.position && scale_arcsec == o.scale_arcsec; }
  static std::optional<SolveHint> merge(const std::optional<SolveHint>& primary, const std::optional<SolveHint>& fallback);
  std::optional<SolveHint> sanitized() const;
  std::optional<double> fov_deg(uint32_t width, uint32_t height) const;
  std::optional<SolveHint> for_frame(uint32_t width, uint32_t height) const;
};

inline constexpr double DEEP_FOV_DEG = 0.3;

using FitsHeader = std::map<std::string, std::string>;
std::optional<SolveHint> hint_from_fits(const FitsHeader& h, uint32_t width, uint32_t height);

}  // namespace skyshot
