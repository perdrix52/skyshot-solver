// Field geometry derived from a solved WCS (centre, size, scale, orientation, corners, sexagesimal forms).
// Port of src/api/field.rs.
#pragma once
#include "json.hpp"
#include "types.hpp"

namespace skyshot::api {

struct FieldSummary {
  double ra_deg, dec_deg;
  std::string ra_hms, dec_dms;
  double galactic_l_deg, galactic_b_deg, width_deg, height_deg, scale_arcsec, up_pa_deg;
  bool mirrored;
  double crota2_deg, cdelt1_deg, cdelt2_deg;
  std::array<Vec2, 4> corners;
  std::optional<std::string> pole;
  uint32_t sip_order;
  Json to_json() const;
};

FieldSummary field_summary(const Wcs& wcs, uint32_t width, uint32_t height);
// Field size measured over `sky` (x0, y0, x1, y1: the open sky of a masked frame).
FieldSummary field_summary_in(const Wcs& wcs, uint32_t width, uint32_t height, const std::optional<std::array<double, 4>>& sky);
std::pair<double, double> galactic(double ra, double dec);
std::string format_hms(double ra_deg);
std::string format_dms(double dec_deg);

}  // namespace skyshot::api
