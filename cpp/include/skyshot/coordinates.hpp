// Deterministic WCS mathematics (TAN/ARC special-cased arithmetic). Port of src/coordinates.rs.
#pragma once
#include "types.hpp"

namespace skyshot::coordinates {

Vec3 unit(double ra_deg, double dec_deg);
Vec2 pixel_to_sky(const Wcs& w, double x, double y);
std::optional<Vec2> sky_to_pixel(const Wcs& w, double ra, double dec);
std::optional<double> scale_arcsec(const Wcs& w, uint32_t width, uint32_t height);

}  // namespace skyshot::coordinates
