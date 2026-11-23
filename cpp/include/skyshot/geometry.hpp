// Projections and WCS arithmetic. Port of src/geometry.rs.
#pragma once
#include "types.hpp"

namespace skyshot::geometry {

Vec3 unit(double ra, double dec);
double separation(const Vec3& a, const Vec3& b);
std::optional<Vec2> project(double ra, double dec, Vec2 center);
Vec2 unproject(double x, double y, Vec2 center);
std::optional<Vec2> project_arc(double ra, double dec, Vec2 center);
Vec2 unproject_arc(double x, double y, Vec2 center);

struct Basis {
  Vec3 c{}, e{}, n{};
};

class PreparedProjection {
 public:
  Proj proj;
  Vec2 center;
  double ra0, sin_dec0, cos_dec0;
  Basis basis;
  double latitude0;

  PreparedProjection(Proj proj, Vec2 center, std::optional<Vec2> pole = std::nullopt);
  std::optional<Vec2> project(double ra, double dec) const;
  Vec2 unproject(double x, double y) const;
};

struct Projection {
  Proj proj = Proj::Tan;
  std::optional<Vec2> native_pole;
  Projection() = default;
  Projection(Proj p) : proj(p) {}
  Projection(Proj p, std::optional<Vec2> pole) : proj(p), native_pole(pole) {}
  Projection fixed(Vec2 center) const;
  PreparedProjection prepare(Vec2 center) const { return PreparedProjection(proj, center, native_pole); }
  Vec2 unproject(double x, double y, Vec2 center) const { return prepare(center).unproject(x, y); }
};

std::optional<Vec2> project_p(Proj proj, double ra, double dec, Vec2 center);
Vec2 unproject_p(Proj proj, double x, double y, Vec2 center);
Vec2 vector_sky(const Vec3& v);
Basis panorama_basis(Vec2 center);
Vec3 panorama_unit(double ra, double dec);

class PreparedWcs {
 public:
  const Wcs* wcs;
  PreparedProjection projection;
  double det;
  std::optional<Vec2> sky_to_pixel(double ra, double dec) const;
  std::optional<Vec2> sky_to_pixel_fast(double ra, double dec) const;
};
PreparedWcs prepare(const Wcs& w);

CatalogStar at_epoch(CatalogStar s, std::optional<double> epoch);
// Least-squares TAN from correspondences (x, y, ra, dec).
std::optional<Wcs> fit_wcs(const std::vector<std::array<double, 4>>& pairs, Vec2 center);
Vec2 healpix_center(uint32_t nside, uint64_t pix);

}  // namespace skyshot::geometry
