#include "skyshot/coordinates.hpp"

#include "skyshot/geometry.hpp"
#include "skyshot/linalg.hpp"
#include "skyshot/sphere.hpp"

namespace skyshot::coordinates {

Vec3 unit(double ra_deg, double dec_deg) {
  double r = to_radians(ra_deg), d = to_radians(dec_deg);
  double cd = std::cos(d);
  return {cd * std::cos(r), cd * std::sin(r), std::sin(d)};
}

static std::optional<Vec2> tan_project(double ra, double dec, double ra0, double dec0) {
  double r = to_radians(ra), d = to_radians(dec), r0 = to_radians(ra0), d0 = to_radians(dec0);
  double dr = r - r0;
  double sd = std::sin(d), cd = std::cos(d), sd0 = std::sin(d0), cd0 = std::cos(d0), cdr = std::cos(dr);
  double den = sd0 * sd + cd0 * cd * cdr;
  if (den <= 0.0) return std::nullopt;
  return Vec2{to_degrees(cd * std::sin(dr) / den), to_degrees((cd0 * sd - sd0 * cd * cdr) / den)};
}

static Vec2 tan_unproject(double x, double y, double ra0, double dec0) {
  x = to_radians(x);
  y = to_radians(y);
  double r = to_radians(ra0), d = to_radians(dec0);
  double sd = std::sin(d), cd = std::cos(d);
  double den = cd - y * sd;
  return {rem_euclid(to_degrees(r + std::atan2(x, den)), 360.0), to_degrees(std::atan2(sd + y * cd, std::sqrt(den * den + x * x)))};
}

struct Basis3 {
  Vec3 c, e, n;
};
static Basis3 zenithal_basis(double ra0, double dec0) {
  double a = to_radians(ra0), d = to_radians(dec0);
  double sa = std::sin(a), ca = std::cos(a), sd = std::sin(d), cd = std::cos(d);
  return {{cd * ca, cd * sa, sd}, {-sa, ca, 0.0}, {-sd * ca, -sd * sa, cd}};
}

static std::optional<Vec2> arc_project(double ra, double dec, double ra0, double dec0) {
  Vec3 v = unit(ra, dec);
  Basis3 b = zenithal_basis(ra0, dec0);
  double ve = sphere::dot(b.e, v), vn = sphere::dot(b.n, v), vc = sphere::dot(b.c, v);
  double s = std::hypot(ve, vn);
  double rho = std::atan2(s, vc);
  if (rho > PI - 1e-6) return std::nullopt;
  double k = s < 1e-12 ? 1.0 : rho / s;
  return Vec2{to_degrees(ve * k), to_degrees(vn * k)};
}

static Vec2 arc_unproject(double x, double y, double ra0, double dec0) {
  double xr = to_radians(x), yr = to_radians(y);
  double rho = std::hypot(xr, yr);
  Basis3 b = zenithal_basis(ra0, dec0);
  double k = rho < 1e-12 ? 1.0 : std::sin(rho) / rho;
  double cr = std::cos(rho);
  Vec3 v;
  for (size_t i = 0; i < 3; ++i) v[i] = b.c[i] * cr + k * (xr * b.e[i] + yr * b.n[i]);
  return {rem_euclid(to_degrees(std::atan2(v[1], v[0])), 360.0), to_degrees(std::atan2(v[2], std::hypot(v[0], v[1])))};
}

static std::optional<Vec2> proj_project(const Wcs& w, double ra, double dec) {
  switch (w.proj) {
    case Proj::Tan: return tan_project(ra, dec, w.ra, w.dec);
    case Proj::Arc: return arc_project(ra, dec, w.ra, w.dec);
    default: return geometry::Projection(w.proj, w.native_pole).prepare({w.ra, w.dec}).project(ra, dec);
  }
}

Vec2 pixel_to_sky(const Wcs& w, double x, double y) {
  auto q = w.uncorrect_pixel({x, y});
  if (!q) return {NaN, NaN};
  double dx = (*q)[0] - w.crpix[0], dy = (*q)[1] - w.crpix[1];
  if (w.sip) {
    auto [fa, fb] = w.sip->eval(dx, dy);
    dx += fa;
    dy += fb;
  }
  double px = w.cd[0][0] * dx + w.cd[0][1] * dy, py = w.cd[1][0] * dx + w.cd[1][1] * dy;
  switch (w.proj) {
    case Proj::Tan: return tan_unproject(px, py, w.ra, w.dec);
    case Proj::Arc: return arc_unproject(px, py, w.ra, w.dec);
    default: return geometry::Projection(w.proj, w.native_pole).unproject(px, py, {w.ra, w.dec});
  }
}

std::optional<Vec2> sky_to_pixel(const Wcs& w, double ra, double dec) {
  auto p = proj_project(w, ra, dec);
  if (!p) return std::nullopt;
  double det = w.cd[0][0] * w.cd[1][1] - w.cd[0][1] * w.cd[1][0];
  if (std::fabs(det) < 1e-16) return std::nullopt;
  double tx = (w.cd[1][1] * (*p)[0] - w.cd[0][1] * (*p)[1]) / det;
  double ty = (-w.cd[1][0] * (*p)[0] + w.cd[0][0] * (*p)[1]) / det;
  double x = tx, y = ty;
  if (w.sip) {
    bool done = false;
    for (int it = 0; it < 20; ++it) {
      auto [fa, fb] = w.sip->eval(x, y);
      double ex = x + fa - tx, ey = y + fb - ty;
      auto j = w.sip->jacobian(x, y);
      double dj = j[0][0] * j[1][1] - j[0][1] * j[1][0];
      if (std::fabs(dj) < 1e-12) return std::nullopt;
      double ddx = (j[1][1] * ex - j[0][1] * ey) / dj;
      double ddy = (-j[1][0] * ex + j[0][0] * ey) / dj;
      x -= ddx;
      y -= ddy;
      if (std::sqrt(ddx * ddx + ddy * ddy) < 1e-7) {
        done = true;
        break;
      }
    }
    if (!done) return std::nullopt;
  }
  if (std::isfinite(x) && std::isfinite(y)) return w.correct_pixel({x + w.crpix[0], y + w.crpix[1]});
  return std::nullopt;
}

std::optional<double> scale_arcsec(const Wcs& w, uint32_t width, uint32_t height) {
  double cx = (double(width) - 1.0) / 2.0, cy = (double(height) - 1.0) / 2.0;
  Vec2 s0 = pixel_to_sky(w, cx, cy);
  auto step = [&](double dx, double dy) {
    Vec2 s = pixel_to_sky(w, cx + dx, cy + dy);
    return tan_project(s[0], s[1], s0[0], s0[1]).value_or(Vec2{0.0, 0.0});
  };
  Vec2 jx = step(1.0, 0.0), jy = step(0.0, 1.0);
  double sc = std::sqrt(std::fabs(jx[0] * jy[1] - jy[0] * jx[1])) * 3600.0;
  if (std::isfinite(sc) && sc > 0.0) return sc;
  return std::nullopt;
}

}  // namespace skyshot::coordinates
