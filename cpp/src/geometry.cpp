#include "skyshot/geometry.hpp"

#include <cstdio>

#include "skyshot/linalg.hpp"

namespace skyshot::geometry {

Vec3 unit(double ra, double dec) {
  double r = to_radians(ra), d = to_radians(dec);
  return {std::cos(d) * std::cos(r), std::cos(d) * std::sin(r), std::sin(d)};
}
double separation(const Vec3& a, const Vec3& b) {
  Vec3 cross = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
  double cs = 0;
  for (double x : cross) cs += x * x;
  double d = 0;
  for (int i = 0; i < 3; ++i) d += a[size_t(i)] * b[size_t(i)];
  return std::atan2(std::sqrt(cs), d);
}
std::optional<Vec2> project(double ra, double dec, Vec2 center) {
  double r = to_radians(ra), d = to_radians(dec), r0 = to_radians(center[0]), d0 = to_radians(center[1]);
  double dr = r - r0;
  double den = std::sin(d0) * std::sin(d) + std::cos(d0) * std::cos(d) * std::cos(dr);
  if (den <= 0.0) return std::nullopt;
  return Vec2{to_degrees(std::cos(d) * std::sin(dr) / den),
              to_degrees((std::cos(d0) * std::sin(d) - std::sin(d0) * std::cos(d) * std::cos(dr)) / den)};
}
Vec2 unproject(double x, double y, Vec2 center) {
  x = to_radians(x);
  y = to_radians(y);
  double r = to_radians(center[0]), d = to_radians(center[1]);
  double den = std::cos(d) - y * std::sin(d);
  return {rem_euclid(to_degrees(r + std::atan2(x, den)), 360.0),
          to_degrees(std::atan2(std::sin(d) + y * std::cos(d), std::sqrt(den * den + x * x)))};
}

static Basis zenithal_basis(Vec2 center) {
  double sa = std::sin(to_radians(center[0])), ca = std::cos(to_radians(center[0]));
  double sd = std::sin(to_radians(center[1])), cd = std::cos(to_radians(center[1]));
  return {{cd * ca, cd * sa, sd}, {-sa, ca, 0.0}, {-sd * ca, -sd * sa, cd}};
}

std::optional<Vec2> project_arc(double ra, double dec, Vec2 center) {
  Vec3 v = unit(ra, dec);
  Basis b = zenithal_basis(center);
  auto dot = [&](const Vec3& a) { return a[0] * v[0] + a[1] * v[1] + a[2] * v[2]; };
  double ve = dot(b.e), vn = dot(b.n), vc = dot(b.c);
  double s = std::hypot(ve, vn);
  double rho = std::atan2(s, vc);
  if (rho > PI - 1e-6) return std::nullopt;
  double k = s < 1e-12 ? 1.0 : rho / s;
  return Vec2{to_degrees(ve * k), to_degrees(vn * k)};
}

Vec2 unproject_arc(double x, double y, Vec2 center) {
  double xr = to_radians(x), yr = to_radians(y);
  double rho = std::hypot(xr, yr);
  Basis b = zenithal_basis(center);
  double k = rho < 1e-12 ? 1.0 : std::sin(rho) / rho;
  double cr = std::cos(rho);
  Vec3 v;
  for (size_t i = 0; i < 3; ++i) v[i] = b.c[i] * cr + k * (xr * b.e[i] + yr * b.n[i]);
  return {rem_euclid(to_degrees(std::atan2(v[1], v[0])), 360.0), to_degrees(std::atan2(v[2], std::hypot(v[0], v[1])))};
}

Vec3 panorama_unit(double ra, double dec) {
  double a = to_radians(ra), d = to_radians(dec);
  return {std::cos(d) * std::cos(a), std::cos(d) * std::sin(a), std::sin(d)};
}
Basis panorama_basis(Vec2 center) {
  double a = to_radians(center[0]), d = to_radians(center[1]);
  double sa = std::sin(a), ca = std::cos(a), sd = std::sin(d), cd = std::cos(d);
  return {{cd * ca, cd * sa, sd}, {-sa, ca, 0.}, {-sd * ca, -sd * sa, cd}};
}
Vec2 vector_sky(const Vec3& v) {
  return {rem_euclid(to_degrees(std::atan2(v[1], v[0])), 360.0), to_degrees(std::atan2(v[2], std::hypot(v[0], v[1])))};
}

static double cylinder_y(Proj proj, double lat) {
  switch (proj) {
    case Proj::Car: return lat;
    case Proj::Mer: return std::asinh(std::tan(lat));
    default: return std::tan(lat);
  }
}

Projection Projection::fixed(Vec2 center) const {
  Projection p = *this;
  if (is_cylindrical(p.proj) && !p.native_pole) {
    Basis b = panorama_basis(center);
    p.native_pole = vector_sky(b.n);
  }
  return p;
}

std::optional<Vec2> project_p(Proj proj, double ra, double dec, Vec2 center) {
  return PreparedProjection(proj, center).project(ra, dec);
}
Vec2 unproject_p(Proj proj, double x, double y, Vec2 center) { return PreparedProjection(proj, center).unproject(x, y); }

PreparedProjection::PreparedProjection(Proj proj_, Vec2 center_, std::optional<Vec2> pole)
    : proj(proj_), center(center_), latitude0(0.0) {
  double d0 = to_radians(center[1]);
  if (is_tan(proj)) basis = Basis{};
  else if (proj == Proj::Arc) basis = zenithal_basis(center);
  else basis = panorama_basis(center);
  if (is_cylindrical(proj) && pole) {
    Vec3 c = panorama_unit(center[0], center[1]);
    Vec3 n = panorama_unit((*pole)[0], (*pole)[1]);
    double dot = 0;
    for (int i = 0; i < 3; ++i) dot += c[size_t(i)] * n[size_t(i)];
    latitude0 = std::asin(clampd(dot, -1.0, 1.0));
    Vec3 a;
    for (size_t i = 0; i < 3; ++i) a[i] = c[i] - dot * n[i];
    double nn = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    for (auto& x : a) x = x / nn;
    Vec3 e = {n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0]};
    basis = Basis{a, e, n};
  }
  ra0 = to_radians(center[0]);
  sin_dec0 = std::sin(d0);
  cos_dec0 = std::cos(d0);
}

std::optional<Vec2> PreparedProjection::project(double ra, double dec) const {
  if (is_tan(proj)) {
    double r = to_radians(ra), d = to_radians(dec);
    double dr = r - ra0;
    double den = sin_dec0 * std::sin(d) + cos_dec0 * std::cos(d) * std::cos(dr);
    if (den <= 0.0) return std::nullopt;
    return Vec2{to_degrees(std::cos(d) * std::sin(dr) / den),
                to_degrees((cos_dec0 * std::sin(d) - sin_dec0 * std::cos(d) * std::cos(dr)) / den)};
  }
  Vec3 v = proj == Proj::Arc ? unit(ra, dec) : panorama_unit(ra, dec);
  auto dot = [&](const Vec3& a) { return a[0] * v[0] + a[1] * v[1] + a[2] * v[2]; };
  double ve = dot(basis.e), vn = dot(basis.n), vc = dot(basis.c);
  Vec2 xy;
  if (is_cylindrical(proj)) {
    if (std::hypot(ve, vc) < 1e-10) return std::nullopt;
    double lat = std::atan2(vn, std::hypot(ve, vc));
    xy = {to_degrees(std::atan2(ve, vc)), to_degrees(cylinder_y(proj, lat) - cylinder_y(proj, latitude0))};
  } else {
    double s = std::hypot(ve, vn);
    double rho = std::atan2(s, vc);
    if (rho > PI - 1e-6) return std::nullopt;
    double k;
    if (proj == Proj::Stg) k = 2.0 / (1.0 + vc);
    else if (s < 1e-12) k = 1.0;
    else k = rho / s;
    xy = {to_degrees(ve * k), to_degrees(vn * k)};
  }
  if (std::isfinite(xy[0]) && std::isfinite(xy[1])) return xy;
  return std::nullopt;
}

Vec2 PreparedProjection::unproject(double x, double y) const {
  if (is_tan(proj)) return geometry::unproject(x, y, center);
  if (proj == Proj::Arc) return unproject_arc(x, y, center);
  x = to_radians(x);
  y = to_radians(y);
  Vec3 v;
  if (is_cylindrical(proj)) {
    double yy = y + cylinder_y(proj, latitude0);
    double lat;
    switch (proj) {
      case Proj::Car: lat = yy; break;
      case Proj::Mer: lat = std::asin(std::tanh(yy)); break;
      default: lat = std::atan(yy); break;
    }
    if (std::fabs(x) > PI + 1e-12 || std::fabs(lat) > PI / 2.0 + 1e-12) return {NaN, NaN};
    for (size_t i = 0; i < 3; ++i)
      v[i] = std::cos(lat) * (std::cos(x) * basis.c[i] + std::sin(x) * basis.e[i]) + std::sin(lat) * basis.n[i];
  } else {
    double r = std::hypot(x, y);
    double rho = 2.0 * std::atan(r / 2.0);
    double k = r < 1e-12 ? 1.0 : std::sin(rho) / r;
    for (size_t i = 0; i < 3; ++i) v[i] = std::cos(rho) * basis.c[i] + k * (x * basis.e[i] + y * basis.n[i]);
  }
  return vector_sky(v);
}

PreparedWcs prepare(const Wcs& w) {
  Projection p(w.proj, w.native_pole);
  return PreparedWcs{&w, p.prepare({w.ra, w.dec}), w.determinant()};
}

std::optional<Vec2> PreparedWcs::sky_to_pixel(double ra, double dec) const {
  auto p = projection.project(ra, dec);
  if (!p) return std::nullopt;
  if (std::fabs(det) < 1e-16) return std::nullopt;
  const auto& cd = wcs->cd;
  double tx = (cd[1][1] * (*p)[0] - cd[0][1] * (*p)[1]) / det;
  double ty = (-cd[1][0] * (*p)[0] + cd[0][0] * (*p)[1]) / det;
  double x = tx, y = ty;
  if (wcs->sip) {
    for (int it = 0; it < 20; ++it) {
      auto ej = wcs->sip->eval_jacobian(x, y);
      double ex = x + ej.fa - tx, ey = y + ej.fb - ty;
      auto inv = linalg::inverse2({{{ej.j[0][0], ej.j[0][1]}, {ej.j[1][0], ej.j[1][1]}}});
      if (!inv) return std::nullopt;
      double dx = (*inv)[0][0] * ex + (*inv)[0][1] * ey;
      double dy = (*inv)[1][0] * ex + (*inv)[1][1] * ey;
      x -= dx;
      y -= dy;
      if (std::sqrt(dx * dx + dy * dy) < 1e-7) break;
    }
  }
  if (std::isfinite(x) && std::isfinite(y)) return wcs->correct_pixel({x + wcs->crpix[0], y + wcs->crpix[1]});
  return std::nullopt;
}

std::optional<Vec2> PreparedWcs::sky_to_pixel_fast(double ra, double dec) const {
  auto p = projection.project(ra, dec);
  if (!p) return std::nullopt;
  if (std::fabs(det) < 1e-16) return std::nullopt;
  const auto& cd = wcs->cd;
  double tx = (cd[1][1] * (*p)[0] - cd[0][1] * (*p)[1]) / det;
  double ty = (-cd[1][0] * (*p)[0] + cd[0][0] * (*p)[1]) / det;
  double x = tx, y = ty;
  if (wcs->sip) {
    for (int it = 0; it < 8; ++it) {
      auto ej = wcs->sip->eval_jacobian(x, y);
      double ex = x + ej.fa - tx, ey = y + ej.fb - ty;
      double d = ej.j[0][0] * ej.j[1][1] - ej.j[0][1] * ej.j[1][0];
      if (std::fabs(d) < 1e-12) return std::nullopt;
      double dx = (ej.j[1][1] * ex - ej.j[0][1] * ey) / d;
      double dy = (-ej.j[1][0] * ex + ej.j[0][0] * ey) / d;
      x -= dx;
      y -= dy;
      if (std::fabs(dx) + std::fabs(dy) < 1e-3) break;
    }
  }
  if (std::isfinite(x) && std::isfinite(y)) return wcs->correct_pixel({x + wcs->crpix[0], y + wcs->crpix[1]});
  return std::nullopt;
}

CatalogStar at_epoch(CatalogStar s, std::optional<double> epoch) {
  if (epoch) {
    double dt = *epoch - double(s.epoch);
    Vec3 u = unit(s.ra, s.dec);
    double r = to_radians(s.ra), d = to_radians(s.dec);
    Vec3 e = {-std::sin(r), std::cos(r), 0.0};
    Vec3 n = {-std::cos(r) * std::sin(d), -std::sin(r) * std::sin(d), std::cos(d)};
    double k = PI / (180.0 * 3600000.0);
    Vec3 v;
    for (size_t i = 0; i < 3; ++i) v[i] = u[i] + dt * k * (double(s.pmra) * e[i] + double(s.pmdec) * n[i]);
    s.ra = rem_euclid(to_degrees(std::atan2(v[1], v[0])), 360.0);
    s.dec = to_degrees(std::atan2(v[2], std::hypot(v[0], v[1])));
  }
  return s;
}

std::optional<Wcs> fit_wcs(const std::vector<std::array<double, 4>>& pairs, Vec2 center) {
  if (pairs.size() < 3) return std::nullopt;
  double sx = 0, sy = 0;
  for (auto& p : pairs) sx += p[0];
  for (auto& p : pairs) sy += p[1];
  double mx = sx / double(pairs.size()), my = sy / double(pairs.size());
  double scale = 1.0;
  for (auto& p : pairs) scale = fmax_r(scale, std::hypot(p[0] - mx, p[1] - my));
  linalg::Mat a(pairs.size(), 3);
  std::vector<double> u(pairs.size()), v(pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i) {
    auto q = project(pairs[i][2], pairs[i][3], center);
    if (!q) return std::nullopt;
    a(i, 0) = (pairs[i][0] - mx) / scale;
    a(i, 1) = (pairs[i][1] - my) / scale;
    a(i, 2) = 1.0;
    u[i] = (*q)[0];
    v[i] = (*q)[1];
  }
  auto svd = linalg::svd(a);
  auto x = linalg::svd_solve(svd, u, 1e-10);
  if (!x) return std::nullopt;
  auto y = linalg::svd_solve(svd, v, 1e-10);
  if (!y) return std::nullopt;
  std::array<std::array<double, 2>, 2> cd = {{{(*x)[0] / scale, (*x)[1] / scale}, {(*y)[0] / scale, (*y)[1] / scale}}};
  auto inv = linalg::inverse2(cd);
  if (!inv) return std::nullopt;
  double ox = (*inv)[0][0] * (*x)[2] + (*inv)[0][1] * (*y)[2];
  double oy = (*inv)[1][0] * (*x)[2] + (*inv)[1][1] * (*y)[2];
  Wcs w;
  w.ra = center[0];
  w.dec = center[1];
  w.crpix = {mx - ox, my - oy};
  w.cd = cd;
  w.proj = Proj::Tan;
  return w;
}

Vec2 healpix_center(uint32_t nside, uint64_t pix) {
  int64_t n = int64_t(nside), p = int64_t(pix);
  int64_t np = 12 * n * n, cap = 2 * n * (n - 1);
  double z, phi;
  if (p < cap) {
    int64_t ring = sat_cast<int64_t>(std::floor((1.0 + std::sqrt(1.0 + 2.0 * double(p))) / 2.0));
    int64_t ip = p + 1 - 2 * ring * (ring - 1);
    z = 1.0 - double(ring * ring) / double(3 * n * n);
    phi = (double(ip) - 0.5) * PI / double(2 * ring);
  } else if (p < np - cap) {
    int64_t ip = p - cap;
    int64_t ring = ip / (4 * n) + n;
    int64_t az = ip % (4 * n) + 1;
    double f = 0.5 * double(1 + ((ring + n) & 1));
    z = double(2 * n - ring) * 2.0 / double(3 * n);
    phi = (double(az) - f) * PI / double(2 * n);
  } else {
    int64_t ip = np - p;
    int64_t ring = sat_cast<int64_t>(std::floor((1.0 + std::sqrt(2.0 * double(ip) - 1.0)) / 2.0));
    int64_t az = 4 * ring + 1 - (ip - 2 * ring * (ring - 1));
    z = -1.0 + double(ring * ring) / double(3 * n * n);
    phi = (double(az) - 0.5) * PI / double(2 * ring);
  }
  return {rem_euclid(to_degrees(phi), 360.0), to_degrees(std::asin(z))};
}

}  // namespace skyshot::geometry

namespace skyshot {

using geometry::PreparedProjection;
using geometry::Projection;

Vec2 Wcs::correct_pixel(Vec2 p) const { return local_grid ? local_grid->forward(p) : p; }
std::optional<Vec2> Wcs::uncorrect_pixel(Vec2 p) const {
  if (!local_grid) return p;
  return local_grid->inverse(p);
}
double Wcs::width_deg(uint32_t width, uint32_t height) const {
  double y = (double(height) - 1.0) / 2.0;
  return arc_deg({-0.5, y}, {double(width) - 0.5, y});
}
double Wcs::arc_deg(Vec2 a, Vec2 b) const {
  constexpr int STEPS = 64;
  auto at = [&](double t) {
    Vec2 s = pixel_to_sky(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]));
    return geometry::unit(s[0], s[1]);
  };
  Vec3 prev = at(0.0);
  double total = 0.0;
  for (int k = 1; k <= STEPS; ++k) {
    Vec3 p = at(double(k) / double(STEPS));
    total += geometry::separation(prev, p);
    prev = p;
  }
  return to_degrees(total);
}
double Wcs::determinant() const { return cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0]; }
Vec2 Wcs::pixel_to_sky(double x, double y) const {
  auto q = uncorrect_pixel({x, y});
  if (!q) return {NaN, NaN};
  double dx = (*q)[0] - crpix[0], dy = (*q)[1] - crpix[1];
  if (sip) {
    auto [fa, fb] = sip->eval(dx, dy);
    dx += fa;
    dy += fb;
  }
  return Projection(proj, native_pole).unproject(cd[0][0] * dx + cd[0][1] * dy, cd[1][0] * dx + cd[1][1] * dy, {ra, dec});
}
std::optional<Vec2> Wcs::sky_to_pixel_fast(double ra_, double dec_) const {
  auto p = Projection(proj, native_pole).prepare({ra, dec}).project(ra_, dec_);
  if (!p) return std::nullopt;
  double det = determinant();
  if (std::fabs(det) < 1e-16) return std::nullopt;
  double tx = (cd[1][1] * (*p)[0] - cd[0][1] * (*p)[1]) / det;
  double ty = (-cd[1][0] * (*p)[0] + cd[0][0] * (*p)[1]) / det;
  double x = tx, y = ty;
  if (sip) {
    for (int it = 0; it < 8; ++it) {
      auto [fa, fb] = sip->eval(x, y);
      double ex = x + fa - tx, ey = y + fb - ty;
      auto jj = sip->jacobian(x, y);
      double d = jj[0][0] * jj[1][1] - jj[0][1] * jj[1][0];
      if (std::fabs(d) < 1e-12) return std::nullopt;
      double dx = (jj[1][1] * ex - jj[0][1] * ey) / d;
      double dy = (-jj[1][0] * ex + jj[0][0] * ey) / d;
      x -= dx;
      y -= dy;
      if (std::fabs(dx) + std::fabs(dy) < 1e-3) break;
    }
  }
  if (std::isfinite(x) && std::isfinite(y)) return correct_pixel({x + crpix[0], y + crpix[1]});
  return std::nullopt;
}
std::optional<Vec2> Wcs::sky_to_pixel(double ra_, double dec_) const {
  auto p = Projection(proj, native_pole).prepare({ra, dec}).project(ra_, dec_);
  if (!p) return std::nullopt;
  double det = determinant();
  if (std::fabs(det) < 1e-16) return std::nullopt;
  double tx = (cd[1][1] * (*p)[0] - cd[0][1] * (*p)[1]) / det;
  double ty = (-cd[1][0] * (*p)[0] + cd[0][0] * (*p)[1]) / det;
  double x = tx, y = ty;
  if (sip) {
    for (int it = 0; it < 20; ++it) {
      auto [fa, fb] = sip->eval(x, y);
      double ex = x + fa - tx, ey = y + fb - ty;
      auto jj = sip->jacobian(x, y);
      auto inv = linalg::inverse2({{{jj[0][0], jj[0][1]}, {jj[1][0], jj[1][1]}}});
      if (!inv) return std::nullopt;
      double dx = (*inv)[0][0] * ex + (*inv)[0][1] * ey;
      double dy = (*inv)[1][0] * ex + (*inv)[1][1] * ey;
      x -= dx;
      y -= dy;
      if (std::sqrt(dx * dx + dy * dy) < 1e-7) break;
    }
  }
  if (std::isfinite(x) && std::isfinite(y)) return correct_pixel({x + crpix[0], y + crpix[1]});
  return std::nullopt;
}
double Wcs::scale_arcsec() const {
  double metric = 1.0;
  if (is_cylindrical(proj)) {
    auto p = Projection(proj, native_pole).prepare({ra, dec});
    double c = std::cos(p.latitude0);
    metric = proj == Proj::Car ? c : proj == Proj::Mer ? c * c : c * c * c;
  }
  return std::sqrt(std::fabs(determinant()) * metric) * 3600.0;
}
double Wcs::rotation_deg() const {
  if (!is_cylindrical(proj)) return rem_euclid(to_degrees(std::atan2(cd[0][1], cd[1][1])), 360.0);
  auto p = Projection(proj, native_pole).prepare({ra, dec});
  const auto& a = p.basis.c;
  const auto& e = p.basis.e;
  const auto& n = p.basis.n;
  double sn = std::sin(p.latitude0), cs = std::cos(p.latitude0);
  double dy = proj == Proj::Car ? 1.0 : proj == Proj::Mer ? cs : cs * cs;
  Vec3 v;
  for (size_t i = 0; i < 3; ++i) v[i] = cs * e[i] * cd[0][1] + dy * (-sn * a[i] + cs * n[i]) * cd[1][1];
  auto pb = geometry::panorama_basis({ra, dec});
  auto dot = [&](const Vec3& q) { return q[0] * v[0] + q[1] * v[1] + q[2] * v[2]; };
  return rem_euclid(to_degrees(std::atan2(dot(pb.e), dot(pb.n))), 360.0);
}

static std::string fmt_e12(double v) {
  // Rust `{:20.12E}`: upper-case exponent without sign padding or leading zeros ("1.000000000000E0").
  if (std::isnan(v)) return "NaN";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.12E", v);
  std::string s(buf);
  size_t e = s.find('E');
  std::string mant = s.substr(0, e);
  int ex = std::atoi(s.c_str() + e + 1);
  return mant + "E" + std::to_string(ex);
}
static std::string pad_left(const std::string& s, size_t w) { return s.size() >= w ? s : std::string(w - s.size(), ' ') + s; }
static std::string pad_right(const std::string& s, size_t w) { return s.size() >= w ? s : s + std::string(w - s.size(), ' '); }

std::string Wcs::fits_header() const {
  std::vector<std::string> cards = {"SIMPLE  =                    T", "BITPIX  =                    8",
                                    "NAXIS   =                    0", "EXTEND  =                    T"};
  if (local_grid) {
    cards.push_back("COMMENT Base projection/SIP only; local pixel grid is in the JSON result.");
    cards.push_back("COMMENT Do not use this header for precise astrometry near stitched edges.");
  }
  std::string suf = sip ? "-SIP" : "";
  std::string code = proj_code(proj);
  cards.push_back("CTYPE1  = 'RA---" + code + suf + "'");
  cards.push_back("CTYPE2  = 'DEC--" + code + suf + "'");
  auto num_card = [&](const std::string& k, double v) { return pad_right(k, 8) + "= " + pad_left(fmt_e12(v), 20); };
  if (is_cylindrical(proj)) {
    auto p = Projection(proj, native_pole).prepare({ra, dec});
    const auto& a = p.basis.c;
    const auto& e = p.basis.e;
    const auto& n = p.basis.n;
    cards.push_back(num_card("PV1_0", 1.0));
    cards.push_back(num_card("PV1_1", 0.0));
    cards.push_back(num_card("PV1_2", to_degrees(p.latitude0)));
    cards.push_back(num_card("LONPOLE", to_degrees(std::atan2(e[2], a[2]))));
    cards.push_back(num_card("LATPOLE", to_degrees(std::asin(clampd(n[2], -1.0, 1.0)))));
    if (proj == Proj::Cyp) {
      cards.push_back("PV2_1   =                  0.0");
      cards.push_back("PV2_2   =                  1.0");
    }
  }
  cards.push_back("RADESYS = 'ICRS'");
  cards.push_back("CUNIT1  = 'deg'");
  cards.push_back("CUNIT2  = 'deg'");
  cards.push_back(num_card("CRVAL1", ra));
  cards.push_back(num_card("CRVAL2", dec));
  cards.push_back(num_card("CRPIX1", crpix[0] + 1.0));
  cards.push_back(num_card("CRPIX2", crpix[1] + 1.0));
  cards.push_back(num_card("CD1_1", cd[0][0]));
  cards.push_back(num_card("CD1_2", cd[0][1]));
  cards.push_back(num_card("CD2_1", cd[1][0]));
  cards.push_back(num_card("CD2_2", cd[1][1]));
  if (sip) {
    cards.push_back("A_ORDER = " + pad_left(std::to_string(sip->order), 20));
    cards.push_back("B_ORDER = " + pad_left(std::to_string(sip->order), 20));
    auto terms = Sip::terms(sip->order);
    for (size_t i = 0; i < terms.size(); ++i) {
      std::string pq = std::to_string(terms[i].first) + "_" + std::to_string(terms[i].second);
      cards.push_back(num_card("A_" + pq, sip->a[i]));
      cards.push_back(num_card("B_" + pq, sip->b[i]));
    }
  }
  cards.push_back("END");
  std::string out;
  for (auto& c : cards) out += pad_right(c, 80);
  size_t n = (out.size() + 2879) / 2880 * 2880;
  out.append(n - out.size(), ' ');
  return out;
}

}  // namespace skyshot
