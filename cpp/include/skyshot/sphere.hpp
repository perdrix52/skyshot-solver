// Spherical utilities: unit vectors, equal-angle cube-face grid with CSR cell lists.
// Port of src/sphere.rs.
#pragma once
#include <cassert>

#include "common.hpp"

namespace skyshot::sphere {

inline Vec3 unit(double ra_deg, double dec_deg) {
  double r = to_radians(ra_deg), d = to_radians(dec_deg);
  double cd = std::cos(d);
  return {cd * std::cos(r), cd * std::sin(r), std::sin(d)};
}
inline std::pair<double, double> radec(const Vec3& v) {
  double ra = rem_euclid(to_degrees(std::atan2(v[1], v[0])), 360.0);
  double dec = to_degrees(std::atan2(v[2], std::hypot(v[0], v[1])));
  return {ra, dec};
}
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) {
  double n = norm(a);
  return {a[0] / n, a[1] / n, a[2] / n};
}
inline double angle(const Vec3& a, const Vec3& b) { return std::atan2(norm(cross(a, b)), dot(a, b)); }
inline double chord(double ang) { return 2.0 * std::sin(ang / 2.0); }

inline std::pair<Vec3, Vec3> tangent_basis(const Vec3& c) {
  Vec3 e{-c[1], c[0], 0.0};
  double n = norm(e);
  Vec3 east = n < 1e-12 ? Vec3{0.0, 1.0, 0.0} : Vec3{e[0] / n, e[1] / n, 0.0};
  Vec3 north = cross(c, east);
  return {east, north};
}
inline std::optional<Vec2> gnomonic(const Vec3& p, const Vec3& c, const Vec3& east, const Vec3& north) {
  double d = dot(p, c);
  if (d <= 1e-12) return std::nullopt;
  return Vec2{dot(p, east) / d, dot(p, north) / d};
}
inline Vec3 ungnomonic(const Vec2& xy, const Vec3& c, const Vec3& east, const Vec3& north) {
  return normalize({c[0] + xy[0] * east[0] + xy[1] * north[0], c[1] + xy[0] * east[1] + xy[1] * north[1],
                    c[2] + xy[0] * east[2] + xy[1] * north[2]});
}
inline std::optional<Vec2> stereographic(const Vec3& p, const Vec3& c, const Vec3& east, const Vec3& north) {
  double d = 1.0 + dot(p, c);
  if (d <= 1e-12) return std::nullopt;
  return Vec2{2.0 * dot(p, east) / d, 2.0 * dot(p, north) / d};
}
inline Vec3 unstereographic(const Vec2& xy, const Vec3& c, const Vec3& east, const Vec3& north) {
  double t2 = (xy[0] * xy[0] + xy[1] * xy[1]) / 4.0;
  double a = 1.0 - t2;
  double s = 1.0 / (1.0 + t2);
  return {(a * c[0] + xy[0] * east[0] + xy[1] * north[0]) * s, (a * c[1] + xy[0] * east[1] + xy[1] * north[1]) * s,
          (a * c[2] + xy[0] * east[2] + xy[1] * north[2]) * s};
}

struct FaceCoords {
  uint32_t face;
  double alpha, beta;
};
inline FaceCoords face_coords(const Vec3& v) {
  double ax = std::fabs(v[0]), ay = std::fabs(v[1]), az = std::fabs(v[2]);
  uint32_t face;
  double u, w;
  if (ax >= ay && ax >= az) {
    if (v[0] > 0.0) { face = 0; u = v[1] / ax; w = v[2] / ax; }
    else { face = 1; u = -v[1] / ax; w = v[2] / ax; }
  } else if (ay >= az) {
    if (v[1] > 0.0) { face = 2; u = -v[0] / ay; w = v[2] / ay; }
    else { face = 3; u = v[0] / ay; w = v[2] / ay; }
  } else if (v[2] > 0.0) {
    face = 4; u = v[1] / az; w = -v[0] / az;
  } else {
    face = 5; u = v[1] / az; w = v[0] / az;
  }
  return {face, std::atan(u), std::atan(w)};
}
inline Vec3 face_vector(uint32_t face, double alpha, double beta) {
  double u = std::tan(alpha), w = std::tan(beta);
  Vec3 v;
  switch (face) {
    case 0: v = {1.0, u, w}; break;
    case 1: v = {-1.0, -u, w}; break;
    case 2: v = {-u, 1.0, w}; break;
    case 3: v = {u, -1.0, w}; break;
    case 4: v = {-w, u, 1.0}; break;
    default: v = {w, u, -1.0}; break;
  }
  return normalize(v);
}

struct CubeGrid {
  uint32_t n = 1;
  CubeGrid() = default;
  explicit CubeGrid(uint32_t n_) : n(n_) {
    if (!(n >= 1 && uint64_t(n) * n * 6 < uint64_t(UINT32_MAX))) throw Error("invalid cube grid size");
  }
  static CubeGrid with_min_width(double min_width) {
    uint32_t n = 1;
    while (n < 16384 && CubeGrid(n * 2).min_cell_width() >= min_width) n *= 2;
    return CubeGrid(n);
  }
  size_t cells() const { return 6 * size_t(n) * size_t(n); }
  double min_cell_width() const { return 0.70 * (FRAC_PI_2 / double(n)); }
  double max_cell_diameter() const { return 1.5 * (FRAC_PI_2 / double(n)); }
  uint32_t cell(const Vec3& v) const {
    auto f = face_coords(v);
    return cell_of(f.face, f.alpha, f.beta);
  }
  uint32_t cell_of(uint32_t face, double alpha, double beta) const {
    double nn = double(n);
    double i = std::floor(((alpha + FRAC_PI_4) / FRAC_PI_2) * nn);
    double j = std::floor(((beta + FRAC_PI_4) / FRAC_PI_2) * nn);
    uint32_t ii = std::min(sat_cast<uint32_t>(fmax_r(i, 0.0)), n - 1);
    uint32_t jj = std::min(sat_cast<uint32_t>(fmax_r(j, 0.0)), n - 1);
    return face * n * n + jj * n + ii;
  }
  // (face, i, j)
  std::array<uint32_t, 3> split(uint32_t cell) const {
    uint32_t nn = n * n;
    return {cell / nn, cell % n, (cell % nn) / n};
  }
  Vec3 center(uint32_t cell) const {
    auto s = split(cell);
    double step = FRAC_PI_2 / double(n);
    return face_vector(s[0], -FRAC_PI_4 + (double(s[1]) + 0.5) * step, -FRAC_PI_4 + (double(s[2]) + 0.5) * step);
  }
  void cap_cells(const Vec3& c, double radius, std::vector<uint32_t>& out) const {
    out.clear();
    double step = FRAC_PI_2 / double(n);
    auto fc = face_coords(c);
    int64_t reach = sat_cast<int64_t>(std::ceil(radius / (0.70 * step))) + 1;
    int64_t ci = sat_cast<int64_t>(std::floor((fc.alpha + FRAC_PI_4) / step));
    int64_t cj = sat_cast<int64_t>(std::floor((fc.beta + FRAC_PI_4) / step));
    int64_t nn = n;
    if (ci - reach >= 0 && cj - reach >= 0 && ci + reach < nn && cj + reach < nn && radius < 0.3) {
      for (int64_t j = cj - reach; j <= cj + reach; ++j)
        for (int64_t i = ci - reach; i <= ci + reach; ++i)
          out.push_back(fc.face * n * n + uint32_t(j) * n + uint32_t(i));
      return;
    }
    double r = std::min(radius + max_cell_diameter(), FRAC_PI_2 * 0.98);
    double h = min_cell_width() / SQRT_2;
    auto [east, north] = tangent_basis(c);
    double extent = std::tan(r);
    int64_t k = sat_cast<int64_t>(std::ceil(extent / std::min(std::tan(h), h)));
    k = std::min<int64_t>(k, 4000);
    double dh = extent / double(std::max<int64_t>(k, 1));
    for (int64_t y = -k; y <= k; ++y) {
      for (int64_t x = -k; x <= k; ++x) {
        double px = double(x) * dh, py = double(y) * dh;
        if (px * px + py * py > extent * extent * 1.0001) continue;
        out.push_back(cell(ungnomonic({px, py}, c, east, north)));
      }
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
  }
};

struct CellIndex {
  CubeGrid grid;
  std::vector<uint32_t> offsets;
  std::vector<uint32_t> items;
  // `order` lists point ids in the order items should appear inside each cell.
  template <class It>
  static CellIndex build(CubeGrid grid, const std::vector<uint32_t>& cells, It order_begin, It order_end) {
    size_t ncell = grid.cells();
    std::vector<uint32_t> counts(ncell + 1, 0);
    for (uint32_t c : cells) counts[size_t(c) + 1] += 1;
    for (size_t i = 0; i < ncell; ++i) counts[i + 1] += counts[i];
    CellIndex ci;
    ci.grid = grid;
    ci.offsets = counts;
    std::vector<uint32_t> fill = counts;
    ci.items.assign(cells.size(), 0);
    for (It it = order_begin; it != order_end; ++it) {
      uint32_t p = *it;
      size_t c = cells[p];
      ci.items[fill[c]] = p;
      fill[c] += 1;
    }
    return ci;
  }
  const uint32_t* cell_begin(uint32_t cell) const { return items.data() + offsets[cell]; }
  const uint32_t* cell_end(uint32_t cell) const { return items.data() + offsets[size_t(cell) + 1]; }
  size_t cell_len(uint32_t cell) const { return offsets[size_t(cell) + 1] - offsets[cell]; }
};

}  // namespace skyshot::sphere
