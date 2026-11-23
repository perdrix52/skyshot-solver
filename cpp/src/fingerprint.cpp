#include "skyshot/fingerprint.hpp"

namespace skyshot::fingerprint {

std::optional<BasisCoords> basis_coords(P2 a, P2 r, const std::array<P2, 3>& nbrs) {
  double ux = r[0] - a[0], uy = r[1] - a[1];
  double len = std::hypot(ux, uy);
  if (len <= 0.0 || !std::isfinite(len)) return std::nullopt;
  double ex0 = ux / len, ex1 = uy / len;
  double ey0 = -ex1, ey1 = ex0;
  BasisCoords out;
  for (size_t k = 0; k < 3; ++k) {
    const P2& p = nbrs[k];
    double vx = p[0] - a[0], vy = p[1] - a[1];
    double x = (vx * ex0 + vy * ex1) / len;
    double y = (vx * ey0 + vy * ey1) / len;
    double h = std::hypot(x, y);
    if (h <= 0.0) return std::nullopt;
    out.c[k] = {std::log(h), std::atan2(y, x)};
  }
  out.len = len;
  return out;
}

std::optional<CanonicalKey> canonical_key(P2 a, P2 r, const std::array<P2, 3>& nbrs) {
  auto bc = basis_coords(a, r, nbrs);
  if (!bc) return std::nullopt;
  auto c = bc->c;
  std::stable_sort(c.begin(), c.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
  bool flip = c[0].second < 0.0;
  if (flip)
    for (auto& v : c) v.second = -v.second;
  uint32_t key = pack({rho_bin(c[0].first), rho_bin(c[1].first), rho_bin(c[2].first), t1_bin(c[0].second),
                       t_bin(c[1].second), t_bin(c[2].second)});
  return CanonicalKey{key, flip, bc->len};
}

static size_t push_bins_rho(double v, double tol, uint32_t out[3]) {
  uint32_t b = rho_bin(v);
  size_t n = 0;
  out[n++] = b;
  double lo = double(b) * RHO_STEP;
  double hi = double(b + 1) * RHO_STEP;
  if (v - lo < tol && b > 0) out[n++] = b - 1;
  if (hi - v < tol && b < RHO_BINS - 1) out[n++] = b + 1;
  return n;
}
static size_t push_bins_t1(double v, double tol, uint32_t out[3]) {
  uint32_t b = t1_bin(v);
  size_t n = 0;
  out[n++] = b;
  double lo = double(b) * THETA_STEP;
  double hi = double(b + 1) * THETA_STEP;
  if (v - lo < tol && b > 0) out[n++] = b - 1;
  if (hi - v < tol && b < THETA1_BINS - 1) out[n++] = b + 1;
  return n;
}
static size_t push_bins_t(double v, double tol, uint32_t out[3]) {
  v = rem_euclid(v, TAU);
  uint32_t b = t_bin(v);
  size_t n = 0;
  out[n++] = b;
  double lo = double(b) * THETA_STEP;
  double hi = double(b + 1) * THETA_STEP;
  if (v - lo < tol) out[n++] = (b + THETA_BINS - 1) % THETA_BINS;
  if (hi - v < tol) out[n++] = (b + 1) % THETA_BINS;
  return n;
}

static const size_t PERMS[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

std::optional<double> image_gate(P2 a, P2 r, const std::array<P2, 3>& nbrs, double sigma_px) {
  auto bc = basis_coords(a, r, nbrs);
  if (!bc) return std::nullopt;
  double tr = fmax_r(RHO_TOL_FLOOR, 2.8 * sigma_px / bc->len);
  for (auto& v : bc->c)
    if (v.first < -tr) return std::nullopt;
  return bc->len;
}

std::optional<double> image_keys(P2 a, P2 r, const std::array<P2, 3>& nbrs, double sigma_px,
                                 std::vector<KeyVariant>& out) {
  out.clear();
  auto bc = basis_coords(a, r, nbrs);
  if (!bc) return std::nullopt;
  const auto& c = bc->c;
  double len = bc->len;
  double tr = fmax_r(RHO_TOL_FLOOR, 2.8 * sigma_px / len);
  double tt = fmax_r(THETA_TOL_FLOOR, 1.6 * sigma_px / len);
  for (auto& v : c)
    if (v.first < -tr) return std::nullopt;
  for (const auto& p : PERMS) {
    std::pair<double, double> cs[3] = {c[p[0]], c[p[1]], c[p[2]]};
    if (cs[0].first > cs[1].first + 2.0 * tr || cs[0].first > cs[2].first + 2.0 * tr ||
        cs[1].first > cs[2].first + 2.0 * tr)
      continue;
    double tf = cs[0].second;
    bool ambiguous = std::fabs(tf) < tt || std::fabs(tf) > PI - tt;
    bool flips[2];
    size_t nflips;
    if (ambiguous) {
      flips[0] = false;
      flips[1] = true;
      nflips = 2;
    } else if (tf < 0.0) {
      flips[0] = true;
      nflips = 1;
    } else {
      flips[0] = false;
      nflips = 1;
    }
    for (size_t fi = 0; fi < nflips; ++fi) {
      bool flip = flips[fi];
      double s = flip ? -1.0 : 1.0;
      double t1 = clampd(cs[0].second * s, 0.0, PI);
      uint32_t b[6][3] = {};
      size_t n[6];
      n[0] = push_bins_rho(cs[0].first, tr, b[0]);
      n[1] = push_bins_rho(cs[1].first, tr, b[1]);
      n[2] = push_bins_rho(cs[2].first, tr, b[2]);
      n[3] = push_bins_t1(t1, tt, b[3]);
      n[4] = push_bins_t(cs[1].second * s, tt, b[4]);
      n[5] = push_bins_t(cs[2].second * s, tt, b[5]);
      for (size_t i0 = 0; i0 < n[0]; ++i0)
        for (size_t i1 = 0; i1 < n[1]; ++i1)
          for (size_t i2 = 0; i2 < n[2]; ++i2)
            for (size_t i3 = 0; i3 < n[3]; ++i3)
              for (size_t i4 = 0; i4 < n[4]; ++i4)
                for (size_t i5 = 0; i5 < n[5]; ++i5)
                  out.push_back(KeyVariant{pack({b[0][i0], b[1][i1], b[2][i2], b[3][i3], b[4][i4], b[5][i5]}), flip});
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  if (out.size() > MAX_VARIANTS) out.resize(MAX_VARIANTS);
  return len;
}

}  // namespace skyshot::fingerprint
