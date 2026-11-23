// Atmospheric refraction measured from a solved wide field. Port of src/solve/refract.rs.
#include <algorithm>

#include "solve_internal.hpp"

namespace skyshot::solve::refract {

using sphere::dot;
using sphere::normalize;
using sphere::radec;
using sphere::tangent_basis;

static double saemundsson(double h_deg) {
  double h = fmax_r(h_deg, -1.0);
  return to_radians(1.02 / std::tan(to_radians(h + 10.3 / (h + 5.11))) / 60.0);
}

static Vec3 refract_dir(const Vec3& t, const Vec3& z, double scale) {
  double c = clampd(dot(t, z), -1.0, 1.0);
  double h = 90.0 - to_degrees(std::acos(c));
  double r = scale * saemundsson(h);
  Vec3 d = {z[0] - t[0] * c, z[1] - t[1] * c, z[2] - t[2] * c};
  double n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (n < 1e-12) return t;
  double s = std::sin(r), co = std::cos(r);
  return normalize({t[0] * co + d[0] / n * s, t[1] * co + d[1] / n * s, t[2] * co + d[2] / n * s});
}

static Vec3 offset_dir(const Vec3& c, double east, double north) {
  auto [e, n] = tangent_basis(c);
  return normalize({c[0] + east * e[0] + north * n[0], c[1] + east * e[1] + north * n[1], c[2] + east * e[2] + north * n[2]});
}

namespace {
struct Problem {
  const Pairs* pairs;
  std::vector<Vec3> t;
  std::vector<double> weights;
  Vec2 crpix, tangent;
  uint32_t order, radial;
  Proj proj;

  double rms(const Vec3& z, double k) const {
    Pairs refr;
    refr.reserve(pairs->size());
    for (size_t i = 0; i < pairs->size(); ++i) {
      const auto& p = (*pairs)[i];
      Vec3 a = k > 0.0 ? refract_dir(t[i], z, k) : t[i];
      auto [ra, dec] = radec(a);
      refr.push_back({p[0], p[1], ra, dec});
    }
    auto fit = wcsfit::fit_model(refr, weights, crpix, tangent, order, radial, geometry::Projection(proj));
    if (!fit) return INF;
    return wcsfit::rms(wcsfit::residuals(*fit, refr));
  }
  double min_alt(const Vec3& z) const {
    double m = INF;
    for (const Vec3& tt : t) m = fmin_r(m, 90.0 - to_degrees(std::acos(clampd(dot(tt, z), -1.0, 1.0))));
    return m;
  }
};

constexpr size_t MIN_PAIRS = 60;
constexpr size_t MAX_PAIRS = 250;
constexpr double MAX_MIN_ALT = 20.0;
constexpr double MAX_RIGID_RMS = 1.0;
constexpr double SCALE_LO = 0.5, SCALE_HI = 1.6;

using V3 = std::array<double, 3>;

template <class F>
V3 nelder_mead(F&& f, V3 x0, V3 step, size_t iters) {
  std::vector<std::pair<V3, double>> s;
  for (int i = 0; i < 4; ++i) {
    V3 x = x0;
    if (i > 0) x[size_t(i - 1)] += step[size_t(i - 1)];
    s.emplace_back(x, f(x));
  }
  auto comb = [](const V3& a, const V3& b, double t) { return V3{a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])}; };
  auto sort_s = [&] {
    std::stable_sort(s.begin(), s.end(), [](const auto& a, const auto& b) { return total_cmp(a.second, b.second) < 0; });
  };
  for (size_t it = 0; it < iters; ++it) {
    sort_s();
    V3 c = {(s[0].first[0] + s[1].first[0] + s[2].first[0]) / 3.0, (s[0].first[1] + s[1].first[1] + s[2].first[1]) / 3.0,
            (s[0].first[2] + s[1].first[2] + s[2].first[2]) / 3.0};
    auto worst = s[3];
    V3 xr = comb(c, worst.first, -1.0);
    double fr = f(xr);
    if (fr < s[0].second) {
      V3 xe = comb(c, worst.first, -2.0);
      double fe = f(xe);
      s[3] = fe < fr ? std::make_pair(xe, fe) : std::make_pair(xr, fr);
    } else if (fr < s[2].second) {
      s[3] = {xr, fr};
    } else {
      V3 xc = comb(c, worst.first, 0.5);
      double fc = f(xc);
      if (fc < worst.second) {
        s[3] = {xc, fc};
      } else {
        V3 b = s[0].first;
        for (size_t i = 1; i < s.size(); ++i) {
          s[i].first = comb(b, s[i].first, 0.5);
          s[i].second = f(s[i].first);
        }
      }
    }
    double mx = std::numeric_limits<double>::lowest();
    for (auto& v : s) mx = fmax_r(mx, v.second);
    double spread = mx - s[0].second;
    if (std::fabs(spread) < 1e-6) break;
  }
  sort_s();
  return s[0].first;
}
}  // namespace

std::optional<Refraction> measure(const Wcs& wcs, const Pairs& pairs, double w, double h) {
  if (is_cylindrical(wcs.proj) || pairs.size() < MIN_PAIRS) return std::nullopt;
  size_t step = (pairs.size() + MAX_PAIRS - 1) / MAX_PAIRS;
  Pairs sub;
  for (size_t i = 0; i < pairs.size(); i += step) sub.push_back(pairs[i]);
  Vec2 crpix = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  size_t n = sub.size();
  Problem p;
  p.pairs = &sub;
  for (auto& q : sub) p.t.push_back(sphere::unit(q[2], q[3]));
  p.crpix = crpix;
  p.tangent = wcs.pixel_to_sky(crpix[0], crpix[1]);
  p.order = 1;
  p.radial = n >= 120 ? 7 : 5;
  p.proj = wcs.proj;
  double rms0 = p.rms({0.0, 0.0, 1.0}, 0.0);
  if (!std::isfinite(rms0)) return std::nullopt;
  Vec3 c0 = sphere::unit(p.tangent[0], p.tangent[1]);
  int n_grid = 1200;
  double golden = PI * (3.0 - std::sqrt(5.0));
  double best_e = INF;
  Vec3 best_z = c0;
  int tried = 0;
  for (int i = 0; i < n_grid; ++i) {
    double zc = 1.0 - 2.0 * (double(i) + 0.5) / double(n_grid);
    double r = std::sqrt(1.0 - zc * zc);
    Vec3 z = {r * std::cos(golden * double(i)), r * std::sin(golden * double(i)), zc};
    if (dot(z, c0) < std::sin(to_radians(-10.0))) continue;
    double lo = p.min_alt(z);
    if (!(lo >= -1.5 && lo <= MAX_MIN_ALT)) continue;
    ++tried;
    double e = p.rms(z, 1.0);
    if (e < best_e) {
      best_e = e;
      best_z = z;
    }
  }
  if (tried == 0 || !std::isfinite(best_e)) return std::nullopt;
  Vec3 z0 = best_z;
  auto f = [&](V3 x) -> double {
    Vec3 z = offset_dir(z0, x[0], x[1]);
    double k = std::exp(x[2]);
    if (!(k >= 0.2 && k <= 4.0) || p.min_alt(z) < -1.5) return INF;
    return p.rms(z, k);
  };
  V3 x = nelder_mead(f, {0.0, 0.0, 0.0}, {0.035, 0.035, 0.25}, 120);
  Vec3 z = offset_dir(z0, x[0], x[1]);
  double k = std::exp(x[2]);
  double rms1 = f(x);
  double lo = p.min_alt(z);
  bool supported = std::isfinite(rms1) && rms1 < 0.85 * rms0 && rms1 <= MAX_RIGID_RMS && lo >= -1.0 && lo <= MAX_MIN_ALT &&
                   k >= SCALE_LO && k <= SCALE_HI;
  if (!supported) return std::nullopt;
  auto [ra, dec] = radec(z);
  Refraction r;
  r.zenith = {ra, dec};
  r.scale = k;
  r.rms_px = rms0;
  r.rms_refracted_px = rms1;
  r.min_alt_deg = lo;
  r.n_pairs = n;
  return r;
}

}  // namespace skyshot::solve::refract
