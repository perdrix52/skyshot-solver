// Stereographic fitting with a free optical centre. Port of src/wcsfit/stereographic.rs.
#include <algorithm>

#include "skyshot/linalg.hpp"
#include "skyshot/sphere.hpp"
#include "skyshot/wcsfit.hpp"

namespace skyshot::wcsfit {

using sphere::normalize;
using sphere::radec;
using sphere::tangent_basis;

namespace {

struct GeometricFit {
  Wcs wcs;
  std::vector<std::array<double, 2>> residuals;
  double score;
};

using M2 = linalg::M2;

M2 outer_acc(M2 m, double w, double a0, double a1, double b0, double b1) {
  double wa0 = w * a0, wa1 = w * a1;
  m[0][0] += wa0 * b0;
  m[0][1] += wa0 * b1;
  m[1][0] += wa1 * b0;
  m[1][1] += wa1 * b1;
  return m;
}

std::optional<GeometricFit> at(const Pairs& pairs, const std::vector<double>& weights, const Vec3& center) {
  auto [ra, dec] = radec(center);
  geometry::PreparedProjection proj(Proj::Stg, {ra, dec});
  std::vector<Vec2> q;
  q.reserve(pairs.size());
  for (auto& p : pairs) {
    auto r = proj.project(p[2], p[3]);
    if (!r) return std::nullopt;
    q.push_back(*r);
  }
  double sw = 0.0;
  double qm[2] = {0, 0}, pm[2] = {0, 0};
  for (size_t i = 0; i < pairs.size(); ++i) {
    double w = i < weights.size() ? weights[i] : 1.0;
    sw += w;
    for (int j = 0; j < 2; ++j) {
      qm[j] += w * q[i][size_t(j)];
      pm[j] += w * pairs[i][size_t(j)];
    }
  }
  if (!(std::isfinite(sw) && sw > 0.0)) return std::nullopt;
  for (int j = 0; j < 2; ++j) {
    qm[j] /= sw;
    pm[j] /= sw;
  }
  M2 cov{}, cross{};
  for (size_t i = 0; i < pairs.size(); ++i) {
    double w = i < weights.size() ? weights[i] : 1.0;
    double a0 = q[i][0] - qm[0], a1 = q[i][1] - qm[1];
    double b0 = pairs[i][0] - pm[0], b1 = pairs[i][1] - pm[1];
    cov = outer_acc(cov, w, a0, a1, a0, a1);
    cross = outer_acc(cross, w, b0, b1, a0, a1);
  }
  auto cinv = linalg::inverse2(cov);
  if (!cinv) return std::nullopt;
  M2 a;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) a[size_t(i)][size_t(j)] = cross[size_t(i)][0] * (*cinv)[0][size_t(j)] + cross[size_t(i)][1] * (*cinv)[1][size_t(j)];
  auto cd = linalg::inverse2(a);
  if (!cd) return std::nullopt;
  double cp0 = pm[0] - (a[0][0] * qm[0] + a[0][1] * qm[1]);
  double cp1 = pm[1] - (a[1][0] * qm[0] + a[1][1] * qm[1]);
  GeometricFit gf;
  double score = 0.0;
  for (size_t i = 0; i < pairs.size(); ++i) {
    double pred0 = (a[0][0] * q[i][0] + a[0][1] * q[i][1]) + cp0;
    double pred1 = (a[1][0] * q[i][0] + a[1][1] * q[i][1]) + cp1;
    std::array<double, 2> r = {pred0 - pairs[i][0], pred1 - pairs[i][1]};
    score += (i < weights.size() ? weights[i] : 1.0) * (r[0] * r[0] + r[1] * r[1]);
    gf.residuals.push_back(r);
  }
  if (!std::isfinite(score)) return std::nullopt;
  gf.wcs.ra = ra;
  gf.wcs.dec = dec;
  gf.wcs.crpix = {cp0, cp1};
  gf.wcs.cd = {{{(*cd)[0][0], (*cd)[0][1]}, {(*cd)[1][0], (*cd)[1][1]}}};
  gf.wcs.proj = Proj::Stg;
  gf.score = score;
  return gf;
}

std::optional<GeometricFit> geometric_fit(const Pairs& pairs, const std::vector<double>& weights, Vec3 c) {
  auto cur0 = at(pairs, weights, c);
  if (!cur0) return std::nullopt;
  GeometricFit current = std::move(*cur0);
  double damping = 1e-3;
  for (int it = 0; it < 60; ++it) {
    auto [e, n] = tangent_basis(c);
    double eps = 1e-4;
    Vec3 ce = normalize({c[0] + eps * e[0], c[1] + eps * e[1], c[2] + eps * e[2]});
    Vec3 cn = normalize({c[0] + eps * n[0], c[1] + eps * n[1], c[2] + eps * n[2]});
    auto er = at(pairs, weights, ce);
    if (!er) return std::nullopt;
    auto nr = at(pairs, weights, cn);
    if (!nr) return std::nullopt;
    M2 h{};
    double g0 = 0.0, g1 = 0.0;
    for (size_t i = 0; i < pairs.size(); ++i) {
      double w = i < weights.size() ? weights[i] : 1.0;
      for (size_t axis = 0; axis < 2; ++axis) {
        double j0 = (er->residuals[i][axis] - current.residuals[i][axis]) / eps;
        double j1 = (nr->residuals[i][axis] - current.residuals[i][axis]) / eps;
        h = outer_acc(h, w, j0, j1, j0, j1);
        g0 += (w * j0) * current.residuals[i][axis];
        g1 += (w * j1) * current.residuals[i][axis];
      }
    }
    bool progressed = false;
    for (int inner = 0; inner < 12; ++inner) {
      double h00 = h[0][0] + damping * fmax_r(h[0][0], 1e-10);
      double h01 = h[0][1];
      double h10 = h[1][0];
      double h11 = h[1][1] + damping * fmax_r(h[1][1], 1e-10);
      // LU with partial pivoting solve of hd * step = -g
      double b0 = -g0, b1 = -g1;
      double s0, s1;
      if (std::fabs(h10) > std::fabs(h00)) {
        std::swap(h00, h10);
        std::swap(h01, h11);
        std::swap(b0, b1);
      }
      if (h00 == 0.0) return std::nullopt;
      double l = h10 / h00;
      double u11 = h11 - l * h01;
      double y1 = b1 - l * b0;
      if (u11 == 0.0) return std::nullopt;
      s1 = y1 / u11;
      s0 = (b0 - h01 * s1) / h00;
      double norm = std::sqrt(s0 * s0 + s1 * s1);
      if (norm > 0.3) {
        s0 *= 0.3 / norm;
        s1 *= 0.3 / norm;
      }
      if (std::sqrt(s0 * s0 + s1 * s1) < 1e-9) return current;
      Vec3 next = normalize({c[0] + s0 * e[0] + s1 * n[0], c[1] + s0 * e[1] + s1 * n[1], c[2] + s0 * e[2] + s1 * n[2]});
      auto cand = at(pairs, weights, next);
      if (cand && cand->score < current.score) {
        double improvement = current.score - cand->score;
        c = next;
        current = std::move(*cand);
        damping = fmax_r(damping * 0.3, 1e-9);
        progressed = true;
        if (improvement < 1e-10) return current;
        break;
      }
      damping *= 10.0;
    }
    if (!progressed) break;
  }
  return current;
}

std::optional<Wcs> fit(const Pairs& pairs, const std::vector<double>& weights) {
  if (pairs.size() < 24) return std::nullopt;
  for (auto& p : pairs)
    for (double v : p)
      if (!std::isfinite(v)) return std::nullopt;
  double sx = 0, sy = 0;
  for (auto& p : pairs) sx += p[0];
  for (auto& p : pairs) sy += p[1];
  Vec2 cp = {sx / double(pairs.size()), sy / double(pairs.size())};
  double scale = 1.0;
  for (auto& p : pairs) scale = fmax_r(scale, fmax_r(std::fabs(p[0] - cp[0]), std::fabs(p[1] - cp[1])));
  linalg::Mat m(2 * pairs.size(), 11);
  std::vector<double> y(2 * pairs.size());
  for (size_t i = 0; i < pairs.size(); ++i) {
    Vec3 v = geometry::unit(pairs[i][2], pairs[i][3]);
    double x = (pairs[i][0] - cp[0]) / scale;
    double yy = (pairs[i][1] - cp[1]) / scale;
    for (size_t j = 0; j < 3; ++j) {
      m(2 * i, j) = v[j];
      m(2 * i + 1, j + 4) = v[j];
      m(2 * i, j + 8) = -x * v[j];
      m(2 * i + 1, j + 8) = -yy * v[j];
    }
    m(2 * i, 3) = 1.0;
    m(2 * i + 1, 7) = 1.0;
    y[2 * i] = x;
    y[2 * i + 1] = yy;
  }
  auto svd = linalg::svd(m);
  double largest = svd.s[0];
  if (svd.s[10] < largest * 1e-11) return std::nullopt;
  auto p = linalg::svd_solve(svd, y, largest * 1e-12);
  if (!p) return std::nullopt;
  Vec3 c = normalize({(*p)[8], (*p)[9], (*p)[10]});
  for (double x : c)
    if (!std::isfinite(x)) return std::nullopt;
  Vec3 sum{0, 0, 0};
  for (auto& pr : pairs) {
    Vec3 v = geometry::unit(pr[2], pr[3]);
    for (size_t k = 0; k < 3; ++k) sum[k] += v[k];
  }
  Vec3 mean = normalize(sum);
  auto a = geometric_fit(pairs, weights, c);
  auto b = geometric_fit(pairs, weights, mean);
  if (a && b) return b->score < a->score ? b->wcs : a->wcs;  // min_by keeps the first on ties
  if (a) return a->wcs;
  if (b) return b->wcs;
  return std::nullopt;
}

}  // namespace

std::optional<RobustFit> fit_stereographic(const Pairs& pairs, const std::vector<double>& weights) {
  if (!weights.empty()) {
    if (weights.size() != pairs.size()) return std::nullopt;
    for (double w : weights)
      if (!std::isfinite(w) || w <= 0.0) return std::nullopt;
  }
  std::vector<bool> keep(pairs.size(), true);
  for (int pass = 0; pass <= 4; ++pass) {
    Pairs sub;
    std::vector<double> w;
    for (size_t i = 0; i < pairs.size(); ++i)
      if (keep[i]) {
        sub.push_back(pairs[i]);
        w.push_back(i < weights.size() ? weights[i] : 1.0);
      }
    auto fitted = fit(sub, w);
    if (!fitted) return std::nullopt;
    auto errors = residuals(*fitted, pairs);
    std::vector<double> finite;
    for (double e : errors)
      if (std::isfinite(e)) finite.push_back(e);
    std::stable_sort(finite.begin(), finite.end(), [](double a, double b) { return total_cmp(a, b) < 0; });
    if (finite.size() / 2 >= finite.size()) return std::nullopt;
    double median = finite[finite.size() / 2];
    double threshold = fmax_r(3.0 * 1.4826 * median, 1.0);
    std::vector<bool> next(errors.size());
    for (size_t i = 0; i < errors.size(); ++i) next[i] = std::isfinite(errors[i]) && errors[i] <= threshold;
    if (next == keep || pass == 4) return RobustFit{std::move(*fitted), keep};
    keep = std::move(next);
  }
  return std::nullopt;
}

}  // namespace skyshot::wcsfit
