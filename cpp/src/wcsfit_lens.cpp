// A low-dimensional radial camera model. Port of src/wcsfit/lens.rs.
#include <algorithm>

#include "skyshot/linalg.hpp"
#include "skyshot/wcsfit.hpp"

namespace skyshot::wcsfit {

namespace {

std::optional<double> inverse_radius(double rd, const std::array<double, 2>& k) {
  if (rd == 0.0) return 0.0;
  double r = rd;
  for (int it = 0; it < 24; ++it) {
    double r2 = r * r;
    double f = r * (1.0 + k[0] * r2 + k[1] * r2 * r2) - rd;
    double d = 1.0 + 3.0 * k[0] * r2 + 5.0 * k[1] * r2 * r2;
    if (!std::isfinite(d) || d < 0.05 || !std::isfinite(r) || r < 0.0) return std::nullopt;
    if (std::fabs(f) < 1e-12) return r;
    r -= clampd(f / d, -0.2, 0.2);
  }
  return std::nullopt;
}

struct Evaluation {
  std::array<std::array<double, 3>, 2> coeff;
  std::vector<double> errors;
  double cost;
};

std::optional<Evaluation> evaluate(const std::vector<Vec2>& uv, const std::vector<Vec2>& q, const std::vector<double>& weights,
                                   const std::array<double, 2>& k) {
  std::vector<std::array<double, 3>> rows;
  rows.reserve(uv.size());
  double a[3][3] = {};
  double b[2][3] = {};
  for (size_t i = 0; i < uv.size() && i < q.size(); ++i) {
    const Vec2& p = uv[i];
    double rd = std::hypot(p[0], p[1]);
    double g;
    if (rd > 0.0) {
      auto ir = inverse_radius(rd, k);
      if (!ir) return std::nullopt;
      g = *ir / rd;
    } else {
      g = 1.0;
    }
    std::array<double, 3> row = {1.0, p[0] * g, p[1] * g};
    double w = i < weights.size() ? weights[i] : 1.0;
    if (!std::isfinite(w) || w <= 0.0) return std::nullopt;
    for (int r = 0; r < 3; ++r) {
      double wr = w * row[size_t(r)];
      for (int c = 0; c < 3; ++c) a[r][c] += wr * row[size_t(c)];
      b[0][r] += wr * q[i][0];
      b[1][r] += wr * q[i][1];
    }
    rows.push_back(row);
  }
  linalg::Mat am(3, 3);
  for (size_t r = 0; r < 3; ++r)
    for (size_t c = 0; c < 3; ++c) am(r, c) = a[r][c];
  auto chol = linalg::cholesky(am);
  if (!chol) return std::nullopt;
  Evaluation ev;
  for (int j = 0; j < 2; ++j) {
    auto x = linalg::cholesky_solve(*chol, std::vector<double>(b[j], b[j] + 3));
    ev.coeff[size_t(j)] = {x[0], x[1], x[2]};
  }
  ev.errors.reserve(2 * uv.size());
  for (size_t i = 0; i < rows.size(); ++i) {
    double w = std::sqrt(i < weights.size() ? weights[i] : 1.0);
    double d0 = rows[i][0] * ev.coeff[0][0] + rows[i][1] * ev.coeff[0][1] + rows[i][2] * ev.coeff[0][2];
    double d1 = rows[i][0] * ev.coeff[1][0] + rows[i][1] * ev.coeff[1][1] + rows[i][2] * ev.coeff[1][2];
    ev.errors.push_back(w * (d0 - q[i][0]));
    ev.errors.push_back(w * (d1 - q[i][1]));
  }
  double cost = -0.0;
  for (double v : ev.errors) cost += v * v;
  if (!std::isfinite(cost)) return std::nullopt;
  ev.cost = cost;
  return ev;
}

std::optional<Evaluation> optimize(const std::vector<Vec2>& uv, const std::vector<Vec2>& q, const std::vector<double>& weights,
                                   std::array<double, 2>& k) {
  auto cur = evaluate(uv, q, weights, k);
  if (!cur) return std::nullopt;
  Evaluation current = std::move(*cur);
  double lambda = 1e-3;
  for (int it = 0; it < 16; ++it) {
    std::vector<double> jac[2];
    for (int j = 0; j < 2; ++j) {
      auto perturbed = k;
      perturbed[size_t(j)] += 1e-6;
      auto shifted = evaluate(uv, q, weights, perturbed);
      if (!shifted) return std::nullopt;
      for (size_t t = 0; t < shifted->errors.size(); ++t) jac[j].push_back((shifted->errors[t] - current.errors[t]) / 1e-6);
    }
    auto dot = [](const std::vector<double>& a, const std::vector<double>& b) {
      double s = -0.0;
      for (size_t i = 0; i < a.size() && i < b.size(); ++i) s += a[i] * b[i];
      return s;
    };
    double h00 = dot(jac[0], jac[0]), h01 = dot(jac[0], jac[1]), h11 = dot(jac[1], jac[1]);
    double g[2] = {dot(jac[0], current.errors), dot(jac[1], current.errors)};
    std::optional<std::pair<std::array<double, 2>, Evaluation>> accepted;
    for (int inner = 0; inner < 8; ++inner) {
      double a = h00 * (1.0 + lambda) + 1e-20;
      double d = h11 * (1.0 + lambda) + 1e-20;
      double det = a * d - h01 * h01;
      if (!std::isfinite(det) || det <= 0.0) return std::nullopt;
      double step[2] = {(-d * g[0] + h01 * g[1]) / det, (h01 * g[0] - a * g[1]) / det};
      double norm = std::hypot(step[0], step[1]);
      if (norm < 1e-9) return current;
      if (norm > 0.2) {
        step[0] *= 0.2 / norm;
        step[1] *= 0.2 / norm;
      }
      std::array<double, 2> trial = {k[0] + step[0], k[1] + step[1]};
      auto next = evaluate(uv, q, weights, trial);
      if (next && next->cost < current.cost) {
        accepted = std::make_pair(trial, std::move(*next));
        lambda = fmax_r(lambda * 0.2, 1e-9);
        break;
      }
      lambda *= 10.0;
    }
    if (!accepted) break;
    double improvement = current.cost - accepted->second.cost;
    k = accepted->first;
    current = std::move(accepted->second);
    if (improvement <= 1e-10 * fmax_r(current.cost, 1e-20)) break;
  }
  return current;
}

std::optional<Wcs> fit_lens(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent, double radius) {
  if (pairs.size() < 40 || !std::isfinite(radius) || radius <= 0.0) return std::nullopt;
  std::vector<Vec2> uv;
  for (auto& p : pairs) uv.push_back({(p[0] - crpix[0]) / radius, (p[1] - crpix[1]) / radius});
  double mx = 0.0;
  for (auto& p : uv) mx = fmax_r(mx, std::hypot(p[0], p[1]));
  if (mx < 0.65) return std::nullopt;
  Vec2 tp = tangent;
  std::array<double, 2> k = {0.0, 0.0};
  std::optional<Evaluation> final_fit;
  for (int it = 0; it < 10; ++it) {
    geometry::PreparedProjection projection(Proj::Tan, tp);
    std::vector<Vec2> q;
    for (auto& p : pairs) {
      auto r = projection.project(p[2], p[3]);
      if (!r) return std::nullopt;
      q.push_back(*r);
    }
    auto fit = optimize(uv, q, weights, k);
    if (!fit) return std::nullopt;
    Vec2 off = {fit->coeff[0][0], fit->coeff[1][0]};
    if (fmax_r(std::fabs(off[0]), std::fabs(off[1])) < 1e-8) {
      final_fit = std::move(fit);
      break;
    }
    tp = geometry::unproject_p(Proj::Tan, off[0], off[1], tp);
  }
  if (!final_fit) return std::nullopt;
  const Evaluation& fit = *final_fit;
  auto edge_o = inverse_radius(1.0, k);
  if (!edge_o) return std::nullopt;
  double edge = *edge_o;
  auto derivative = [&](double r2) { return 1.0 + 3.0 * k[0] * r2 + 5.0 * k[1] * r2 * r2; };
  double critical = k[1] > 0.0 ? clampd(-3.0 * k[0] / (10.0 * k[1]), 0.0, edge * edge) : 0.0;
  if (fmin_r(fmin_r(derivative(0.0), derivative(edge * edge)), derivative(critical)) < 0.05) return std::nullopt;
  linalg::Mat a(4, 4);
  std::vector<double> b(4, 0.0);
  for (int i = 1; i <= 128; ++i) {
    double r = double(i) / 128.0;
    double row[4] = {powi(r, 3), powi(r, 5), powi(r, 7), powi(r, 9)};
    auto ir = inverse_radius(r, k);
    if (!ir) return std::nullopt;
    double y = *ir - r;
    for (size_t j = 0; j < 4; ++j) {
      b[j] += row[j] * y;
      for (size_t l = 0; l < 4; ++l) a(j, l) += row[j] * row[l];
    }
  }
  auto chol = linalg::cholesky(a);
  if (!chol) return std::nullopt;
  auto coeff = linalg::cholesky_solve(*chol, b);
  for (int i = 0; i <= 256; ++i) {
    double r = (double(i) + 0.5) / 257.0;
    double s = -0.0;
    for (int j = 0; j < 4; ++j) s += coeff[size_t(j)] * powi(r, 2 * j + 3);
    double approx = r + s;
    auto ir = inverse_radius(r, k);
    if (!ir) return std::nullopt;
    if (std::fabs(approx - *ir) * radius > 0.25) return std::nullopt;
  }
  auto terms = Sip::terms(9);
  Sip sip;
  sip.order = 9;
  sip.a.assign(terms.size(), 0.0);
  sip.b.assign(terms.size(), 0.0);
  auto find_term = [&](uint32_t p, uint32_t q2) -> std::optional<size_t> {
    for (size_t i = 0; i < terms.size(); ++i)
      if (terms[i].first == p && terms[i].second == q2) return i;
    return std::nullopt;
  };
  for (uint32_t j = 1; j <= 4; ++j) {
    double scale = coeff[j - 1] / powi(radius, int(2 * j));
    double binom = 1.0;
    for (uint32_t i = 0; i <= j; ++i) {
      auto ai = find_term(2 * i + 1, 2 * (j - i));
      auto bi = find_term(2 * i, 2 * (j - i) + 1);
      if (!ai || !bi) return std::nullopt;
      sip.a[*ai] += scale * binom;
      sip.b[*bi] += scale * binom;
      binom *= double(j - i) / double(i + 1);
    }
  }
  Wcs w;
  w.ra = tp[0];
  w.dec = tp[1];
  w.crpix = crpix;
  w.cd = {{{fit.coeff[0][1] / radius, fit.coeff[0][2] / radius}, {fit.coeff[1][1] / radius, fit.coeff[1][2] / radius}}};
  w.sip = std::move(sip);
  w.proj = Proj::Tan;
  return w;
}

}  // namespace

std::optional<Wcs> select_lens(const Pairs& pairs, const std::vector<double>& weights, const Wcs& reference,
                               std::pair<uint32_t, uint32_t> model, double width, double height) {
  if (!is_tan(reference.proj) || pairs.size() < 80 || model.second == 0) return std::nullopt;
  double radius = 0.5 * std::hypot(width, height);
  size_t counts[16] = {};
  Pairs train, test;
  std::vector<double> train_w, test_w;
  for (size_t i = 0; i < pairs.size(); ++i) {
    const Pair& p = pairs[i];
    size_t x = sat_cast<size_t>(clampd(std::floor(4.0 * p[0] / width), 0.0, 3.0));
    size_t y = sat_cast<size_t>(clampd(std::floor(4.0 * p[1] / height), 0.0, 3.0));
    size_t cell = 4 * y + x;
    double wt = i < weights.size() ? weights[i] : 1.0;
    if (counts[cell] % 4 == 0) {
      test.push_back(p);
      test_w.push_back(wt);
    } else {
      train.push_back(p);
      train_w.push_back(wt);
    }
    counts[cell] += 1;
  }
  if (test.size() < 16) return std::nullopt;
  Vec2 tangent = {reference.ra, reference.dec};
  auto lens = fit_lens(train, train_w, reference.crpix, tangent, radius);
  if (!lens) return std::nullopt;
  auto base = robust_fit_wmp(train, train_w, reference.crpix, tangent, model.first, model.second,
                             Projection(reference.proj, reference.native_pole));
  if (!base) return std::nullopt;
  auto le = residuals(*lens, test);
  auto be = residuals(base->wcs, test);
  double lr = robust_test_rms(le, test_w);
  double br = robust_test_rms(be, test_w);
  if (!std::isfinite(lr) || !std::isfinite(br) || lr > 1.05 * br) return std::nullopt;
  double n = 2.0 * double(test.size());
  auto score = [&](double e, double k) { return n * std::log(fmax_r(e * e, 1e-20)) + k * std::log(n); };
  if (score(lr, 8.0) >= score(br, 2.0 * double(model_columns(model.first, model.second)))) return std::nullopt;
  for (size_t cell = 0; cell < 16; ++cell) {
    double a = 0.0, b = 0.0;
    int count = 0;
    for (size_t i = 0; i < test.size(); ++i) {
      const Pair& p = test[i];
      double l = le[i], r = be[i];
      size_t x = sat_cast<size_t>(clampd(std::floor(4.0 * p[0] / width), 0.0, 3.0));
      size_t y = sat_cast<size_t>(clampd(std::floor(4.0 * p[1] / height), 0.0, 3.0));
      if (!std::isfinite(l) || !std::isfinite(r)) return std::nullopt;
      if (cell == 4 * y + x) {
        a += l * l;
        b += r * r;
        count += 1;
      }
    }
    if (count > 0 && std::sqrt(a / double(count)) > 1.25 * std::sqrt(b / double(count)) + 0.1) return std::nullopt;
  }
  return fit_lens(pairs, weights, reference.crpix, tangent, radius);
}

}  // namespace skyshot::wcsfit
