#include "skyshot/wcsfit.hpp"

#include <algorithm>

#include "skyshot/linalg.hpp"

namespace skyshot::wcsfit {

static std::vector<uint32_t> radial_degrees(uint32_t order, uint32_t radial) {
  std::vector<uint32_t> out;
  for (uint32_t d = order + 1; d <= radial; ++d)
    if (d % 2 == 1) out.push_back(d);
  return out;
}

size_t model_columns(uint32_t order, uint32_t radial) {
  return 3 + Sip::terms(order).size() + 2 * radial_degrees(order, radial).size();
}

std::optional<Wcs> fit_tan_sip(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent, uint32_t order) {
  return fit_tan_sip_model(pairs, weights, crpix, tangent, order, 0);
}
std::optional<Wcs> fit_tan_sip_model(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                     uint32_t order, uint32_t radial) {
  return fit_model(pairs, weights, crpix, tangent, order, radial, Projection(Proj::Tan));
}

namespace {
struct PolynomialFit {
  double scale;
  Vec2 tangent;
  std::vector<double> x, y;
};

std::optional<PolynomialFit> fit_polynomial(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                            uint32_t order, uint32_t radial, const Projection& proj_in) {
  Projection proj = proj_in.fixed(tangent);
  auto terms = Sip::terms(order);
  auto rdeg = radial_degrees(order, radial);
  size_t ncol = 3 + terms.size() + 2 * rdeg.size();
  if (pairs.size() < ncol + 3) return std::nullopt;
  double scale = 1.0;
  for (auto& p : pairs) scale = fmax_r(scale, fmax_r(std::fabs(p[0] - crpix[0]), std::fabs(p[1] - crpix[1])));
  std::vector<double> row(ncol, 0.0);
  std::vector<double> weighted_rows;
  weighted_rows.reserve(pairs.size() * ncol);
  linalg::Mat ata(ncol, ncol);
  for (size_t i = 0; i < pairs.size(); ++i) {
    const auto& p = pairs[i];
    double w = i < weights.size() ? weights[i] : 1.0;
    double u = (p[0] - crpix[0]) / scale;
    double v = (p[1] - crpix[1]) / scale;
    row[0] = 1.0;
    row[1] = u;
    row[2] = v;
    double pu[16], pv[16];
    for (int k = 0; k < 16; ++k) pu[k] = pv[k] = 1.0;
    for (size_t k = 1; k <= std::min<size_t>(order, 15); ++k) {
      pu[k] = pu[k - 1] * u;
      pv[k] = pv[k - 1] * v;
    }
    for (size_t k = 0; k < terms.size(); ++k) row[3 + k] = pu[terms[k].first] * pv[terms[k].second];
    double rho2 = u * u + v * v;
    for (size_t k = 0; k < rdeg.size(); ++k) {
      double f = powi(rho2, int((rdeg[k] - 1) / 2));
      row[3 + terms.size() + 2 * k] = u * f;
      row[3 + terms.size() + 2 * k + 1] = v * f;
    }
    for (size_t r = 0; r < ncol; ++r) {
      double wr = w * row[r];
      weighted_rows.push_back(wr);
      for (size_t c = r; c < ncol; ++c) ata(r, c) += wr * row[c];
    }
  }
  for (size_t r = 0; r < ncol; ++r)
    for (size_t c = 0; c < r; ++c) ata(r, c) = ata(c, r);
  auto chol = linalg::cholesky(ata);
  if (!chol) return std::nullopt;
  Vec2 tp = tangent;
  std::optional<std::pair<std::vector<double>, std::vector<double>>> sol;
  for (int it = 0; it < 3; ++it) {
    auto projection = proj.prepare(tp);
    std::vector<double> atx(ncol, 0.0), aty(ncol, 0.0);
    for (size_t i = 0; i < pairs.size(); ++i) {
      auto q = projection.project(pairs[i][2], pairs[i][3]);
      if (!q) return std::nullopt;
      const double* wr = weighted_rows.data() + i * ncol;
      for (size_t r = 0; r < ncol; ++r) {
        atx[r] += wr[r] * (*q)[0];
        aty[r] += wr[r] * (*q)[1];
      }
    }
    auto cx = linalg::cholesky_solve(*chol, atx);
    auto cy = linalg::cholesky_solve(*chol, aty);
    Vec2 off = {cx[0], cy[0]};
    tp = proj.unproject(off[0], off[1], tp);
    bool done = std::fabs(off[0]) < 1e-11 && std::fabs(off[1]) < 1e-11;
    sol = std::make_pair(cx, cy);
    if (done) break;
  }
  if (!sol) return std::nullopt;
  return PolynomialFit{scale, tp, sol->first, sol->second};
}
}  // namespace

std::optional<Wcs> fit_model(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent, uint32_t order,
                             uint32_t radial, const Projection& proj_in) {
  Projection proj = proj_in.fixed(tangent);
  auto pf = fit_polynomial(pairs, weights, crpix, tangent, order, radial, proj);
  if (!pf) return std::nullopt;
  const double scale = pf->scale;
  const auto& cx = pf->x;
  const auto& cy = pf->y;
  auto terms = Sip::terms(order);
  auto rdeg = radial_degrees(order, radial);
  std::array<std::array<double, 2>, 2> cd = {{{cx[1] / scale, cx[2] / scale}, {cy[1] / scale, cy[2] / scale}}};
  double det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
  if (std::fabs(det) < 1e-30) return std::nullopt;
  double inv[2][2] = {{cd[1][1] / det, -cd[0][1] / det}, {-cd[1][0] / det, cd[0][0] / det}};
  uint32_t out_order = std::max(rdeg.empty() ? order : rdeg.back(), order);
  std::optional<Sip> sip;
  if (out_order >= 2) {
    auto all = Sip::terms(out_order);
    auto pos = [&](uint32_t p, uint32_t q) -> size_t {
      for (size_t i = 0; i < all.size(); ++i)
        if (all[i].first == p && all[i].second == q) return i;
      throw Error("SIP term");
    };
    std::vector<double> ex(all.size(), 0.0), ey(all.size(), 0.0);
    for (size_t k = 0; k < terms.size(); ++k) {
      size_t i = pos(terms[k].first, terms[k].second);
      ex[i] += cx[3 + k];
      ey[i] += cy[3 + k];
    }
    for (size_t k = 0; k < rdeg.size(); ++k) {
      uint32_t j = (rdeg[k] - 1) / 2;
      size_t c0 = 3 + terms.size() + 2 * k;
      double binom = 1.0;
      for (uint32_t i = 0; i <= j; ++i) {
        uint32_t pe = 2 * i, qe = 2 * (j - i);
        size_t iu = pos(pe + 1, qe), iv = pos(pe, qe + 1);
        ex[iu] += binom * cx[c0];
        ey[iu] += binom * cy[c0];
        ex[iv] += binom * cx[c0 + 1];
        ey[iv] += binom * cy[c0 + 1];
        binom = binom * double(j - i) / double(i + 1);
      }
    }
    Sip s;
    s.order = out_order;
    for (size_t i = 0; i < all.size(); ++i) {
      double f = powi(scale, int(all[i].first + all[i].second));
      double x = ex[i] / f, y = ey[i] / f;
      s.a.push_back(inv[0][0] * x + inv[0][1] * y);
      s.b.push_back(inv[1][0] * x + inv[1][1] * y);
    }
    sip = std::move(s);
  }
  Wcs w;
  w.ra = pf->tangent[0];
  w.dec = pf->tangent[1];
  w.crpix = crpix;
  w.cd = cd;
  w.sip = std::move(sip);
  w.proj = proj.proj;
  w.native_pole = proj.native_pole;
  return w;
}

std::vector<double> residuals(const Wcs& wcs, const Pairs& pairs) {
  double scale = std::sqrt(std::fabs(wcs.determinant()));
  auto projection = Projection(wcs.proj, wcs.native_pole).prepare({wcs.ra, wcs.dec});
  std::vector<double> out;
  out.reserve(pairs.size());
  for (const auto& p : pairs) {
    if (wcs.local_grid) {
      auto q = wcs.sky_to_pixel(p[2], p[3]);
      out.push_back(q ? std::hypot((*q)[0] - p[0], (*q)[1] - p[1]) : INF);
      continue;
    }
    auto q = projection.project(p[2], p[3]);
    if (!q) {
      out.push_back(INF);
      continue;
    }
    double u = p[0] - wcs.crpix[0], v = p[1] - wcs.crpix[1];
    if (wcs.sip) {
      auto [fa, fb] = wcs.sip->eval(u, v);
      u += fa;
      v += fb;
    }
    double xi = wcs.cd[0][0] * u + wcs.cd[0][1] * v;
    double eta = wcs.cd[1][0] * u + wcs.cd[1][1] * v;
    out.push_back(std::hypot(xi - (*q)[0], eta - (*q)[1]) / scale);
  }
  return out;
}

double rms(const std::vector<double>& r) {
  double s = -0.0;
  size_t n = 0;
  for (double x : r)
    if (std::isfinite(x)) {
      s += x * x;
      ++n;
    }
  if (n == 0) return INF;
  return std::sqrt(s / double(n));
}

static double median_of(std::vector<double> v) {
  if (v.empty()) return NaN;
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + long(n), v.end(), [](double a, double b) { return a < b; });
  return v[n];
}

std::optional<RobustFit> robust_fit(const Pairs& pairs, Vec2 crpix, Vec2 tangent, uint32_t order) {
  return robust_fit_w(pairs, {}, crpix, tangent, order);
}
std::optional<RobustFit> robust_fit_w(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent, uint32_t order) {
  return robust_fit_wm(pairs, weights, crpix, tangent, order, 0);
}
std::optional<RobustFit> robust_fit_wm(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                       uint32_t order, uint32_t radial) {
  return robust_fit_wmp(pairs, weights, crpix, tangent, order, radial, Projection(Proj::Tan));
}

std::optional<RobustFit> robust_fit_wmp(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                        uint32_t order, uint32_t radial, const Projection& proj_in) {
  Projection proj = proj_in.fixed(tangent);
  auto wt = [&](size_t i) { return i < weights.size() ? weights[i] : 1.0; };
  std::vector<bool> keep(pairs.size(), true);
  std::optional<Wcs> wcs;
  for (int it = 0; it < 4; ++it) {
    Pairs sub;
    std::vector<double> w;
    for (size_t i = 0; i < pairs.size(); ++i)
      if (keep[i]) {
        sub.push_back(pairs[i]);
        w.push_back(wt(i));
      }
    auto fit = fit_model(sub, w, crpix, tangent, order, radial, proj);
    if (!fit) return std::nullopt;
    auto r = residuals(*fit, pairs);
    std::vector<double> fin;
    for (double x : r)
      if (std::isfinite(x)) fin.push_back(x);
    double sigma = fmax_r(1.4826 * median_of(fin), 0.3);
    std::vector<bool> new_keep(r.size());
    for (size_t i = 0; i < r.size(); ++i) new_keep[i] = r[i] <= 3.0 * sigma;
    bool changed = new_keep != keep;
    keep = std::move(new_keep);
    wcs = std::move(fit);
    if (!changed) break;
  }
  if (!wcs) return std::nullopt;
  return RobustFit{std::move(*wcs), std::move(keep)};
}

double robust_test_rms(const std::vector<double>& r, const std::vector<double>& w) {
  std::vector<double> fin;
  for (double x : r)
    if (std::isfinite(x)) fin.push_back(x);
  if (fin.empty()) return INF;
  double cap = fmax_r(5.0 * median_of(fin), 3.0);
  double wmed = w.empty() ? 1.0 : median_of(w);
  double wcap = 4.0 * wmed;
  double s = 0.0, n = 0.0;
  for (size_t i = 0; i < r.size(); ++i) {
    double x = std::isfinite(r[i]) ? fmin_r(r[i], cap) : cap;
    double wi = fmin_r(i < w.size() ? w[i] : 1.0, wcap);
    s += wi * x * x;
    n += wi;
  }
  return n > 0.0 ? std::sqrt(s / n) : INF;
}

std::optional<OrderChoice> select_order(const Pairs& pairs, Vec2 crpix, Vec2 tangent, uint32_t max_order) {
  return select_order_w(pairs, {}, crpix, tangent, max_order);
}
std::optional<OrderChoice> select_order_w(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                          uint32_t max_order) {
  auto m = select_model_w(pairs, weights, crpix, tangent, max_order, 0);
  if (!m) return std::nullopt;
  return OrderChoice{std::move(m->wcs), std::move(m->keep), m->err};
}
std::optional<ModelChoice> select_model_w(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                          uint32_t max_order, uint32_t radial) {
  return select_model_wp(pairs, weights, crpix, tangent, max_order, radial, Projection(Proj::Tan));
}

std::optional<ModelChoice> select_model_wp(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                           uint32_t max_order, uint32_t radial, const Projection& proj_in) {
  Projection proj = proj_in.fixed(tangent);
  auto wt = [&](size_t i) { return i < weights.size() ? weights[i] : 1.0; };
  std::pair<uint32_t, uint32_t> baseline_model{1, 0};
  std::optional<RobustFit> baseline_fit;
  if (radial > 0) {
    std::pair<uint32_t, uint32_t> model{std::min<uint32_t>(max_order, 3), radial};
    baseline_fit = robust_fit_wmp(pairs, weights, crpix, tangent, model.first, model.second, proj);
    if (baseline_fit) baseline_model = model;
    else {
      baseline_model = {1, 0};
      baseline_fit = robust_fit_wmp(pairs, weights, crpix, tangent, 1, 0, proj);
      if (!baseline_fit) return std::nullopt;
    }
  } else {
    baseline_fit = robust_fit_wmp(pairs, weights, crpix, tangent, 1, 0, proj);
    if (!baseline_fit) return std::nullopt;
  }
  std::vector<size_t> inl_idx;
  for (size_t i = 0; i < pairs.size(); ++i)
    if (baseline_fit->keep[i]) inl_idx.push_back(i);
  std::vector<size_t> tr, te;
  for (size_t k = 0; k < inl_idx.size(); ++k) (k % 4 != 0 ? tr : te).push_back(inl_idx[k]);
  Pairs train, test;
  std::vector<double> trw, tew;
  for (size_t i : tr) {
    train.push_back(pairs[i]);
    trw.push_back(wt(i));
  }
  for (size_t i : te) {
    test.push_back(pairs[i]);
    tew.push_back(wt(i));
  }
  auto test_err = [&](uint32_t order, uint32_t rad) -> std::optional<double> {
    if (test.size() < 8) return std::nullopt;
    auto fit = robust_fit_wmp(train, trw, crpix, tangent, order, rad, proj);
    if (!fit) return std::nullopt;
    return robust_test_rms(residuals(fit->wcs, test), tew);
  };
  std::vector<std::pair<uint32_t, uint32_t>> cand;
  for (uint32_t o = 2; o <= max_order; ++o) cand.emplace_back(o, 0);
  if (radial > 0)
    for (uint32_t o = 1; o <= max_order; ++o)
      if (!radial_degrees(o, radial).empty()) cand.emplace_back(o, radial);
  std::stable_sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
    size_t ca = model_columns(a.first, a.second), cb = model_columns(b.first, b.second);
    if (ca != cb) return ca < cb;
    return a.first < b.first;
  });
  std::pair<std::pair<uint32_t, uint32_t>, double> best{{1, 0}, test_err(1, 0).value_or(INF)};
  for (auto [o, r] : cand) {
    size_t base;
    switch (o) {
      case 2: base = 30; break;
      case 3: base = 60; break;
      case 4: base = 100; break;
      case 5: base = 160; break;
      default: base = 240; break;
    }
    size_t need = r > 0 ? std::max<size_t>(base, 6 * model_columns(o, r)) : base;
    if (inl_idx.size() < need) continue;
    auto e = test_err(o, r);
    if (e && *e < 0.9 * best.second) best = {{o, r}, *e};
  }
  RobustFit fit;
  if (best.first == baseline_model) {
    fit = std::move(*baseline_fit);
  } else {
    auto f = robust_fit_wmp(pairs, weights, crpix, tangent, best.first.first, best.first.second, proj);
    if (!f) return std::nullopt;
    fit = std::move(*f);
  }
  return ModelChoice{std::move(fit.wcs), std::move(fit.keep), best.second, best.first};
}

}  // namespace skyshot::wcsfit
