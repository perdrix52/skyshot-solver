// Recover a cylinder's axis from an already identified window. Port of src/wcsfit/panorama.rs.
#include <algorithm>

#include "skyshot/sphere.hpp"
#include "skyshot/wcsfit.hpp"

namespace skyshot::wcsfit {

using sphere::normalize;
using sphere::radec;
using sphere::tangent_basis;

std::optional<Wcs> cylinder_seed(const Pairs& pairs, Vec2 crpix, Vec2 tangent, Vec2 pole_in) {
  if (pairs.size() < 32) return std::nullopt;
  Pairs train, test;
  for (size_t i = 0; i < pairs.size(); ++i) {
    if (i % 4 != 0) train.push_back(pairs[i]);
    if (i % 4 == 0) test.push_back(pairs[i]);
  }
  auto at = [&](Vec2 pole) -> std::optional<std::pair<double, Wcs>> {
    Projection spec(Proj::Car, pole);
    Pairs sub = train;
    auto fit = fit_model(sub, {}, crpix, tangent, 1, 0, spec);
    if (!fit) return std::nullopt;
    for (int it = 0; it < 3; ++it) {
      auto errors = residuals(*fit, train);
      std::vector<size_t> ids;
      for (size_t i = 0; i < train.size(); ++i)
        if (std::isfinite(errors[i])) ids.push_back(i);
      std::stable_sort(ids.begin(), ids.end(), [&](size_t a, size_t b) { return total_cmp(errors[a], errors[b]) < 0; });
      ids.resize(std::min(ids.size(), train.size() * 4 / 5));
      if (ids.size() < 18) return std::nullopt;
      sub.clear();
      for (size_t i : ids) sub.push_back(train[i]);
      fit = fit_model(sub, {}, crpix, tangent, 1, 0, spec);
      if (!fit) return std::nullopt;
    }
    return std::make_pair(rms(residuals(*fit, sub)), std::move(*fit));
  };
  auto first = at(pole_in);
  if (!first) return std::nullopt;
  double score = first->first;
  Wcs fit = std::move(first->second);
  Vec2 pole = pole_in;
  double step = to_radians(12.0);
  for (int it = 0; it < 40; ++it) {
    Vec3 v = sphere::unit(pole[0], pole[1]);
    auto [east, north] = tangent_basis(v);
    double old = score;
    const double ds[4][2] = {{step, 0.}, {-step, 0.}, {0., step}, {0., -step}};
    for (auto& dd : ds) {
      Vec3 q = normalize({v[0] + dd[0] * east[0] + dd[1] * north[0], v[1] + dd[0] * east[1] + dd[1] * north[1],
                          v[2] + dd[0] * east[2] + dd[1] * north[2]});
      auto [ra, dec] = radec(q);
      auto r = at({ra, dec});
      if (r && r->first < score) {
        score = r->first;
        pole = {ra, dec};
        fit = std::move(r->second);
      }
    }
    if (score >= old) step *= 0.5;
    if (step < 1e-6) break;
  }
  auto errors = residuals(fit, test);
  size_t good = 0;
  for (double e : errors)
    if (std::isfinite(e) && e < 2.5) ++good;
  if (good * 10 < test.size() * 7) return std::nullopt;
  return fit;
}

std::vector<Projection> panorama_candidates(const Pairs& pairs, Vec2 crpix, Vec2 tangent) {
  if (pairs.size() < 24) return {};
  Pairs train, test;
  for (size_t i = 0; i < pairs.size(); ++i) {
    if (i % 4 != 0) train.push_back(pairs[i]);
    if (i % 4 == 0) test.push_back(pairs[i]);
  }
  std::vector<std::pair<double, Projection>> out;
  for (Proj proj : {Proj::Car, Proj::Mer, Proj::Cyp}) {
    auto score = [&](Vec2 pole) -> double {
      Projection spec(proj, pole);
      auto fit = fit_model(train, {}, crpix, tangent, 1, 0, spec);
      if (!fit) return INF;
      auto errors = residuals(*fit, train);
      for (double e : errors)
        if (!std::isfinite(e)) return INF;
      return rms(errors);
    };
    std::pair<double, Vec2> best{INF, {0.0, 0.0}};
    double golden = PI * (3.0 - std::sqrt(5.0));
    for (int i = 0; i < 48; ++i) {
      double z = (double(i) + 0.5) / 48.0;
      double r = std::sqrt(1.0 - z * z);
      auto [ra, dec] = radec({r * std::cos(golden * double(i)), r * std::sin(golden * double(i)), z});
      Vec2 pole{ra, dec};
      double e = score(pole);
      if (e < best.first) best = {e, pole};
    }
    double step = to_radians(12.0);
    for (int it = 0; it < 24; ++it) {
      Vec3 v = sphere::unit(best.second[0], best.second[1]);
      auto [east, north] = tangent_basis(v);
      double old = best.first;
      const double ds[4][2] = {{step, 0.0}, {-step, 0.0}, {0.0, step}, {0.0, -step}};
      for (auto& dd : ds) {
        Vec3 q = normalize({v[0] + dd[0] * east[0] + dd[1] * north[0], v[1] + dd[0] * east[1] + dd[1] * north[1],
                            v[2] + dd[0] * east[2] + dd[1] * north[2]});
        auto [ra, dec] = radec(q);
        Vec2 pole{ra, dec};
        double e = score(pole);
        if (e < best.first) best = {e, pole};
      }
      if (best.first >= old) step *= 0.5;
    }
    if (std::isfinite(best.first)) {
      Projection spec(proj, best.second);
      auto fit = fit_model(train, {}, crpix, tangent, 1, 0, spec);
      if (fit) {
        auto errors = residuals(*fit, test);
        bool all = true;
        for (double e : errors)
          if (!std::isfinite(e)) all = false;
        if (all) out.emplace_back(rms(errors), spec);
      }
    }
  }
  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return total_cmp(a.first, b.first) < 0; });
  std::vector<Projection> res;
  for (auto& p : out) res.push_back(p.second);
  return res;
}

}  // namespace skyshot::wcsfit
