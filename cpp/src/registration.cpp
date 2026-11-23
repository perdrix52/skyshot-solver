#include "skyshot/registration.hpp"

#include <algorithm>
#include <unordered_set>

#include "skyshot/geometry.hpp"
#include "skyshot/linalg.hpp"
#include "skyshot/select.hpp"

namespace skyshot::registration {

namespace {
struct Tie {
  size_t cat;
  uint32_t image;
  double distance;
};

double median(std::vector<double> v) {
  std::stable_sort(v.begin(), v.end(), [](double a, double b) { return total_cmp(a, b) < 0; });
  return v[v.size() / 2];
}
}  // namespace

std::optional<PixelGrid> fit(const std::vector<std::array<double, 5>>& cat, const std::vector<Vec2>& image, double width,
                             double height, const std::function<bool(Vec2)>& visible, const SkyMask* mask, double budget_ms) {
  using solve::Clock;
  using solve::ms_since;
  if (cat.size() < 100 || image.size() < 100 || budget_ms < 20.) return std::nullopt;
  auto start = Clock::now();
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < image.size(); ++i)
    if (visible(image[i])) ids.push_back(i);
  auto detections = select::Grid2::build(image, ids, width, height, 3.);
  auto train = [&](size_t i) { return cat[i][4] >= 5.5 && i % 4 != 0; };
  std::vector<size_t> training;
  for (size_t i = 0; i < cat.size(); ++i)
    if (train(i) && visible({cat[i][0], cat[i][1]})) training.push_back(i);
  std::vector<std::optional<Tie>> ties(cat.size());
  double spacing = fmax_r(fmax_r(width, height) / 16., 48.);
  double radius = fmax_r(spacing, 96.);
  double search = clampd(fmax_r(width, height) * 0.016, 8., 32.);
  double bin = search / 16.;
  size_t nx = sat_cast<size_t>(std::ceil(width / spacing)), ny = sat_cast<size_t>(std::ceil(height / spacing));
  double threshold = 14. + std::log(double((nx + 1) * (ny + 1) * 33 * 33));
  std::vector<std::pair<double, uint32_t>> nearby;
  for (size_t gy = 0; gy <= ny; ++gy)
    for (size_t gx = 0; gx <= nx; ++gx) {
      if (ms_since(start) > budget_ms) return std::nullopt;
      Vec2 center = {fmin_r(double(gx) * spacing, width - 1.), fmin_r(double(gy) * spacing, height - 1.)};
      std::vector<size_t> local;
      for (size_t i : training)
        if (std::fabs(cat[i][0] - center[0]) < radius && std::fabs(cat[i][1] - center[1]) < radius) local.push_back(i);
      size_t local_detections = 0;
      for (uint32_t i : ids)
        if (std::fabs(image[i][0] - center[0]) < radius && std::fabs(image[i][1] - center[1]) < radius) ++local_detections;
      std::stable_sort(local.begin(), local.end(), [&](size_t a, size_t b) { return total_cmp(cat[a][4], cat[b][4]) < 0; });
      if (local.size() > local_detections) local.resize(local_detections);
      if (local.size() < 12) continue;
      uint32_t hist[33 * 33] = {};
      for (size_t i : local) {
        Vec2 p = {cat[i][0], cat[i][1]};
        detections.knn(image, p, 32, UINT32_MAX, nearby);
        for (auto& [distance, j] : nearby) {
          if (distance > search * search) break;
          double d0 = (image[j][0] - p[0]) / bin, d1 = (image[j][1] - p[1]) / bin;
          int x = std::clamp(int(sat_cast<int32_t>(std::round(d0))) + 16, 0, 32);
          int y = std::clamp(int(sat_cast<int32_t>(std::round(d1))) + 16, 0, 32);
          hist[y * 33 + x] += 1;
        }
      }
      uint32_t best_v = 0;
      int best_x = 16, best_y = 16;
      for (int y = 1; y < 32; ++y)
        for (int x = 1; x < 32; ++x) {
          uint32_t value = 0;
          const int dws[3][2] = {{-1, 1}, {0, 2}, {1, 1}};
          for (auto& dy : dws)
            for (auto& dx : dws) value += hist[(y + dy[0]) * 33 + (x + dx[0])] * uint32_t(dy[1]) * uint32_t(dx[1]);
          if (value > best_v) {
            best_v = value;
            best_x = x;
            best_y = y;
          }
        }
      Vec2 shift = {(double(best_x) - 16.) * bin, (double(best_y) - 16.) * bin};
      std::vector<std::pair<size_t, uint32_t>> matched;
      for (int it = 0; it < 2; ++it) {
        matched.clear();
        std::unordered_set<uint32_t> used;
        for (size_t i : local) {
          Vec2 q = {cat[i][0] + shift[0], cat[i][1] + shift[1]};
          auto nn = detections.nearest_within(image, q, 2.8, UINT32_MAX);
          if (nn && used.insert(nn->second).second && visible(image[nn->second])) matched.emplace_back(i, nn->second);
        }
        if (matched.size() < 12) break;
        for (size_t k = 0; k < 2; ++k) {
          std::vector<double> v;
          for (auto& [i, j] : matched) v.push_back(image[j][k] - cat[i][k]);
          shift[k] = median(v);
        }
      }
      if (matched.size() < 12) continue;
      double x0 = fmax_r(center[0] - radius - search, 0.), x1 = fmin_r(center[0] + radius + search, width);
      double y0 = fmax_r(center[1] - radius - search, 0.), y1 = fmin_r(center[1] + radius + search, height);
      size_t n_image = 0;
      for (uint32_t j : ids)
        if (image[j][0] >= x0 && image[j][0] < x1 && image[j][1] >= y0 && image[j][1] < y1) ++n_image;
      double area = fmax_r((x1 - x0) * (y1 - y0) - (mask ? mask->blocked_area(x0, y0, x1, y1) : 0.), 1.);
      double lambda = double(local.size()) * double(n_image) * PI * 2.8 * 2.8 / area;
      if (solve::poisson_log_odds(matched.size(), lambda) < threshold) continue;
      for (auto& [i, j] : matched) {
        double distance = std::hypot(cat[i][0] - center[0], cat[i][1] - center[1]);
        if (!ties[i] || distance < ties[i]->distance) ties[i] = Tie{i, j, distance};
      }
    }
  std::unordered_set<uint32_t> seen;
  std::vector<Tie> tl;
  for (auto& t : ties)
    if (t && seen.insert(t->image).second) tl.push_back(*t);
  if (tl.size() < 80) return std::nullopt;
  std::vector<Vec2> q, delta;
  for (auto& t : tl) {
    q.push_back({cat[t.cat][0], cat[t.cat][1]});
    delta.push_back({image[t.image][0] - cat[t.cat][0], image[t.image][1] - cat[t.cat][1]});
  }
  std::vector<uint32_t> tie_ids(tl.size());
  for (size_t i = 0; i < tie_ids.size(); ++i) tie_ids[i] = uint32_t(i);
  auto tie_grid = select::Grid2::build(q, tie_ids, width, height, 3.);
  size_t cols = 65;
  size_t rows = std::clamp<size_t>(sat_cast<size_t>(std::ceil(height / width * 64.)) + 1, 2, 129);
  Vec2 step = {(width - 1.) / double(cols - 1), (height - 1.) / double(rows - 1)};
  double bandwidth = fmax_r(fmax_r(width, height) / 64., 8.);
  std::vector<Vec2> offsets;
  for (size_t y = 0; y < rows; ++y) {
    if (ms_since(start) > budget_ms) return std::nullopt;
    for (size_t x = 0; x < cols; ++x) {
      Vec2 at = {double(x) * step[0], double(y) * step[1]};
      tie_grid.knn(q, at, 32, UINT32_MAX, nearby);
      std::vector<size_t> local;
      for (auto& [distance, i] : nearby)
        if (distance <= powi(3. * bandwidth, 2)) local.push_back(i);
      if (local.size() < 6) {
        offsets.push_back({0., 0.});
        continue;
      }
      Vec2 median_delta;
      for (size_t k = 0; k < 2; ++k) {
        std::vector<double> v;
        for (size_t i : local) v.push_back(delta[i][k]);
        median_delta[k] = median(v);
      }
      double lhs[3][3] = {};
      double rhs[2][3] = {};
      int count = 0;
      double weight = 0.;
      for (auto& [distance, i] : nearby) {
        if (distance > powi(3. * bandwidth, 2)) break;
        if (std::hypot(delta[i][0] - median_delta[0], delta[i][1] - median_delta[1]) > 2.) continue;
        double z[3] = {1., (q[i][0] - at[0]) / bandwidth, (q[i][1] - at[1]) / bandwidth};
        double w = std::exp(-distance / (2. * bandwidth * bandwidth));
        for (int a = 0; a < 3; ++a) {
          double wz = w * z[a];
          for (int b = 0; b < 3; ++b) lhs[a][b] += wz * z[b];
          for (int k = 0; k < 2; ++k) rhs[k][a] += wz * delta[i][size_t(k)];
        }
        count += 1;
        weight += w;
      }
      lhs[1][1] += 0.2;
      lhs[2][2] += 0.2;
      Vec2 d = {0., 0.};
      if (count >= 6 && weight >= 1.5) {
        linalg::M3 m = {{{lhs[0][0], lhs[0][1], lhs[0][2]}, {lhs[1][0], lhs[1][1], lhs[1][2]}, {lhs[2][0], lhs[2][1], lhs[2][2]}}};
        if (auto inv = linalg::inverse3(m)) {
          for (size_t k = 0; k < 2; ++k) d[k] = (*inv)[0][0] * rhs[k][0] + (*inv)[0][1] * rhs[k][1] + (*inv)[0][2] * rhs[k][2];
        }
      }
      offsets.push_back(std::hypot(d[0], d[1]) < 0.5 ? Vec2{0., 0.} : d);
    }
  }
  PixelGrid grid;
  grid.cols = cols;
  grid.rows = rows;
  grid.origin = {0., 0.};
  grid.step = step;
  grid.offsets = std::move(offsets);
  struct Val {
    Vec2 p;
    double before, after;
    bool bright;
  };
  std::vector<Val> validation;
  for (size_t i = 0; i < cat.size(); ++i) {
    if (train(i) || !visible({cat[i][0], cat[i][1]})) continue;
    Vec2 p = {cat[i][0], cat[i][1]};
    Vec2 corrected = grid.forward(p);
    auto nb = detections.nearest_within(image, p, search, UINT32_MAX);
    auto na = detections.nearest_within(image, corrected, search, UINT32_MAX);
    double before = nb ? std::sqrt(nb->first) : search;
    double after = na ? std::sqrt(na->first) : search;
    if (before < 2. || after < 2.) validation.push_back({p, before, after, cat[i][4] < 5.5});
  }
  std::vector<Vec2> vpoints;
  for (auto& v : validation) vpoints.push_back(v.p);
  std::vector<uint32_t> vids(validation.size());
  for (size_t i = 0; i < vids.size(); ++i) vids[i] = uint32_t(i);
  auto vgrid = select::Grid2::build(vpoints, vids, width, height, 3.);
  for (size_t y = 0; y < rows; ++y) {
    if (ms_since(start) > budget_ms) return std::nullopt;
    for (size_t x = 0; x < cols; ++x) {
      Vec2 at = {double(x) * step[0], double(y) * step[1]};
      double before = 0., after = 0., weight = 0.;
      int count = 0;
      bool veto = false;
      vgrid.knn(vpoints, at, 64, UINT32_MAX, nearby);
      for (auto& [distance, j] : nearby) {
        if (distance > powi(2. * bandwidth, 2)) break;
        const Val& v = validation[j];
        double w = std::exp(-distance / (2. * bandwidth * bandwidth));
        before += w * powi(fmin_r(v.before, 6.), 2);
        after += w * powi(fmin_r(v.after, 6.), 2);
        weight += w;
        count += 1;
        if (v.bright && v.before < 1. && v.after > 1.5) veto = true;
      }
      if (veto || count < 3 || weight < 0.5 || after >= 0.8 * before) grid.offsets[y * cols + x] = {0., 0.};
    }
  }
  for (int it = 0; it < 4; ++it) {
    if (!grid.validate()) break;
    auto old = grid.offsets;
    for (size_t y = 0; y < rows; ++y)
      for (size_t x = 0; x < cols; ++x)
        for (size_t k = 0; k < 2; ++k) {
          double old_value = old[y * cols + x][k];
          double sum = 0., n = 0.;
          for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              size_t xx = size_t(std::clamp<int64_t>(int64_t(x) + dx, 0, int64_t(cols) - 1));
              size_t yy = size_t(std::clamp<int64_t>(int64_t(y) + dy, 0, int64_t(rows) - 1));
              sum += old[yy * cols + xx][k];
              n += 1.;
            }
          grid.offsets[y * cols + x][k] = 0.5 * old_value + 0.5 * sum / n;
        }
  }
  if (grid.validate()) return std::nullopt;
  std::vector<double> before_v, after_v;
  int gained = 0, lost = 0;
  std::unordered_set<uint32_t> seen_before, seen_after;
  for (size_t i = 0; i < cat.size(); ++i) {
    if (train(i)) continue;
    Vec2 p = {cat[i][0], cat[i][1]};
    if (!visible(p)) continue;
    Vec2 corrected = grid.forward(p);
    auto a0 = detections.nearest_within(image, p, search, UINT32_MAX);
    auto b0 = detections.nearest_within(image, corrected, search, UINT32_MAX);
    double a = search, b = search;
    if (a0 && (a0->first >= 4. || seen_before.insert(a0->second).second)) a = std::sqrt(a0->first);
    if (b0 && (b0->first >= 4. || seen_after.insert(b0->second).second)) b = std::sqrt(b0->first);
    if (b < 2. && a >= 2.) ++gained;
    if (a < 2. && b >= 2.) ++lost;
    Vec2 off = grid.offset(p);
    if ((a < 2. || b < 2.) && (std::fabs(off[0]) > 2. || std::fabs(off[1]) > 2.)) {
      before_v.push_back(fmin_r(a, 6.));
      after_v.push_back(fmin_r(b, 6.));
    }
  }
  if (after_v.size() < 12 || gained < lost + 8 || median(after_v) > 1.5) return std::nullopt;
  auto rms = [](const std::vector<double>& v) {
    double s = -0.0;
    for (double e : v) s += e * e;
    return std::sqrt(s / double(v.size()));
  };
  if (rms(after_v) > 0.8 * rms(before_v)) return std::nullopt;
  return grid;
}

void refine_solution(const index::Index& idx, solve::Solution& solution, const StarField& field, const solve::SolveConfig& cfg,
                     const SkyMask* mask, double remaining_ms) {
  if (!cfg.allow_local_grid || solution.status != "solved" || remaining_ms < 30.) return;
  if (!solution.wcs) return;
  const Wcs& base = *solution.wcs;
  if ((base.proj != Proj::Stg && !is_cylindrical(base.proj)) || base.local_grid) return;
  auto start = solve::Clock::now();
  double width = double(field.width), height = double(field.height);
  auto visible = [&](Vec2 p) { return p[0] >= 0. && p[0] < width && p[1] >= 0. && p[1] < height && !(mask && mask->is_blocked(p[0], p[1])); };
  std::vector<const Star*> image_s;
  for (const Star& s : field.stars)
    if (visible({s.x, s.y})) image_s.push_back(&s);
  std::stable_sort(image_s.begin(), image_s.end(), [](const Star* a, const Star* b) { return total_cmp(b->flux, a->flux) < 0; });
  if (image_s.size() > cfg.max_stars) image_s.resize(cfg.max_stars);
  std::vector<Vec2> image;
  for (auto* s : image_s) image.push_back({s->x, s->y});
  std::vector<index::BrightStar> stars;
  idx.cone_stars(geometry::unit(base.ra, base.dec), PI, fmin_r(cfg.mag_limit.value_or(9.5f), 9.5f), 0., stars);
  auto prepared = geometry::prepare(base);
  std::vector<std::array<double, 5>> cat;
  for (auto& s : stars) {
    auto [ra, dec] = sphere::radec(s.v);
    auto p = prepared.sky_to_pixel_fast(ra, dec);
    if (!p) continue;
    if (visible(*p)) cat.push_back({(*p)[0], (*p)[1], ra, dec, double(s.mag)});
  }
  auto grid = fit(cat, image, width, height, visible, mask, remaining_ms - solve::ms_since(start));
  if (!grid) return;
  Wcs candidate = base;
  candidate.local_grid = std::make_shared<PixelGrid>(std::move(*grid));
  for (int y = 0; y <= 12; ++y)
    for (int x = 0; x <= 20; ++x) {
      Vec2 p = {double(x) * (width - 1.) / 20., double(y) * (height - 1.) / 12.};
      if (!visible(p)) continue;
      Vec2 s = candidate.pixel_to_sky(p[0], p[1]);
      auto q = candidate.sky_to_pixel(s[0], s[1]);
      if (!q) return;
      if (std::hypot((*q)[0] - p[0], (*q)[1] - p[1]) > 1e-3) return;
    }
  for (auto& m : solution.matches) {
    auto q = candidate.sky_to_pixel(m.ra, m.dec);
    if (q) m.residual_px = std::hypot(m.x - (*q)[0], m.y - (*q)[1]);
  }
  if (!solution.matches.empty()) {
    double s = -0.0;
    for (auto& m : solution.matches) s += powi(m.residual_px, 2);
    solution.rms_px = std::sqrt(s / double(solution.matches.size()));
  }
  Vec2 center = candidate.pixel_to_sky((width - 1.) / 2., (height - 1.) / 2.);
  solution.ra = center[0];
  solution.dec = center[1];
  solution.refraction.reset();
  solution.wcs = std::move(candidate);
}

}  // namespace skyshot::registration
