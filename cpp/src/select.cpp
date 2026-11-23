#include "skyshot/select.hpp"

#include <set>

namespace skyshot::select {

static inline int64_t clamp_cell(double v, size_t count) {
  int64_t c = sat_cast<int64_t>(fmax_r(std::floor(v), 0.0));
  return std::min<int64_t>(c, int64_t(count) - 1);
}

Grid2 Grid2::build(const std::vector<P2>& pts, const std::vector<uint32_t>& ids, double width, double height,
                   double per_cell) {
  double n = double(std::max<size_t>(ids.size(), 1));
  Grid2 g;
  g.cell = fmax_r(std::sqrt(width * height * per_cell / n), 1.0);
  g.nx = std::max<size_t>(sat_cast<uint64_t>(std::ceil(width / g.cell)), 1);
  g.ny = std::max<size_t>(sat_cast<uint64_t>(std::ceil(height / g.cell)), 1);
  std::vector<uint32_t> counts(g.nx * g.ny + 1, 0);
  auto cidx = [&](const P2& p) -> size_t {
    size_t x = std::min<size_t>(sat_cast<uint64_t>(fmax_r(std::floor(p[0] / g.cell), 0.0)), g.nx - 1);
    size_t y = std::min<size_t>(sat_cast<uint64_t>(fmax_r(std::floor(p[1] / g.cell), 0.0)), g.ny - 1);
    return y * g.nx + x;
  };
  for (uint32_t i : ids) counts[cidx(pts[i]) + 1] += 1;
  for (size_t i = 0; i < g.nx * g.ny; ++i) counts[i + 1] += counts[i];
  g.offsets = counts;
  std::vector<uint32_t> fill = counts;
  g.items.assign(ids.size(), 0);
  for (uint32_t i : ids) {
    size_t c = cidx(pts[i]);
    g.items[fill[c]] = i;
    fill[c] += 1;
  }
  return g;
}

double Grid2::cell_axis_distance(double q, size_t c, size_t count) const {
  double lo = c == 0 ? -INF : std::nextafter(double(c) * cell, -INF);
  double hi = c + 1 == count ? INF : std::nextafter(double(c + 1) * cell, INF);
  return fmax_r(fmax_r(lo - q, q - hi), 0.0);
}

double Grid2::outside_distance(P2 q, int64_t cx, int64_t cy, int64_t ring) const {
  double distance = INF;
  struct T {
    double p;
    int64_t c;
    size_t count;
  };
  T ts[2] = {{q[0], cx, nx}, {q[1], cy, ny}};
  for (auto& t : ts) {
    if (t.c - ring > 0) {
      double edge = std::nextafter(double(t.c - ring) * cell, INF);
      distance = fmin_r(distance, fmax_r(t.p - edge, 0.0));
    }
    if (t.c + ring + 1 < int64_t(t.count)) {
      double edge = std::nextafter(double(t.c + ring + 1) * cell, -INF);
      distance = fmin_r(distance, fmax_r(edge - t.p, 0.0));
    }
  }
  return distance;
}

std::optional<std::pair<double, uint32_t>> Grid2::nearest_within(const std::vector<P2>& pts, P2 q, double radius,
                                                                 uint32_t skip) const {
  if (std::isnan(radius) || radius < 0.0 || !std::isfinite(q[0]) || !std::isfinite(q[1])) return std::nullopt;
  double r2 = radius * radius;
  int64_t cx = clamp_cell(q[0] / cell, nx), cy = clamp_cell(q[1] / cell, ny);
  std::optional<std::pair<double, uint32_t>> best;
  int64_t maxring = int64_t(std::max(nx, ny));
  for (int64_t ring = 0; ring <= maxring; ++ring) {
    for (int64_t y = cy - ring; y <= cy + ring; ++y) {
      if (y < 0 || y >= int64_t(ny)) continue;
      double dy = cell_axis_distance(q[1], size_t(y), ny);
      bool full = y == cy - ring || y == cy + ring;
      int64_t x = cx - ring;
      while (x <= cx + ring) {
        if (x >= 0 && x < int64_t(nx)) {
          double dx = cell_axis_distance(q[0], size_t(x), nx);
          double limit = best ? best->first : r2;
          if (dx * dx + dy * dy <= limit) {
            size_t c = size_t(y) * nx + size_t(x);
            for (size_t k = offsets[c]; k < offsets[c + 1]; ++k) {
              uint32_t i = items[k];
              if (i == skip) continue;
              const P2& p = pts[i];
              double d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
              if (d2 <= r2 && (!best || d2 < best->first)) best = std::make_pair(d2, i);
            }
          }
        }
        x += (full || ring == 0) ? 1 : 2 * ring;
      }
    }
    double outside = outside_distance(q, cx, cy, ring);
    double outside2 = outside * outside;
    if ((best && best->first <= outside2) || r2 < outside2) return best;
  }
  return best;
}

void Grid2::knn(const std::vector<P2>& pts, P2 q, size_t k, uint32_t skip,
                std::vector<std::pair<double, uint32_t>>& out) const {
  out.clear();
  int64_t cx = clamp_cell(q[0] / cell, nx), cy = clamp_cell(q[1] / cell, ny);
  int64_t maxr = int64_t(std::max(nx, ny));
  auto by_dist = [](const std::pair<double, uint32_t>& a, const std::pair<double, uint32_t>& b) {
    return a.first < b.first;
  };
  for (int64_t ring = 0; ring <= maxr; ++ring) {
    for (int64_t y = cy - ring; y <= cy + ring; ++y) {
      if (y < 0 || y >= int64_t(ny)) continue;
      bool full = y == cy - ring || y == cy + ring;
      int64_t x = cx - ring;
      while (x <= cx + ring) {
        if (x >= 0 && x < int64_t(nx)) {
          size_t c = size_t(y) * nx + size_t(x);
          for (size_t t = offsets[c]; t < offsets[c + 1]; ++t) {
            uint32_t i = items[t];
            if (i == skip) continue;
            const P2& p = pts[i];
            double d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
            if (out.size() < k) {
              out.emplace_back(d2, i);
              if (out.size() == k) std::stable_sort(out.begin(), out.end(), by_dist);
            } else if (d2 < out[k - 1].first) {
              out[k - 1] = {d2, i};
              size_t j = k - 1;
              while (j > 0 && out[j].first < out[j - 1].first) {
                std::swap(out[j], out[j - 1]);
                --j;
              }
            }
          }
        }
        x += (full || ring == 0) ? 1 : 2 * ring;
      }
    }
    if (out.size() == k) {
      double covered = double(ring) * cell;
      if (out[k - 1].first <= covered * covered) return;
    }
  }
  std::stable_sort(out.begin(), out.end(), by_dist);
}

std::vector<double> dominance_px(const std::vector<P2>& pts, double width, double height) {
  size_t n = pts.size();
  std::vector<double> d(n, INF);
  if (n < 2) return d;
  double cell = fmax_r(std::sqrt(width * height * 2.0 / double(n)), 2.0);
  size_t nx = std::max<size_t>(sat_cast<uint64_t>(std::ceil(width / cell)), 1);
  size_t ny = std::max<size_t>(sat_cast<uint64_t>(std::ceil(height / cell)), 1);
  std::vector<std::vector<uint32_t>> cells(nx * ny);
  auto cidx = [&](const P2& p) { return std::make_pair(clamp_cell(p[0] / cell, nx), clamp_cell(p[1] / cell, ny)); };
  int64_t maxr = int64_t(std::max(nx, ny));
  for (size_t i = 0; i < n; ++i) {
    P2 q = pts[i];
    auto [cx, cy] = cidx(q);
    if (i > 0) {
      double best = INF;
      for (int64_t ring = 0; ring <= maxr; ++ring) {
        for (int64_t y = cy - ring; y <= cy + ring; ++y) {
          if (y < 0 || y >= int64_t(ny)) continue;
          bool full = y == cy - ring || y == cy + ring;
          int64_t x = cx - ring;
          while (x <= cx + ring) {
            if (x >= 0 && x < int64_t(nx)) {
              for (uint32_t j : cells[size_t(y) * nx + size_t(x)]) {
                const P2& p = pts[j];
                double d2 = (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
                if (d2 < best) best = d2;
              }
            }
            x += (full || ring == 0) ? 1 : 2 * ring;
          }
        }
        double covered = double(ring) * cell;
        if (best <= covered * covered) break;
      }
      d[i] = std::sqrt(best);
    }
    cells[size_t(cy) * nx + size_t(cx)].push_back(uint32_t(i));
  }
  return d;
}

std::vector<double> edge_px(const std::vector<P2>& pts, double width, double height) {
  std::vector<double> out;
  out.reserve(pts.size());
  for (const auto& p : pts)
    out.push_back(fmax_r(fmin_r(fmin_r(fmin_r(p[0], p[1]), width - 1.0 - p[0]), height - 1.0 - p[1]), 0.0));
  return out;
}

std::vector<int> rung_order(const std::vector<double>& d, size_t target, size_t min_pool, size_t max_pool) {
  std::vector<double> finite;
  for (double x : d)
    if (std::isfinite(x)) finite.push_back(x);
  std::stable_sort(finite.begin(), finite.end(), [](double a, double b) { return a > b; });
  auto pool_at = [&](double r) -> size_t {
    size_t lo = 0, hi = finite.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (finite[mid] >= r) lo = mid + 1;
      else hi = mid;
    }
    return lo + 1;
  };
  if (finite.size() + 1 < min_pool) return {};
  double rmax = finite[0];
  int kmax = sat_cast<int32_t>(std::floor(std::log2(rmax) / RUNG_STEP_LOG2));
  std::vector<std::pair<int, size_t>> valid;
  int k = kmax;
  while (k > -8) {
    size_t n = pool_at(rung_radius(k));
    if (n > max_pool) break;
    if (n >= min_pool) valid.emplace_back(k, n);
    --k;
  }
  if (valid.empty()) return {};
  size_t start = 0;
  double bestf = 0;
  for (size_t i = 0; i < valid.size(); ++i) {
    double f = std::fabs(std::log(double(valid[i].second) / double(target)));
    if (i == 0 || f < bestf) {
      bestf = f;
      start = i;
    }
  }
  std::vector<int> out{valid[start].first};
  int64_t lo = int64_t(start) - 1;
  size_t hi = start + 1;
  while (lo >= 0 || hi < valid.size()) {
    if (hi < valid.size()) out.push_back(valid[hi++].first);
    if (lo >= 0) out.push_back(valid[size_t(lo--)].first);
  }
  std::set<size_t> seen;
  std::vector<int> res;
  for (int kk : out)
    if (seen.insert(pool_at(rung_radius(kk))).second) res.push_back(kk);
  return res;
}

}  // namespace skyshot::select
