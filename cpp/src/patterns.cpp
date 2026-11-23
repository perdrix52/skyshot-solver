#include "skyshot/patterns.hpp"

#include <algorithm>

namespace skyshot::patterns {

double distance(const Star& a, const Star& b) { return std::hypot(a.x - b.x, a.y - b.y); }

static std::optional<std::pair<std::array<double, 4>, double>> variant(const std::vector<Star>& stars, const std::array<size_t, 4>& ids,
                                                                       double flip) {
  const Star &a = stars[ids[0]], &b = stars[ids[1]], &c = stars[ids[2]], &d = stars[ids[3]];
  double dx = b.x - a.x, dy = b.y - a.y;
  double len = std::hypot(dx, dy);
  if (len < 2.0) return std::nullopt;
  auto norm = [&](const Star& s) {
    double x = s.x - a.x, y = s.y - a.y;
    return std::array<double, 2>{(x * dx + y * dy) / (len * len), 0.5 + flip * (y * dx - x * dy) / (2.0 * len * len)};
  };
  auto u = norm(c);
  auto v = norm(d);
  return std::make_pair(std::array<double, 4>{u[0], u[1], v[0], v[1]}, len);
}

std::vector<Pattern> variants(const std::vector<Star>& stars, std::array<size_t, 4> ids) {
  struct Edge {
    double len;
    size_t i, j;
  };
  std::vector<Edge> edges;
  for (size_t i = 0; i < 4; ++i)
    for (size_t j = i + 1; j < 4; ++j) edges.push_back({distance(stars[ids[i]], stars[ids[j]]), i, j});
  std::stable_sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) { return total_cmp(b.len, a.len) < 0; });
  double mx = edges[0].len;
  std::vector<Pattern> out;
  for (size_t e = 0; e < 3 && e < edges.size(); ++e) {
    double len = edges[e].len;
    size_t i = edges[e].i, j = edges[e].j;
    if (len < mx * 0.98) break;
    std::vector<size_t> rest;
    for (size_t x = 0; x < 4; ++x)
      if (x != i && x != j) rest.push_back(x);
    for (bool swap : {false, true})
      for (bool rev : {false, true})
        for (double sign : {-1.0, 1.0}) {
          size_t a = rev ? j : i, b = rev ? i : j;
          size_t c = swap ? rest[1] : rest[0], d = swap ? rest[0] : rest[1];
          std::array<size_t, 4> ix = {ids[a], ids[b], ids[c], ids[d]};
          auto vr = variant(stars, ix, sign);
          if (!vr) continue;
          const auto& code = vr->first;
          double baseline = vr->second;
          bool bad = false;
          for (double v : code)
            if (v < 0.0 || v > 1.0) bad = true;
          if (bad) continue;
          double area = fmin_r(std::fabs(code[1] - 0.5) + std::fabs(code[3] - 0.5), 0.5);
          if (area < 0.015) continue;
          double sigma = 0.0;
          for (size_t k : ix) sigma += stars[k].sigma;
          sigma /= 4.0;
          double snr = INF;
          for (size_t k : ix) snr = fmin_r(snr, stars[k].snr);
          snr = fmax_r(snr, 1.0);
          size_t sat = 0;
          for (size_t k : ix) sat += size_t(stars[k].saturated);
          Pattern p;
          p.ids = ix;
          p.code = code;
          p.baseline = baseline;
          p.features = {fmin_r(std::log(snr) / 5.0, 2.0), fmin_r(sigma / baseline * 100.0, 2.0), area, std::log(baseline) / 10.0, 0.0,
                        double(sat) / 4.0};
          p.quality = std::log(snr) * area / (0.01 + sigma / baseline * 100.0);
          out.push_back(p);
        }
  }
  return out;
}

std::optional<Pattern> canonical(const std::vector<Star>& stars, std::array<size_t, 4> ids) {
  auto v = variants(stars, ids);
  if (v.empty()) return std::nullopt;
  size_t best = 0;
  for (size_t i = 1; i < v.size(); ++i)
    if (std::lexicographical_compare(v[i].code.begin(), v[i].code.end(), v[best].code.begin(), v[best].code.end())) best = i;
  return v[best];
}

std::vector<size_t> balanced(const std::vector<Star>& stars, uint32_t width, uint32_t height) {
  std::vector<std::vector<size_t>> cells(16);
  for (size_t i = 0; i < stars.size(); ++i) {
    const Star& s = stars[i];
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.flux)) continue;
    size_t x = std::min<size_t>(sat_cast<size_t>(std::floor(s.x / double(std::max<uint32_t>(width, 1)) * 4.0)), 3);
    size_t y = std::min<size_t>(sat_cast<size_t>(std::floor(s.y / double(std::max<uint32_t>(height, 1)) * 4.0)), 3);
    cells[y * 4 + x].push_back(i);
  }
  for (auto& c : cells)
    std::stable_sort(c.begin(), c.end(), [&](size_t a, size_t b) { return total_cmp(stars[b].flux, stars[a].flux) < 0; });
  std::vector<size_t> out;
  for (size_t depth = 0;; ++depth) {
    std::vector<size_t> layer;
    for (auto& c : cells)
      if (depth < c.size()) layer.push_back(c[depth]);
    if (layer.empty()) break;
    std::stable_sort(layer.begin(), layer.end(), [&](size_t a, size_t b) { return total_cmp(stars[b].flux, stars[a].flux) < 0; });
    out.insert(out.end(), layer.begin(), layer.end());
  }
  return out;
}

std::vector<Pattern> enumerate(const std::vector<Star>& stars, uint32_t width, uint32_t height) {
  Emitter e(stars, width, height);
  std::vector<Pattern> out;
  while (auto p = e.next()) out.push_back(*p);
  return out;
}

Emitter::Emitter(const std::vector<Star>& stars, uint32_t width, uint32_t height) : stars_(&stars), ranked_(balanced(stars, width, height)) {}

void Emitter::expand(size_t tier) {
  const auto& stars = *stars_;
  size_t take = std::min(ranked_.size(), tier);
  std::vector<size_t> selected(ranked_.begin(), ranked_.begin() + long(take));
  std::vector<Pattern> batch;
  for (size_t anchor : selected) {
    std::vector<size_t> neighbors;
    for (size_t i : selected)
      if (i != anchor) neighbors.push_back(i);
    std::stable_sort(neighbors.begin(), neighbors.end(), [&](size_t a, size_t b) {
      return total_cmp(distance(stars[anchor], stars[a]), distance(stars[anchor], stars[b])) < 0;
    });
    if (neighbors.size() > 8) neighbors.resize(8);
    for (size_t a = 0; a < neighbors.size(); ++a)
      for (size_t b = a + 1; b < neighbors.size(); ++b)
        for (size_t c = b + 1; c < neighbors.size(); ++c) {
          std::array<size_t, 4> ids = {anchor, neighbors[a], neighbors[b], neighbors[c]};
          std::sort(ids.begin(), ids.end());
          if (!seen_.insert(ids).second) continue;
          if (auto p = canonical(stars, ids)) batch.push_back(*p);
        }
  }
  std::stable_sort(batch.begin(), batch.end(), [](const Pattern& a, const Pattern& b) { return total_cmp(b.quality, a.quality) < 0; });
  pending_ = std::move(batch);
  pending_pos_ = 0;
}

std::optional<Pattern> Emitter::next() {
  static const size_t TIERS[4] = {12, 24, 48, 96};
  for (;;) {
    if (pending_pos_ < pending_.size()) return pending_[pending_pos_++];
    if (tier_ >= 4) return std::nullopt;
    size_t size = TIERS[tier_];
    tier_ += 1;
    expand(size);
  }
}

size_t bin(const std::array<double, 4>& code) {
  size_t v = 0;
  for (double x : code) v = (v << 5) | sat_cast<size_t>(clampd(x, 0.0, 0.999999) * 32.0);
  return v;
}

std::array<uint16_t, 4> quantize(const std::array<double, 4>& code) {
  std::array<uint16_t, 4> out;
  for (size_t i = 0; i < 4; ++i) out[i] = sat_cast<uint16_t>(std::round(clampd(code[i], 0.0, 1.0) * 65535.0));
  return out;
}

}  // namespace skyshot::patterns
