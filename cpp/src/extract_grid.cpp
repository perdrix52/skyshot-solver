// Background model: tiles, robust noise, bicubic interpolation. Part of the port of src/extract.rs.
#include <algorithm>

#include "extract_internal.hpp"
#include "skyshot/parallel.hpp"

namespace skyshot::extract_detail {

float noise_floor(float level, float lsb) {
  return fmax_r(fmax_r(0.5f * lsb, std::fabs(level) * 1e-6f), F32_MIN_POSITIVE);
}

float Plane::bilinear(float x, float y) const {
  if (!(x >= 0.0f && y >= 0.0f)) return std::numeric_limits<float>::quiet_NaN();
  size_t x0 = sat_cast<size_t>(x), y0 = sat_cast<size_t>(y);
  if (x0 + 1 >= w || y0 + 1 >= h) {
    if (x0 < w && y0 < h && float(x0) == x && float(y0) == y) return data[y0 * w + x0];
    return std::numeric_limits<float>::quiet_NaN();
  }
  float tx = x - float(x0), ty = y - float(y0);
  size_t i = y0 * w + x0;
  const float* d = data;
  float top = d[i] + (d[i + 1] - d[i]) * tx;
  float bottom = d[i + w] + (d[i + w + 1] - d[i + w]) * tx;
  return top + (bottom - top) * ty;
}

// ---------- Axis ----------
Axis Axis::make(size_t len, size_t tile) {
  Axis a;
  a.n = std::max<size_t>((len + tile - 1) / tile, 1);
  a.s = float(len) / float(a.n);
  a.len = len;
  return a;
}

std::pair<size_t, size_t> Axis::range(size_t t) const {
  auto edge = [&](size_t k) { return std::min(sat_cast<size_t>(std::round(float(k) * s)), len); };
  return {edge(t), t + 1 == n ? len : edge(t + 1)};
}

size_t Axis::index(size_t c) const { return std::min(sat_cast<size_t>(float(c) / s), n - 1); }

std::pair<size_t, float> Axis::locate(float c) const {
  if (n == 1) return {0, 0.0f};
  float f = (c + 0.5f) / s - 0.5f;
  size_t i = std::min(sat_cast<size_t>(fmax_r(std::floor(f), 0.0f)), n - 2);
  return {i, f - float(i)};
}

Axis::Lin Axis::linear(float c) const {
  auto [i, t] = locate(c);
  if (n == 1) return {0, 0, 0.0f};
  return {i, i + 1, clampf(t, 0.0f, 1.0f)};
}

std::array<std::pair<uint32_t, float>, 6> Axis::cubic(float c) const {
  std::array<std::pair<uint32_t, float>, 6> out{};
  if (n == 1) {
    out[0] = {0, 1.0f};
    return out;
  }
  auto [i, t] = locate(c);
  float t2 = t * t, t3 = t * t * t;
  float w[4] = {0.5f * (-t3 + 2.0f * t2 - t), 0.5f * (3.0f * t3 - 5.0f * t2 + 2.0f), 0.5f * (-3.0f * t3 + 4.0f * t2 + t),
                0.5f * (t3 - t2)};
  out[1] = {uint32_t(i), w[1]};
  out[2] = {uint32_t(i) + 1, w[2]};
  if (i == 0) {
    out[0] = {0, 2.0f * w[0]};
    out[4] = {1, -w[0]};
  } else {
    out[0] = {uint32_t(i) - 1, w[0]};
  }
  if (i + 2 >= n) {
    out[3] = {uint32_t(n) - 1, 2.0f * w[3]};
    out[5] = {uint32_t(n) - 2, -w[3]};
  } else {
    out[3] = {uint32_t(i) + 2, w[3]};
  }
  return out;
}

// ---------- tile statistics ----------
static float z_upper(float p_in) {
  double p = clampd(double(p_in), 1e-9, 1.0 - 1e-9);
  double q, sign;
  if (p > 0.5) {
    q = 1.0 - p;
    sign = -1.0;
  } else {
    q = p;
    sign = 1.0;
  }
  double t = std::sqrt(-2.0 * std::log(q));
  double z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
  return float(sign * z);
}

static std::optional<float> floor_censored_sigma(std::vector<uint32_t>& keys, uint32_t lo, size_t at_lo, float lsb) {
  size_t n = keys.size();
  size_t above = n - at_lo;
  if (above < 8) return std::nullopt;
  float c = unkey(lo);
  size_t i_q = at_lo + above / 2;
  std::nth_element(keys.begin(), keys.begin() + long(i_q), keys.end());
  uint32_t kq = keys[i_q];
  uint32_t v1k = kq;
  {
    bool found = false;
    uint32_t mn = 0;
    for (size_t i = 0; i < i_q; ++i) {
      if (keys[i] > lo && (!found || keys[i] < mn)) {
        mn = keys[i];
        found = true;
      }
    }
    if (found) v1k = mn;
  }
  float v1 = unkey(v1k);
  float vq = unkey(kq);
  if (lsb > 0.0f) {
    size_t below = 0, equal = 0;
    for (uint32_t k : keys) {
      below += size_t(k < kq);
      equal += size_t(k == kq);
    }
    float t = clampf((float(i_q) + 0.5f - float(below)) / float(equal), 0.0f, 1.0f);
    vq += lsb * (t - 0.5f);
  }
  if (v1 - c > fmax_r(1.5f * lsb, 0.25f * (vq - c))) return std::nullopt;
  float p0 = float(at_lo) / float(n), q = (float(i_q) + 0.5f) / float(n);
  float cb = c + 0.5f * lsb;
  float den = z_upper(1.0f - q) - z_upper(1.0f - p0);
  if (den > 0.3f && vq > cb) return (vq - cb) / den;
  return std::nullopt;
}

static std::optional<TileLevel> tile_level(std::vector<uint32_t>& keys, float lsb) {
  size_t n = keys.size();
  if (n < 12) return std::nullopt;
  size_t mid = n / 2;
  std::nth_element(keys.begin(), keys.begin() + long(mid), keys.end());
  uint32_t mk = keys[mid];
  float m = unkey(mk);
  size_t below = 0, equal = 0;
  uint32_t lo = UINT32_MAX;
  size_t at_lo = 0;
  for (uint32_t k : keys) {
    below += size_t(k < mk);
    equal += size_t(k == mk);
    if (k < lo) {
      lo = k;
      at_lo = 1;
    } else if (k == lo) {
      at_lo += 1;
    }
  }
  if (2 * equal < n) {
    float bg = m;
    if (lsb > 0.0f && equal > 1) {
      float t = clampf((float(n) * 0.5f - float(below)) / float(equal), 0.0f, 1.0f);
      bg = m + lsb * (t - 0.5f);
    }
    std::optional<float> floor;
    if (4 * at_lo > n) floor = floor_censored_sigma(keys, lo, at_lo, lsb);
    return TileLevel{bg, std::nullopt, floor.value_or(0.0f)};
  }
  size_t above = n - below - equal;
  float sigma;
  if (above >= 8) {
    float p = float(above) / float(n);
    std::vector<uint32_t> upper;
    for (uint32_t k : keys)
      if (k > mk) upper.push_back(k);
    std::nth_element(upper.begin(), upper.begin() + long(above / 2), upper.end());
    float vq = unkey(upper[above / 2]);
    sigma = (vq - m) / fmax_r(z_upper(p * 0.5f) - z_upper(p), 0.3f);
  } else {
    sigma = 0.0f;
  }
  return TileLevel{m, fmax_r(sigma, noise_floor(m, lsb)), 0.0f};
}

static void mask_clipped_tiles(std::vector<std::optional<TileLevel>>& levels, const std::vector<float>& top) {
  std::vector<float> sky;
  for (auto& l : levels)
    if (l) sky.push_back(l->bg);
  if (sky.empty()) return;
  size_t mid = sky.size() / 2;
  std::nth_element(sky.begin(), sky.begin() + long(mid), sky.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
  float skyv = sky[mid];
  float peak = -INF;
  for (float v : top)
    if (std::isfinite(v)) peak = fmax_r(peak, v);
  if (peak <= skyv) return;
  std::vector<size_t> clipped;
  for (size_t t = 0; t < levels.size(); ++t) {
    if (!levels[t]) continue;
    float mx = top[t];
    float m = levels[t]->bg;
    if (mx - skyv >= 0.5f * (peak - skyv) && mx - m <= 0.03f * (mx - skyv)) clipped.push_back(t);
  }
  size_t valid = 0;
  for (auto& l : levels) valid += size_t(l.has_value());
  if (clipped.empty() || clipped.size() > std::max<size_t>(levels.size() * 3 / 100, 4) || 2 * clipped.size() >= valid) return;
  for (size_t t : clipped) levels[t].reset();
}

static std::optional<float> mad0(std::vector<uint32_t>& keys) {
  size_t n = keys.size();
  if (n < 12) return std::nullopt;
  std::nth_element(keys.begin(), keys.begin() + long(n / 2), keys.end());
  return 1.4826f * unkey(keys[n / 2]);
}

template <class F>
static void for_tile_samples(size_t x0, size_t y0, size_t x1, size_t y1, size_t row_step, F&& f) {
  size_t r = 0;
  for (size_t y = y0; y < y1; y += row_step, ++r)
    for (size_t x = x0 + (r & 1); x < x1; x += 2) f(x, y);
}

struct TileStats {
  float bg;
  std::optional<float> deg;
  float floor;
  bool filled;
};

std::optional<Grid> Grid::build(const Plane& plane, const std::vector<float>& native, size_t w, size_t b, float lsb) {
  size_t tile = std::max<size_t>(TILE_NATIVE / b, 16);
  Axis ax = Axis::make(plane.w, tile), ay = Axis::make(plane.h, tile);
  size_t nx = ax.n, ny = ay.n;
  const size_t row_step = 2;
  float lsb_b = lsb / float(b * b);
  auto bounds = [&](size_t t) {
    auto xr = ax.range(t % nx);
    auto yr = ay.range(t / nx);
    return std::array<size_t, 4>{xr.first, yr.first, xr.second, yr.second};
  };
  std::vector<std::optional<TileLevel>> first(nx * ny);
  std::vector<float> top(nx * ny);
  parallel_for(nx * ny, [&](size_t t) {
    thread_local std::vector<uint32_t> keys;
    keys.clear();
    auto bd = bounds(t);
    for_tile_samples(bd[0], bd[1], bd[2], bd[3], row_step, [&](size_t x, size_t y) {
      float v = plane.data[y * plane.w + x];
      if (std::isfinite(v)) keys.push_back(key_of(v));
    });
    float tp = std::numeric_limits<float>::quiet_NaN();
    if (!keys.empty()) tp = unkey(*std::max_element(keys.begin(), keys.end()));
    top[t] = tp;
    first[t] = tile_level(keys, lsb_b);
  });
  bool any = false;
  for (auto& f : first) any = any || f.has_value();
  if (!any) return std::nullopt;
  mask_clipped_tiles(first, top);
  std::vector<std::optional<TileStats>> stats(nx * ny);
  for (size_t t = 0; t < nx * ny; ++t)
    if (first[t]) stats[t] = TileStats{first[t]->bg, first[t]->degenerate_sigma, first[t]->floor, false};
  auto any_none = [&] {
    for (auto& s : stats)
      if (!s) return true;
    return false;
  };
  while (any_none()) {
    auto prev = stats;
    for (size_t t = 0; t < nx * ny; ++t) {
      if (prev[t]) continue;
      int64_t tx = int64_t(t % nx), ty = int64_t(t / nx);
      float s = 0.0f, n = 0.0f;
      const int dxs[4] = {-1, 1, 0, 0}, dys[4] = {0, 0, -1, 1};
      for (int k = 0; k < 4; ++k) {
        int64_t x = tx + dxs[k], y = ty + dys[k];
        if (x >= 0 && y >= 0 && x < int64_t(nx) && y < int64_t(ny) && prev[size_t(y) * nx + size_t(x)]) {
          s += prev[size_t(y) * nx + size_t(x)]->bg;
          n += 1.0f;
        }
      }
      if (n > 0.0f) stats[t] = TileStats{s / n, std::nullopt, 0.0f, true};
    }
  }
  std::vector<TileStats> st(nx * ny);
  for (size_t t = 0; t < nx * ny; ++t) st[t] = *stats[t];
  std::vector<float> medians(nx * ny);
  for (size_t t = 0; t < nx * ny; ++t) medians[t] = st[t].bg;
  Grid g;
  g.ax = ax;
  g.ay = ay;
  g.nx = nx;
  g.ny = ny;
  g.b = b;
  g.bg.resize(nx * ny);
  for (size_t t = 0; t < nx * ny; ++t) {
    size_t tx = t % nx, ty = t / nx;
    auto lap = [&](std::optional<size_t> a, std::optional<size_t> bb) -> float {
      if (a && bb) return medians[*a] + medians[*bb] - 2.0f * medians[t];
      return 0.0f;
    };
    float lx = lap(tx >= 1 ? std::optional<size_t>(ty * nx + tx - 1) : std::nullopt,
                   tx + 1 < nx ? std::optional<size_t>(ty * nx + tx + 1) : std::nullopt);
    float ly = lap(ty >= 1 ? std::optional<size_t>((ty - 1) * nx + tx) : std::nullopt,
                   ty + 1 < ny ? std::optional<size_t>((ty + 1) * nx + tx) : std::nullopt);
    g.bg[t] = medians[t] - (lx + ly) / 24.0f;
  }
  g.sig_b.assign(nx * ny, 0.0f);
  g.sig_n.assign(nx * ny, 0.0f);
  g.xcub.resize(plane.w);
  g.xlin.resize(plane.w);
  for (size_t x = 0; x < plane.w; ++x) {
    g.xcub[x] = ax.cubic(float(x));
    auto l = ax.linear(float(x));
    g.xlin[x] = {uint32_t(l.i0), uint32_t(l.i1), l.t};
  }
  struct Scratch {
    std::vector<uint32_t> keys;
    std::vector<float> colv;
  };
  auto residual_sigma = [&](size_t t, bool native_px, Scratch& sc) -> std::optional<float> {
    size_t tx = t % nx;
    size_t c0 = tx >= 2 ? tx - 2 : 0, c1 = std::min(tx + 3, nx);
    sc.colv.resize(nx, 0.0f);
    sc.keys.clear();
    size_t last_y = SIZE_MAX;
    auto bd = bounds(t);
    for_tile_samples(bd[0], bd[1], bd[2], bd[3], row_step, [&](size_t x, size_t y) {
      if (y != last_y) {
        g.row_cubic_range(float(y), c0, c1, sc.colv.data());
        last_y = y;
      }
      float v = native_px ? native[y * b * w + x * b] : plane.data[y * plane.w + x];
      if (std::isfinite(v)) sc.keys.push_back(key_of(std::fabs(v - g.col_cubic(x, sc.colv.data()))));
    });
    return mad0(sc.keys);
  };
  std::vector<float> sig_b(nx * ny);
  parallel_for(nx * ny, [&](size_t t) {
    thread_local Scratch sc;
    if (st[t].deg || st[t].filled) {
      sig_b[t] = st[t].deg.value_or(std::numeric_limits<float>::quiet_NaN());
      return;
    }
    auto s = residual_sigma(t, false, sc);
    sig_b[t] = s ? fmax_r(fmax_r(*s, st[t].floor), noise_floor(st[t].bg, lsb_b)) : std::numeric_limits<float>::quiet_NaN();
  });
  float ratio;
  if (b == 1) {
    ratio = 1.0f;
  } else {
    size_t step = std::max<size_t>(nx * ny / 64, 1);
    std::vector<size_t> sel;
    for (size_t t = 0; t < nx * ny; t += step) sel.push_back(t);
    std::vector<std::optional<float>> rr(sel.size());
    parallel_for(sel.size(), [&](size_t k) {
      size_t t = sel[k];
      if (!(!st[t].deg && st[t].floor == 0.0f && !st[t].filled && std::isfinite(sig_b[t]))) return;
      Scratch sc;
      auto s = residual_sigma(t, true, sc);
      if (!s) return;
      float r = *s / sig_b[t];
      if (std::isfinite(r) && r > 0.0f) rr[k] = r;
    });
    std::vector<float> r;
    for (auto& v : rr)
      if (v) r.push_back(*v);
    if (r.empty()) {
      ratio = float(b);
    } else {
      size_t mid = r.size() / 2;
      std::nth_element(r.begin(), r.begin() + long(mid), r.end(), [](float a, float c) { return total_cmp(a, c) < 0; });
      ratio = r[mid];
    }
  }
  std::vector<float> finite;
  for (float s : sig_b)
    if (std::isfinite(s)) finite.push_back(s);
  float typical;
  if (finite.empty()) {
    typical = noise_floor(0.0f, lsb_b);
  } else {
    size_t mid = finite.size() / 2;
    std::nth_element(finite.begin(), finite.begin() + long(mid), finite.end(), [](float a, float c) { return total_cmp(a, c) < 0; });
    typical = finite[mid];
  }
  for (size_t t = 0; t < nx * ny; ++t) {
    g.sig_b[t] = std::isfinite(sig_b[t]) ? sig_b[t] : typical;
    g.sig_n[t] = fmax_r(g.sig_b[t] * ratio, noise_floor(st[t].bg, lsb));
  }
  g.typical_sig_n = typical * ratio;
  return g;
}

void Grid::row_cubic_range(float yb, size_t c0, size_t c1, float* out) const {
  auto wy = ay.cubic(yb);
  for (size_t t = c0; t < c1 && t < nx; ++t) {
    float s = -0.0f;
    for (auto& rw : wy) s += rw.second * bg[size_t(rw.first) * nx + t];
    out[t] = s;
  }
}
void Grid::row_cubic(float yb, float* out) const {
  auto wy = ay.cubic(yb);
  for (size_t t = 0; t < nx; ++t) {
    float s = -0.0f;
    for (auto& rw : wy) s += rw.second * bg[size_t(rw.first) * nx + t];
    out[t] = s;
  }
}
float Grid::col_cubic(size_t xb, const float* colv) const {
  float s = -0.0f;
  for (auto& cw : xcub[xb]) s += cw.second * colv[cw.first];
  return s;
}
void Grid::row_sigma(float yb, float* out) const {
  auto l = ay.linear(yb);
  for (size_t t = 0; t < nx; ++t) {
    float a = sig_b[l.i0 * nx + t], c = sig_b[l.i1 * nx + t];
    out[t] = a + (c - a) * l.t;
  }
}
float Grid::col_linear(size_t xb, const float* colv) const {
  const XLin& l = xlin[xb];
  float a = colv[l.i0], c = colv[l.i1];
  return a + (c - a) * l.t;
}
Grid::At Grid::at(float xb, float yb) const {
  auto wx = ax.cubic(xb), wy = ay.cubic(yb);
  float bgv = 0.0f;
  for (auto& r : wy)
    for (auto& c : wx) bgv += r.second * c.second * bg[size_t(r.first) * nx + c.first];
  auto lx = ax.linear(xb);
  auto ly = ay.linear(yb);
  auto lerp = [&](const std::vector<float>& v) {
    float topv = v[ly.i0 * nx + lx.i0] + (v[ly.i0 * nx + lx.i1] - v[ly.i0 * nx + lx.i0]) * lx.t;
    float bot = v[ly.i1 * nx + lx.i0] + (v[ly.i1 * nx + lx.i1] - v[ly.i1 * nx + lx.i0]) * lx.t;
    return topv + (bot - topv) * ly.t;
  };
  return {bgv, lerp(sig_b), lerp(sig_n)};
}
Grid::At Grid::at_native(float x, float y) const {
  float bb = float(b);
  return at((x + 0.5f) / bb - 0.5f, (y + 0.5f) / bb - 0.5f);
}
size_t Grid::tile_index(size_t xb, size_t yb) const { return ay.index(yb) * nx + ax.index(xb); }

}  // namespace skyshot::extract_detail
