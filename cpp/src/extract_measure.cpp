// Per-source measurement. Part of the port of src/extract.rs.
#include <algorithm>
#include <cstdlib>

#include "extract_internal.hpp"

namespace skyshot::extract_detail {

namespace {
constexpr float NANF = std::numeric_limits<float>::quiet_NaN();
inline int64_t iround(double v) { return sat_cast<int64_t>(std::round(v)); }
}  // namespace

// ---------- Moments ----------
static double truncation_gain(double f) {
  if (f <= 1e-4) return 1.0;
  f = std::min(f, 0.8);
  double u = -std::log(f);
  double num = 1.0 - f * (1.0 + u) - 0.5 * f * u * u;
  double den = 1.0 - f - f * u;
  return clampd(num / den, 0.1, 1.0);
}

Moments::Shape Moments::shape(double f) const {
  double a = xx - yy;
  double disc = std::sqrt(a * a + 4.0 * xy * xy);
  double minor = fmax_r((xx + yy - disc) / 2.0, 0.02);
  double major = fmax_r((xx + yy + disc) / 2.0, minor);
  double fwhm = 2.3548 * std::sqrt((major + minor) / 2.0 / truncation_gain(f));
  double pa = rem_euclid(to_degrees(0.5 * std::atan2(2.0 * xy, xx - yy)), 180.0);
  return {fwhm, std::sqrt(major / minor), pa};
}

std::pair<double, double> Moments::axes() const {
  double a = xx - yy;
  double disc = std::sqrt(a * a + 4.0 * xy * xy);
  double minor = fmax_r((xx + yy - disc) / 2.0, 0.0);
  double major = fmax_r((xx + yy + disc) / 2.0, minor);
  return {major, minor};
}

// ---------- Det ----------
Star Det::to_star(float trail_len) const {
  double sigma = fmax_r(fwhm / (2.3548 * fmax_r(snr, 1e-3)), 0.05);
  if (flags & star_flags::SATURATED) sigma = fmax_r(sigma, 0.3);
  if (flags & star_flags::BLENDED) sigma = fmax_r(sigma, 0.15);
  if (flags & star_flags::NEAR_EDGE) sigma = fmax_r(sigma, 0.1);
  if (flags & star_flags::TRAILED) sigma = fmax_r(sigma, 0.1 * double(trail_len));
  Star s;
  s.x = x;
  s.y = y;
  s.flux = flux;
  s.sigma = fmin_r(sigma, 2.0);
  s.snr = snr;
  s.fwhm = fwhm;
  s.elongation = elong;
  s.saturated = (flags & star_flags::SATURATED) != 0;
  s.peak = double(h);
  s.pa = pa;
  s.flags = flags;
  return s;
}

// ---------- aperture ----------
ApertureResult aperture(const float* img, size_t w, size_t h, double cx, double cy, double r, const Bg& bg) {
  int64_t ri = sat_cast<int64_t>(std::ceil(r));
  int64_t iy = iround(cy);
  double flux = 0.0;
  size_t n = 0;
  bool clipped = false;
  double r2 = r * r;
  for (int64_t y = iy - ri; y <= iy + ri; ++y) {
    double dy = double(y) - cy;
    double dy2 = dy * dy;
    if (dy2 > r2) continue;
    double half = std::sqrt(r2 - dy2);
    int64_t xa = sat_cast<int64_t>(std::ceil(cx - half)), xb = sat_cast<int64_t>(std::floor(cx + half));
    if (y < 0 || y >= int64_t(h) || xa < 0 || xb >= int64_t(w)) clipped = true;
    if (y < 0 || y >= int64_t(h)) continue;
    size_t xa_u = size_t(std::max<int64_t>(xa, 0));
    int64_t xb_c = std::min<int64_t>(xb, int64_t(w) - 1);
    if (xb_c < int64_t(xa_u)) continue;
    const float* row = img + size_t(y) * w;
    float row_bg = bg.at(0.0f, float(y));
    float sum = 0.0f, sx = 0.0f;
    size_t cnt = 0;
    for (size_t k = 0; xa_u + k <= size_t(xb_c); ++k) {
      float v = row[xa_u + k];
      if (std::isfinite(v)) {
        sum += v;
        sx += float(k);
        cnt += 1;
      } else {
        clipped = true;
      }
    }
    float gx_sum = bg.gx * (sx + float(cnt * xa_u));
    flux += double((sum - row_bg * float(cnt)) - gx_sum);
    n += cnt;
  }
  return {flux, n, clipped};
}

// ---------- Ctx ----------
std::array<float, 2> Ctx::to_native(float xb, float yb) const {
  float bb = float(b);
  return {(xb + 0.5f) * bb - 0.5f, (yb + 0.5f) * bb - 0.5f};
}

std::pair<Bg, float> Ctx::plane_bg(size_t pxl, size_t pyl) const {
  float x = float(pxl), y = float(pyl);
  float d = fmax_r(grid->ax.s * float(b) * 0.25f, 2.0f);
  auto c0 = grid->at_native(x, y);
  float gx = (grid->at_native(x + d, y).bg - grid->at_native(x - d, y).bg) / (2.0f * d);
  float gy = (grid->at_native(x, y + d).bg - grid->at_native(x, y - d).bg) / (2.0f * d);
  Bg bg;
  bg.c = c0.bg;
  bg.gx = gx;
  bg.gy = gy;
  bg.x0 = x;
  bg.y0 = y;
  return {bg, c0.sig_n};
}

std::optional<std::pair<size_t, size_t>> Ctx::native_peak(const Cand& c) const {
  size_t x = c.x, y = c.y;
  if (b == 1) return std::make_pair(x, y);
  size_t x0 = x * b >= 1 ? x * b - 1 : 0, y0 = y * b >= 1 ? y * b - 1 : 0;
  size_t x1 = std::min(x * b + b, w - 1), y1 = std::min(y * b + b, h - 1);
  float best = -INF;
  size_t bx = 0, by = 0;
  for (size_t yy = y0; yy <= y1; ++yy)
    for (size_t xx = x0; xx <= x1; ++xx) {
      float v = px(xx, yy);
      if (v > best) {
        best = v;
        bx = xx;
        by = yy;
      }
    }
  if (best > -INF) return std::make_pair(bx, by);
  return std::nullopt;
}

bool Ctx::is_ridge(const Cand& c) const {
  auto l = contrast(c);
  if (!l) return false;
  if (l->contrast >= 0.6f * k * c.sig && l->contrast >= 0.3f * l->height) return false;
  return !(l->floor >= 0.5f * l->height && l->height >= 30.0f * c.sig);
}

bool Ctx::is_compact_star(const Cand& c) const {
  auto l = contrast(c);
  return l && l->contrast >= k * c.sig && l->contrast >= 0.6f * l->height;
}

std::optional<Contrast> Ctx::contrast(const Cand& c) const {
  float hgt = c.v - c.bg;
  float bb = float(b);
  float ff = fmax_r(fwhm_frame, 2.0f) / bb;
  float d = fmax_r(1.5f * ff, 1.5f);
  float theta0 = 0.0f, half = 0.0f;
  if (trail) {
    theta0 = trail->pa;
    half = 0.5f * trail->length / bb;
  }
  float x = float(c.x), y = float(c.y);
  auto probe = [&](float co, float s, float r) -> float {
    float low = INF;
    const float ts[3] = {1.0f, 1.5f, 2.0f};
    for (float t : ts) {
      float v = plane->bilinear(x + t * r * co, y + t * r * s);
      if (std::isfinite(v)) low = fmin_r(low, v);
      else if (t == 1.0f) return NANF;
    }
    return low;
  };
  float amax = -INF, amax2 = -INF, amin = INF;
  const float PIF = 3.14159274f;  // std::f32::consts::PI
  for (int k2 = 0; k2 < 16; ++k2) {
    float phi = float(k2) * PIF / 16.0f;
    float ang = theta0 + phi;
    float s = std::sin(ang), co = std::cos(ang);
    float r;
    if (half > 0.0f && d * std::fabs(std::sin(phi)) < ff) r = fmax_r(d, (half + ff) / fmax_r(std::fabs(std::cos(phi)), 0.3f));
    else r = d;
    float p = probe(co, s, r), q = probe(-co, -s, r);
    float a;
    bool pf = std::isfinite(p), qf = std::isfinite(q);
    if (pf && qf) a = fmin_r(p, q) - c.bg;
    else if (pf) a = p - c.bg;
    else if (qf) a = q - c.bg;
    else continue;
    if (a > amax) {
      amax2 = amax;
      amax = a;
    } else if (a > amax2) {
      amax2 = a;
    }
    amin = fmin_r(amin, a);
  }
  if (amax2 > -INF) return Contrast{hgt, hgt - amax2, amin};
  return std::nullopt;
}

Moments Ctx::moments(double cx, double cy, size_t hw_u, const Bg& bg, float thr) const {
  int64_t ix = iround(cx), iy = iround(cy);
  int64_t hw = int64_t(hw_u);
  Moments m;
  double s = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (int64_t yy = iy - hw; yy <= iy + hw; ++yy) {
    if (yy < 0 || yy >= int64_t(h)) {
      m.clipped = true;
      continue;
    }
    float row_bg = bg.at(0.0f, float(yy)) + thr;
    for (int64_t xx = ix - hw; xx <= ix + hw; ++xx) {
      if (xx < 0 || xx >= int64_t(w)) {
        m.clipped = true;
        continue;
      }
      float v = px(size_t(xx), size_t(yy));
      if (!std::isfinite(v)) {
        m.clipped = true;
        continue;
      }
      double f = double(fmax_r(v - row_bg - bg.gx * float(xx), 0.0f));
      if (f > 0.0) {
        double dx = double(xx) - cx, dy = double(yy) - cy;
        s += f;
        sx += f * dx;
        sy += f * dy;
        sxx += f * dx * dx;
        syy += f * dy * dy;
        sxy += f * dx * dy;
      }
    }
  }
  if (s > 0.0) {
    double mx = sx / s, my = sy / s;
    m.sum = s;
    m.x = cx + mx;
    m.y = cy + my;
    m.xx = sxx / s - mx * mx;
    m.yy = syy / s - my * my;
    m.xy = sxy / s - mx * my;
  }
  return m;
}

std::optional<std::pair<double, double>> Ctx::xwin(double x0, double y0, size_t hw, double s, const Bg& bg) const {
  double x = x0, y = y0;
  size_t n = 2 * hw + 1;
  float wx[31] = {}, wy[31] = {};
  float inv = float(-0.5 / (s * s));
  for (int it = 0; it < 10; ++it) {
    int64_t ix = iround(x) - int64_t(hw), iy = iround(y) - int64_t(hw);
    if (ix < 0 || iy < 0 || size_t(ix) + n > w || size_t(iy) + n > h) return std::nullopt;
    float fx = float(x - double(ix)), fy = float(y - double(iy));
    for (size_t k = 0; k < n; ++k) {
      float dx = float(k) - fx, dy = float(k) - fy;
      wx[k] = std::exp(dx * dx * inv);
      wy[k] = std::exp(dy * dy * inv);
    }
    float sw = 0.0f, sx = 0.0f, sy = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      size_t row = (size_t(iy) + j) * w + size_t(ix);
      float row_bg = bg.at(0.0f, float(iy + int64_t(j)));
      float rw = 0.0f, rx = 0.0f;
      for (size_t i = 0; i < n; ++i) {
        float v = img[row + i];
        if (!std::isfinite(v)) continue;
        float bb = row_bg + bg.gx * float(ix + int64_t(i));
        float f = wx[i] * (v - bb);
        rw += f;
        rx += f * (float(i) - fx);
      }
      sw += wy[j] * rw;
      sx += wy[j] * rx;
      sy += wy[j] * rw * (float(j) - fy);
    }
    double swd = double(sw), sxd = double(sx), syd = double(sy);
    if (swd <= 0.0) return std::nullopt;
    double dx = clampd(2.0 * sxd / swd, -1.0, 1.0);
    double dy = clampd(2.0 * syd / swd, -1.0, 1.0);
    x += dx;
    y += dy;
    if (std::fabs(dx) < 1e-3 && std::fabs(dy) < 1e-3) break;
  }
  if (dist(x - x0, y - y0) <= fmax_r(double(hw), 2.0)) return std::make_pair(x, y);
  return std::nullopt;
}

std::optional<float> Ctx::annulus(double cx, double cy, double r_in, double r_out) const {
  int64_t step = std::max<int64_t>(sat_cast<int64_t>(std::floor(std::sqrt(r_out * r_out * 3.2 / 80.0))), 1);
  int64_t ro = sat_cast<int64_t>(std::ceil(r_out));
  int64_t ix = iround(cx), iy = iround(cy);
  std::vector<float> buf;
  buf.reserve(256);
  double ri2 = r_in * r_in, ro2 = r_out * r_out;
  for (int64_t yy = -ro; yy <= ro; yy += step) {
    int64_t y = iy + yy;
    if (y >= 0 && y < int64_t(h)) {
      for (int64_t xx = -ro; xx <= ro; xx += step) {
        int64_t x = ix + xx;
        double ddx = double(x) - cx, ddy = double(y) - cy;
        double d2 = ddx * ddx + ddy * ddy;
        if (x >= 0 && x < int64_t(w) && d2 >= ri2 && d2 <= ro2) {
          float v = px(size_t(x), size_t(y));
          if (std::isfinite(v)) buf.push_back(v);
        }
      }
    }
  }
  if (buf.size() < 12) return std::nullopt;
  size_t mid = buf.size() / 2;
  std::nth_element(buf.begin(), buf.begin() + long(mid), buf.end(), [](float a, float b2) { return total_cmp(a, b2) < 0; });
  return buf[mid];
}

float Ctx::bilinear_native(double x, double y) const {
  if (!(x >= 0.0 && y >= 0.0)) return NANF;
  size_t x0 = sat_cast<size_t>(x), y0 = sat_cast<size_t>(y);
  if (x0 + 1 >= w || y0 + 1 >= h) {
    if (x0 < w && y0 < h) return px(x0, y0);
    return NANF;
  }
  float tx = float(x - double(x0)), ty = float(y - double(y0));
  size_t i = y0 * w + x0;
  const float* d = img;
  float top = d[i] + (d[i + 1] - d[i]) * tx;
  float bottom = d[i + w] + (d[i + w + 1] - d[i + w]) * tx;
  return top + (bottom - top) * ty;
}

bool Ctx::separate(std::pair<double, double> a, std::pair<double, double> b2, float bg, float sig_n) const {
  double d = dist(a.first - b2.first, a.second - b2.second);
  if (d < 1.5 || d < 0.9 * double(fwhm_frame)) return false;
  float va = bilinear_native(a.first, a.second);
  float vb = bilinear_native(b2.first, b2.second);
  float lower = fmin_r(va, vb);
  if (!std::isfinite(lower)) return false;
  size_t n = sat_cast<size_t>(fmax_r(std::ceil(d), 2.0));
  float low = INF;
  for (size_t i = 1; i < n; ++i) {
    double t = double(i) / double(n);
    float v = bilinear_native(a.first + (b2.first - a.first) * t, a.second + (b2.second - a.second) * t);
    if (std::isfinite(v)) low = fmin_r(low, v);
  }
  return lower - low >= fmax_r(3.0f * sig_n, 0.1f * (lower - bg));
}

std::optional<Ctx::PlateauRes> Ctx::plateau_in(size_t pxl, size_t pyl, float level, float top, size_t r) const {
  size_t x0 = pxl >= r ? pxl - r : 0, y0 = pyl >= r ? pyl - r : 0;
  size_t x1 = std::min(pxl + r, w - 1), y1 = std::min(pyl + r, h - 1);
  size_t bw = x1 - x0 + 1;
  std::vector<uint8_t> seen(bw * (y1 - y0 + 1), 0);
  std::vector<std::pair<size_t, size_t>> stack{{pxl, pyl}};
  seen[(pyl - y0) * bw + pxl - x0] = 1;
  size_t n = 0;
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  Moments m;
  size_t n_top = 0, n_max = 0;
  while (!stack.empty()) {
    auto [x, y] = stack.back();
    stack.pop_back();
    n += 1;
    float v = px(x, y);
    n_top += size_t(v >= top);
    if (x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
      bool higher = false;
      for (size_t yy = y - 1; yy <= y + 1 && !higher; ++yy)
        for (size_t xx = x - 1; xx <= x + 1; ++xx)
          if (!(xx == x && yy == y) && px(xx, yy) >= v) {
            higher = true;
            break;
          }
      n_max += size_t(!higher);
    }
    double fx = double(x), fy = double(y);
    sx += fx;
    sy += fy;
    sxx += fx * fx;
    syy += fy * fy;
    sxy += fx * fy;
    if (x == 0 || y == 0 || x + 1 == w || y + 1 == h) {
      m.clipped = true;
    } else if ((x == x0 || y == y0 || x == x1 || y == y1) && r > 1 && r < 128) {
      return std::nullopt;
    }
    int64_t sxi = int64_t(x), syi = int64_t(y);
    const int64_t nb[4][2] = {{sxi - 1, syi}, {sxi + 1, syi}, {sxi, syi - 1}, {sxi, syi + 1}};
    for (auto& q : nb) {
      int64_t nx = q[0], ny = q[1];
      if (nx < int64_t(x0) || ny < int64_t(y0) || nx > int64_t(x1) || ny > int64_t(y1)) continue;
      size_t i = (size_t(ny) - y0) * bw + size_t(nx) - x0;
      if (seen[i]) continue;
      seen[i] = 1;
      float vv = px(size_t(nx), size_t(ny));
      if (vv >= level || std::isnan(vv)) {
        if (std::isnan(vv)) {
          m.clipped = true;
          continue;
        }
        stack.emplace_back(size_t(nx), size_t(ny));
      }
    }
  }
  double nf = double(n);
  double cx = sx / nf, cy = sy / nf;
  m.sum = nf;
  m.x = cx;
  m.y = cy;
  m.xx = sxx / nf - cx * cx + 1.0 / 12.0;
  m.yy = syy / nf - cy * cy + 1.0 / 12.0;
  m.xy = sxy / nf - cx * cy;
  return PlateauRes{cx, cy, n, n_top, n_max, m};
}

Ctx::PlateauRes Ctx::plateau(size_t pxl, size_t pyl, float level, float top) const {
  if (auto r = plateau_in(pxl, pyl, level, top, 24)) return *r;
  if (auto r = plateau_in(pxl, pyl, level, top, 128)) return *r;
  auto r = plateau_in(pxl, pyl, INF, top, 1);
  if (!r) throw Error("single-pixel region");
  return *r;
}

std::optional<std::pair<double, double>> Ctx::com(double x0, double y0, double r, const Bg& bg, float thr) const {
  double x = x0, y = y0;
  int64_t ri = sat_cast<int64_t>(std::ceil(r));
  double r2 = r * r;
  for (int it = 0; it < 2; ++it) {
    int64_t ix = iround(x), iy = iround(y);
    if (ix - ri < 0 || iy - ri < 0 || ix + ri >= int64_t(w) || iy + ri >= int64_t(h)) return std::nullopt;
    double s = 0, sx = 0, sy = 0;
    for (int64_t yy = iy - ri; yy <= iy + ri; ++yy) {
      double dy = double(yy) - y;
      if (dy * dy > r2) continue;
      double half = std::sqrt(r2 - dy * dy);
      size_t xa = size_t(std::max<int64_t>(sat_cast<int64_t>(std::ceil(x - half)), 0));
      int64_t xb = std::min<int64_t>(sat_cast<int64_t>(std::floor(x + half)), int64_t(w) - 1);
      if (xb < int64_t(xa)) continue;
      const float* row = img + size_t(yy) * w;
      float row_bg = bg.at(0.0f, float(yy)) + thr;
      float fx0 = float(double(xa) - x), gx = bg.gx;
      float rs = 0.0f, rx = 0.0f;
      for (size_t k = 0; xa + k <= size_t(xb); ++k) {
        float v = row[xa + k];
        float f = v - row_bg - gx * float(xa + k);
        if (f > 0.0f) {
          rs += f;
          rx += f * (fx0 + float(k));
        }
      }
      s += double(rs);
      sx += double(rx);
      sy += double(rs) * dy;
    }
    if (s <= 0.0) return std::nullopt;
    x += sx / s;
    y += sy / s;
  }
  if (dist(x - x0, y - y0) < 1.5) return std::make_pair(x, y);
  return std::nullopt;
}

void Ctx::find_secondary(Det& det, size_t pxl, size_t pyl, float hpk, float bg, float sig) const {
  int64_t r = int64_t(b) + 1;
  float min_h = fmax_r(k * sig, 0.1f * hpk);
  int found = 0;
  for (int64_t dy = -r; dy <= r; ++dy) {
    for (int64_t dx = -r; dx <= r; ++dx) {
      if (std::max(std::llabs(dx), std::llabs(dy)) < 2) continue;
      int64_t x = int64_t(pxl) + dx, y = int64_t(pyl) + dy;
      if (x < 1 || y < 1 || x + 1 >= int64_t(w) || y + 1 >= int64_t(h)) continue;
      size_t xu = size_t(x), yu = size_t(y);
      float v = px(xu, yu);
      if (std::isnan(v - bg) || v - bg < min_h) continue;
      bool is_max = true;
      for (size_t yy = yu - 1; yy <= yu + 1 && is_max; ++yy)
        for (size_t xx = xu - 1; xx <= xu + 1; ++xx)
          if (!(xx == xu && yy == yu) && px(xx, yy) >= v) {
            is_max = false;
            break;
          }
      if (is_max && separate({double(pxl), double(pyl)}, {double(xu), double(yu)}, bg, sig) && found < 2) {
        det.secondary[found] = SecPt{true, uint32_t(xu), uint32_t(yu)};
        det.flags |= star_flags::BLENDED;
        found += 1;
      }
    }
  }
}

std::optional<Det> Ctx::measure(size_t pxl, size_t pyl, bool compact, float sw_cap, bool deblend) const {
  float pv = px(pxl, pyl);
  if (!std::isfinite(pv)) return std::nullopt;
  auto [plane_bg_v, sig_n] = plane_bg(pxl, pyl);
  float bg_tile = plane_bg_v.c;
  float hpk = pv - bg_tile;
  if (hpk <= 0.0f) return std::nullopt;
  double ff = double(fwhm_frame);
  Det det;
  det.px = uint32_t(pxl);
  det.py = uint32_t(pyl);
  det.sig_n = sig_n;
  det.bg = bg_tile;
  det.plane = plane_bg_v;
  float tol = fmax_r(fmax_r(0.01f * hpk, 2.0f * sig_n), 0.5f * lsb);
  float tol_sat = fmax_r(0.5f * lsb, 1e-6f * std::fabs(sat_level));
  bool at_sat = pv >= sat_level - tol_sat;
  int n_flat = 0;
  size_t n_clip = 0;
  float max4 = -INF;
  const int offs[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}};
  for (auto& o : offs) {
    int64_t x = int64_t(pxl) + o[0], y = int64_t(pyl) + o[1];
    if (x < 0 || y < 0 || x >= int64_t(w) || y >= int64_t(h)) continue;
    float v = px(size_t(x), size_t(y));
    if (v >= pv - tol) n_flat += 1;
    n_clip += size_t(at_sat && v >= pv - tol_sat);
    if (o[0] == 0 || o[1] == 0) max4 = fmax_r(max4, v);
  }
  det.ratio4 = fmax_r((max4 - bg_tile) / hpk, 0.0f);
  bool bright = hpk >= 50.0f * sig_n;
  bool clipped_disc = !bright && at_sat && n_clip >= 1;
  bool flat = bright && n_flat >= 4;
  bool near_sat = bright && pv >= bg_tile + 0.85f * (sat_level - bg_tile);

  std::optional<Trail> trailed = compact ? std::nullopt : trail;
  double cx, cy, fwhm, elong, pa, r_ap;
  bool clipped;
  std::optional<PlateauRes> core_region;
  if (flat || near_sat || bright || clipped_disc) {
    float level;
    if (clipped_disc) level = pv - fmax_r(0.05f * hpk, tol_sat);
    else if (flat && !near_sat) level = pv - fmax_r(0.05f * hpk, tol);
    else level = bg_tile + 0.7f * hpk;
    PlateauRes r = plateau(pxl, pyl, level, pv - tol);
    bool lumpy = r.n_max >= 2;
    double fm = fmax_r(ff, 3.0);
    bool large = double(r.n) >= 0.8 * (fm * fm) + 6.0;
    bool use_core;
    if (clipped_disc) use_core = r.n >= 5 && r.m.shape(0.0).elong <= 1.4;
    else use_core = flat || lumpy || large;
    if (use_core) core_region = r;
  }
  if (core_region) {
    const PlateauRes& r = *core_region;
    double core = std::sqrt(double(r.n) / PI);
    auto sh = r.m.shape(0.0);
    det.core = float(core);
    if (r.n_top >= 5) det.flags |= star_flags::SATURATED;
    cx = r.cx;
    cy = r.cy;
    fwhm = fmax_r(2.0 * core, ff);
    elong = sh.elong;
    pa = sh.pa;
    r_ap = core + 1.5 * fmax_r(ff, 2.0) + 1.0;
    clipped = r.m.clipped;
  } else {
    float thr = fmax_r(1.5f * sig_n, 0.02f * hpk);
    double f0 = double(thr / hpk);
    size_t hw = compact ? 2 : 3;
    Moments m0 = moments(double(pxl), double(pyl), hw, plane_bg_v, thr);
    double fwhm0 = m0.shape(f0).fwhm;
    if (!compact && fwhm0 > 4.0) hw = 5;
    if (trailed) {
      double v = std::ceil(double(0.5f * trailed->length) + fmax_r(ff, 2.0));
      hw = sat_cast<size_t>(clampd(v, 3.0, 15.0));
    }
    double fref;
    if (ff > 0.0) fref = clampd(fwhm0, 0.7 * ff, 1.5 * ff);
    else fref = fmax_r(fwhm0, 1.5);
    double r_guess = compact ? clampd(1.2 * fref, 2.0, 6.0) : clampd(1.5 * fref, 2.5, 20.0);
    double r_in = r_guess + 1.5;
    double r_out = r_in + fmax_r(fmax_r(ff, fref), 3.0);
    Bg bg = plane_bg_v;
    bool rough = sig_n > 1.5f * grid->typical_sig_n;
    if (local_background || rough) {
      if (auto ann = annulus(double(pxl), double(pyl), r_in, r_out)) {
        bool structured = std::fabs(*ann - bg_tile) > fmax_r(sig_n, 0.02f * hpk);
        if (local_background || structured) bg = plane_bg_v.with_level(*ann);
      }
    }
    hpk = pv - bg.c;
    if (hpk <= 0.0f) return std::nullopt;
    det.bg = bg.c;
    det.plane = bg;
    double s_w = std::min(clampd(fref / 2.3548, 0.8, 3.0), double(fmax_r(sw_cap, 0.6f)));
    double x, y;
    if (trailed) {
      Moments m = moments(double(pxl), double(pyl), hw, bg, thr);
      m = moments(m.x, m.y, hw, bg, thr);
      if (m.sum <= 0.0) return std::nullopt;
      x = m.x;
      y = m.y;
    } else {
      double sx, sy;
      if (m0.sum > 0.0 && std::fabs(m0.x - double(pxl)) < 1.0 && std::fabs(m0.y - double(pyl)) < 1.0) {
        sx = m0.x;
        sy = m0.y;
      } else {
        sx = double(pxl);
        sy = double(pyl);
      }
      if (auto c = xwin(sx, sy, hw, s_w, bg)) {
        x = c->first;
        y = c->second;
      } else {
        Moments m = moments(double(pxl), double(pyl), hw, bg, thr);
        if (m.sum <= 0.0) return std::nullopt;
        x = m.x;
        y = m.y;
      }
    }
    if (std::hypot(x - double(pxl), y - double(pyl)) > 2.5 + double(hw) * 0.5) return std::nullopt;
    Moments m = moments(x, y, hw, bg, thr);
    Moments::Shape shape;
    if (m.sum > 0.0) shape = m.shape(double(thr / hpk));
    else shape = {fref, 1.0, 0.0};
    cx = x;
    cy = y;
    fwhm = shape.fwhm;
    elong = shape.elong;
    pa = shape.pa;
    double fa = ff > 0.0 ? fmax_r(ff, 0.8 * fref) : fmax_r(fwhm, fref);
    r_ap = compact ? clampd(1.2 * fa, 2.0, 6.0) : clampd(1.25 * fa, 2.5, 16.0);
    clipped = m.clipped;
    if (deblend) find_secondary(det, pxl, pyl, hpk, bg.c, sig_n);
  }
  auto ap = aperture(img, w, h, cx, cy, r_ap, det.plane);
  if (ap.flux <= 0.0 || ap.n == 0) return std::nullopt;
  det.snr = ap.flux / fmax_r(double(sig_n) * std::sqrt(double(ap.n)), std::numeric_limits<double>::min());
  if (det.core == 0.0f && !trailed && std::isinf(sw_cap)) {
    double alpha = clampd((det.snr - 20.0) / 60.0, 0.0, 1.0);
    if (alpha > 0.0) {
      double r = compact ? clampd(1.2 * fwhm, 2.0, 5.0) : clampd(1.25 * fwhm, 2.5, 8.0);
      if (auto xy = com(cx, cy, r, det.plane, 1.5f * sig_n)) {
        cx += alpha * (xy->first - cx);
        cy += alpha * (xy->second - cy);
      }
    }
  }
  det.x = cx;
  det.y = cy;
  det.h = hpk;
  det.flux = ap.flux;
  det.fwhm = fwhm;
  det.elong = elong;
  det.pa = pa;
  if (clipped || ap.clipped) det.flags |= star_flags::NEAR_EDGE;
  return det;
}

}  // namespace skyshot::extract_detail
