#include "skyshot/highpass.hpp"

#include <algorithm>
#include <map>
#include <unordered_set>

#include "skyshot/parallel.hpp"

namespace skyshot::highpass {

size_t box_radius(float sigma) {
  float v = std::sqrt((12.0f * sigma * sigma / 3.0f) + 1.0f);
  return sat_cast<size_t>(fmax_r(std::round((v - 1.0f) / 2.0f), 1.0f));
}

static constexpr size_t PAR_MIN_PX = size_t(1) << 16;

std::vector<float> box_blur3(const std::vector<float>& src, size_t w, size_t h, float sigma) {
  size_t r = box_radius(sigma);
  float n = float(2 * r + 1);
  bool par = w * h >= PAR_MIN_PX;
  auto row_blur = [&](float* dst, const float* row) {
    float acc = -0.0f;
    for (size_t i = 0; i <= std::min(r, w - 1); ++i) acc += row[i];
    acc += row[0] * float(r);
    for (size_t x = 0; x < w; ++x) {
      dst[x] = acc / n;
      float add = row[std::min(x + r + 1, w - 1)];
      float sub = row[x >= r ? x - r : 0];
      acc += add - sub;
    }
  };
  auto col_blur = [&](size_t x0, size_t x1, const float* s, float* out) {
    for (size_t x = x0; x < x1; ++x) {
      float acc = 0.0f;
      for (size_t y = 0; y <= std::min(r, h - 1); ++y) acc += s[y * w + x];
      acc += s[x] * float(r);
      for (size_t y = 0; y < h; ++y) {
        out[y * w + x] = acc / n;
        float add = s[std::min(y + r + 1, h - 1) * w + x];
        float sub = s[(y >= r ? y - r : 0) * w + x];
        acc += add - sub;
      }
    }
  };
  std::vector<float> a = src;
  std::vector<float> b(src.size(), 0.0f);
  for (int pass = 0; pass < 3; ++pass) {
    if (par) parallel_for(h, [&](size_t y) { row_blur(b.data() + y * w, a.data() + y * w); });
    else
      for (size_t y = 0; y < h; ++y) row_blur(b.data() + y * w, a.data() + y * w);
    std::vector<float> out(src.size(), 0.0f);
    if (par) {
      size_t strips = 64;
      size_t chunk = (w + strips - 1) / strips;
      parallel_for(strips, [&](size_t s) {
        size_t x0 = s * chunk, x1 = std::min((s + 1) * chunk, w);
        if (x0 >= x1) return;
        col_blur(x0, x1, b.data(), out.data());
      });
    } else {
      col_blur(0, w, b.data(), out.data());
    }
    a.swap(out);
  }
  return a;
}

static float median_f(std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + long(n), v.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
  return v[n];
}

static size_t clipped_area(const std::vector<float>& plane, size_t w, size_t h, size_t x, size_t y, float sat,
                           std::unordered_set<size_t>& seen) {
  seen.clear();
  std::vector<size_t> stack;
  for (size_t yy = y >= 1 ? y - 1 : 0; yy <= std::min(y + 1, h - 1); ++yy)
    for (size_t xx = x >= 1 ? x - 1 : 0; xx <= std::min(x + 1, w - 1); ++xx) {
      size_t i = yy * w + xx;
      if (plane[i] >= sat && seen.insert(i).second) stack.push_back(i);
    }
  size_t n = 0;
  while (!stack.empty()) {
    size_t i = stack.back();
    stack.pop_back();
    n += 1;
    if (n >= 4096) break;
    size_t cx = i % w, cy = i / w;
    for (size_t yy = cy >= 1 ? cy - 1 : 0; yy <= std::min(cy + 1, h - 1); ++yy)
      for (size_t xx = cx >= 1 ? cx - 1 : 0; xx <= std::min(cx + 1, w - 1); ++xx) {
        size_t j = yy * w + xx;
        if (plane[j] >= sat && seen.insert(j).second) stack.push_back(j);
      }
  }
  return n;
}

StarField detect_highpass(const image_io::GrayImage& img, float fwhm_px, float threshold) {
  size_t w0 = img.width, h0 = img.height;
  size_t bin = w0 * h0 > 8000000 ? 2 : 1;
  size_t w = w0 / bin, h = h0 / bin;
  std::vector<float> plane(w * h, 0.0f);
  parallel_for(h, [&](size_t y) {
    for (size_t x = 0; x < w; ++x) {
      float s = 0.0f, n = 0.0f;
      for (size_t dy = 0; dy < bin; ++dy)
        for (size_t dx = 0; dx < bin; ++dx) {
          float p = img.pixels[(y * bin + dy) * w0 + x * bin + dx];
          if (std::isfinite(p)) {
            s += p;
            n += 1.0f;
          }
        }
      plane[y * w + x] = n > 0.0f ? s / n : 0.0f;
    }
  });
  StarField out;
  out.width = img.width;
  out.height = img.height;
  out.epoch = img.epoch;
  if (w < 16 || h < 16) return out;
  float s1 = fmax_r(fwhm_px / float(bin) / 2.355f, 0.7f);
  auto g1 = box_blur3(plane, w, h, s1);
  auto g2 = box_blur3(plane, w, h, 3.0f * s1);
  std::vector<float> dog(g1.size());
  for (size_t i = 0; i < dog.size(); ++i) dog[i] = g1[i] - g2[i];
  const size_t t = 64;
  size_t tx = (w + t - 1) / t, ty = (h + t - 1) / t;
  std::vector<float> noise(tx * ty);
  parallel_for(tx * ty, [&](size_t k) {
    size_t cx = k % tx, cy = k / tx;
    std::vector<float> v;
    for (size_t y = cy * t; y < std::min((cy + 1) * t, h); y += 2)
      for (size_t x = cx * t; x < std::min((cx + 1) * t, w); x += 2) v.push_back(dog[y * w + x]);
    float m = median_f(v);
    for (float& x : v) x = std::fabs(x - m);
    noise[k] = fmax_r(1.4826f * median_f(v), 1e-6f);
  });
  struct Peak {
    size_t x, y;
    float v;
  };
  std::vector<std::vector<Peak>> per_row(h > 2 ? h - 2 : 0);
  parallel_for(per_row.size(), [&](size_t yi) {
    size_t y = yi + 1;
    for (size_t x = 1; x + 1 < w; ++x) {
      float v = dog[y * w + x];
      float sig = noise[(y / t) * tx + x / t];
      if (v < threshold * sig) continue;
      bool ok = true;
      for (int dy = -1; dy <= 1 && ok; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
          if ((dx != 0 || dy != 0) && dog[size_t(int64_t(y) + dy) * w + size_t(int64_t(x) + dx)] > v) {
            ok = false;
            break;
          }
      if (ok) per_row[yi].push_back({x, y, v});
    }
  });
  std::map<size_t, std::vector<Peak>> by_tile;
  for (auto& row : per_row)
    for (auto& p : row) by_tile[(p.y / t) * tx + p.x / t].push_back(p);
  float sat = -INF;
  for (float v : plane)
    if (std::isfinite(v)) sat = fmax_r(sat, v);
  std::unordered_set<size_t> seen;
  std::vector<Star> stars;
  std::vector<size_t> clipped;
  for (auto& [tile, v] : by_tile) {
    (void)tile;
    std::stable_sort(v.begin(), v.end(), [](const Peak& a, const Peak& b) { return total_cmp(b.v, a.v) < 0; });
    if (v.size() > 48) v.resize(48);
    for (const Peak& p : v) {
      size_t x = p.x, y = p.y;
      float val = p.v;
      float c = dog[y * w + x];
      float l = dog[y * w + x - 1], r = dog[y * w + x + 1];
      float u = dog[(y - 1) * w + x], d = dog[(y + 1) * w + x];
      float ox = 0.0f;
      {
        float den = l - 2.0f * c + r;
        if (den < 0.0f) ox = clampf(0.5f * (l - r) / den, -0.5f, 0.5f);
      }
      float oy = 0.0f;
      {
        float den = u - 2.0f * c + d;
        if (den < 0.0f) oy = clampf(0.5f * (u - d) / den, -0.5f, 0.5f);
      }
      float fx = (float(x) + ox + 0.5f) * float(bin) - 0.5f;
      float fy = (float(y) + oy + 0.5f) * float(bin) - 0.5f;
      float sig = noise[(y / t) * tx + x / t];
      float e = 0.0f;
      for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) e += fmax_r(dog[size_t(int64_t(y) + dy) * w + size_t(int64_t(x) + dx)], 0.0f);
      double snr = double(val / sig);
      clipped.push_back(clipped_area(plane, w, h, x, y, sat, seen));
      Star s;
      s.x = double(fx);
      s.y = double(fy);
      s.flux = double(e);
      s.sigma = clampd(0.6 * double(fwhm_px) / fmax_r(snr, 1.0), 0.1, 1.5);
      s.snr = snr;
      s.fwhm = double(fwhm_px);
      s.elongation = 1.0;
      s.saturated = false;
      stars.push_back(s);
    }
  }
  double e_max = 0.0;
  for (auto& s : stars) e_max = fmax_r(e_max, s.flux);
  e_max = fmax_r(e_max, std::numeric_limits<double>::min());
  double e_sat = 0.0;
  for (size_t i = 0; i < stars.size(); ++i)
    if (clipped[i] < 2) e_sat = fmax_r(e_sat, stars[i].flux);
  e_sat = fmax_r(e_sat, e_max * 1e-3);
  double two_pi_s2 = 2.0 * PI * (double(s1) * double(s1));
  for (size_t i = 0; i < stars.size(); ++i) {
    size_t a = clipped[i];
    if (a >= 2) {
      double area = double(a) + 0.5 * stars[i].flux / e_max;
      stars[i].flux = e_sat * std::exp(fmin_r(area / two_pi_s2, 40.0));
      stars[i].saturated = true;
    }
  }
  std::sort(stars.begin(), stars.end(), [](const Star& a, const Star& b) {
    int c = total_cmp(b.flux, a.flux);
    if (c != 0) return c < 0;
    c = total_cmp(a.y, b.y);
    if (c != 0) return c < 0;
    return total_cmp(a.x, b.x) < 0;
  });
  out.stars = std::move(stars);
  return out;
}

}  // namespace skyshot::highpass
