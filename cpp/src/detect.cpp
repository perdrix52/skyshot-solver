#include "skyshot/detect.hpp"

#include <algorithm>
#include <set>

namespace skyshot::detect {

using Clock = std::chrono::steady_clock;

static float median_f(std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  size_t n = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + long(n), v.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
  return v[n];
}

static StarField detect_aperture(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline, size_t aperture,
                                 bool local_background) {
  size_t factor = std::max<size_t>(std::max(image.width, image.height) / 2400, 1);
  size_t w = image.width / factor, h = image.height / factor;
  std::vector<float> data(w * h, std::numeric_limits<float>::quiet_NaN());
  auto stop = [&] { return cancel.load(std::memory_order_relaxed) || Clock::now() >= deadline; };
  for (size_t y = 0; y < h; ++y) {
    if (y % 32 == 0 && stop()) break;
    for (size_t x = 0; x < w; ++x) {
      float sum = 0.0f;
      int n = 0;
      for (size_t dy = 0; dy < factor; ++dy)
        for (size_t dx = 0; dx < factor; ++dx) {
          float v = image.pixels[(y * factor + dy) * image.width + x * factor + dx];
          if (std::isfinite(v)) {
            sum += v;
            n += 1;
          }
        }
      if (n > 0) data[y * w + x] = sum / float(n);
    }
  }
  size_t nx = (w + 63) / 64, ny = (h + 63) / 64;
  std::vector<std::pair<float, float>> backgrounds(nx * ny, {0.0f, 1.0f});
  for (size_t by = 0; by < ny; ++by)
    for (size_t bx = 0; bx < nx; ++bx) {
      std::vector<float> sample;
      for (size_t y = by * 64; y < std::min((by + 1) * 64, h); y += 2)
        for (size_t x = bx * 64; x < std::min((bx + 1) * 64, w); x += 2) {
          float v = data[y * w + x];
          if (std::isfinite(v)) sample.push_back(v);
        }
      float bg = median_f(sample);
      for (float& v : sample) v = std::fabs(v - bg);
      float noise = fmax_r(median_f(sample) * 1.4826f, std::fabs(bg) * 1e-6f + 1e-8f);
      backgrounds[by * nx + bx] = {bg, noise};
    }
  struct Peak {
    float v;
    size_t x, y;
  };
  std::vector<Peak> peaks;
  for (size_t y = 2; y < (h >= 2 ? h - 2 : 0); ++y) {
    if (y % 32 == 0 && stop()) break;
    for (size_t x = 2; x < (w >= 2 ? w - 2 : 0); ++x) {
      float v = data[y * w + x];
      auto [bg, noise] = backgrounds[(y / 64) * nx + x / 64];
      if (!std::isfinite(v) || v < bg + 5.0f * noise) continue;
      bool largest = true;
      int support = 0;
      for (size_t yy = y - 1; yy <= y + 1; ++yy)
        for (size_t xx = x - 1; xx <= x + 1; ++xx) {
          float q = data[yy * w + xx];
          if (q > v || (q == v && (yy * w + xx) < y * w + x)) largest = false;
          if (q > bg + 2.0f * noise) support += 1;
        }
      if (largest && support >= 3) peaks.push_back({v - bg, x, y});
    }
  }
  std::stable_sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) { return total_cmp(b.v, a.v) < 0; });
  if (peaks.size() > 20000) peaks.resize(20000);
  std::vector<Star> stars;
  struct Measured {
    double x, y, fwhm;
  };
  std::vector<Measured> measured;
  std::set<std::pair<size_t, size_t>> occupied;
  const size_t r = aperture;
  for (const Peak& pk : peaks) {
    size_t x = pk.x, y = pk.y;
    if (stop()) break;
    if (occupied.count({x, y})) continue;
    auto [bg, noise] = backgrounds[(y / 64) * nx + x / 64];
    if (local_background) {
      size_t outer = r + 3;
      std::vector<float> annulus;
      annulus.reserve(128);
      for (size_t yy = y >= outer ? y - outer : 0; yy <= std::min(y + outer, h - 1); ++yy)
        for (size_t xx = x >= outer ? x - outer : 0; xx <= std::min(x + outer, w - 1); ++xx) {
          size_t ax = xx > x ? xx - x : x - xx, ay = yy > y ? yy - y : y - yy;
          if (std::max(ax, ay) > r) {
            float value = data[yy * w + xx];
            if (std::isfinite(value)) annulus.push_back(value);
          }
        }
      if (annulus.size() < 16) continue;
      bg = median_f(annulus);
      if (data[y * w + x] < bg + 5.0f * noise) continue;
    }
    size_t x0 = x >= r ? x - r : 0, x1 = std::min(x + r, w - 1), y0 = y >= r ? y - r : 0, y1 = std::min(y + r, h - 1);
    double flux = 0, sx = 0, sy = 0;
    int support = 0;
    for (size_t yy = y0; yy <= y1; ++yy)
      for (size_t xx = x0; xx <= x1; ++xx) {
        float value = data[yy * w + xx];
        if (!std::isfinite(value)) continue;
        double f = double(fmax_r(value - bg - 1.5f * noise, 0.0f));
        if (f > 0.0) {
          flux += f;
          sx += f * double(xx);
          sy += f * double(yy);
          support += 1;
        }
      }
    if (flux <= 0.0 || support < 3) continue;
    double cx = sx / flux, cy = sy / flux;
    if (std::hypot(cx - double(x), cy - double(y)) > 2.5) continue;
    double xxm = 0, yym = 0, xym = 0;
    for (size_t yy = y0; yy <= y1; ++yy)
      for (size_t xx = x0; xx <= x1; ++xx) {
        float v = data[yy * w + xx];
        if (!std::isfinite(v)) continue;
        double f = double(fmax_r(v - bg - 1.5f * noise, 0.0f));
        double dx = double(xx) - cx, dy = double(yy) - cy;
        xxm += f * dx * dx;
        yym += f * dy * dy;
        xym += f * dx * dy;
      }
    xxm /= flux;
    yym /= flux;
    xym /= flux;
    double a = xxm - yym;
    double disc = std::sqrt(a * a + 4.0 * xym * xym);
    double minor = fmax_r((xxm + yym - disc) / 2.0, 0.02);
    double major = fmax_r((xxm + yym + disc) / 2.0, minor);
    double elongation = std::sqrt(major / minor);
    if (elongation > 5.0 || minor < 0.06) continue;
    double fwhm = 2.35482 * std::sqrt((major + minor) / 2.0) * double(factor);
    if (local_background) {
      double radius = fmax_r(fwhm / double(factor) * 0.75, 2.0);
      bool near = false;
      for (const Measured& p : measured) {
        double dx = p.x - cx, dy = p.y - cy;
        if (dx * dx + dy * dy > radius * radius) continue;
        double limit = fmax_r(fmin_r(fwhm, p.fwhm) / double(factor) * 0.75, 2.0);
        if (std::hypot(p.x - cx, p.y - cy) < limit) {
          near = true;
          break;
        }
      }
      if (near) continue;
      measured.push_back({cx, cy, fwhm});
    }
    double snr = flux / (double(noise) * std::sqrt(double(support)));
    float peak = data[y * w + x];
    float tolerance = fmax_r(std::fabs(peak) * 1e-6f, noise * 0.05f);
    size_t ix = sat_cast<size_t>(std::round(cx)), iy = sat_cast<size_t>(std::round(cy));
    int plateau = 0;
    for (size_t yy = iy >= 1 ? iy - 1 : 0; yy <= std::min(iy + 1, h - 1); ++yy)
      for (size_t xx = ix >= 1 ? ix - 1 : 0; xx <= std::min(ix + 1, w - 1); ++xx)
        if (std::fabs(data[yy * w + xx] - peak) <= tolerance) plateau += 1;
    bool saturated = snr > 50.0 && plateau >= 5;
    Star s;
    s.x = (cx + 0.5) * double(factor) - 0.5;
    s.y = (cy + 0.5) * double(factor) - 0.5;
    s.flux = flux * double(factor * factor);
    s.sigma = clampd(fwhm / (2.35 * snr), saturated ? 0.5 : 0.08, 1.5);
    s.snr = snr;
    s.fwhm = fwhm;
    s.elongation = elongation;
    s.saturated = saturated;
    s.peak = double(peak - bg);
    s.pa = rem_euclid(to_degrees(0.5 * std::atan2(2.0 * xym, xxm - yym)), 180.0);
    s.flags = saturated ? star_flags::SATURATED : 0;
    stars.push_back(s);
    for (size_t yy = y >= 2 ? y - 2 : 0; yy <= std::min(y + 2, h - 1); ++yy)
      for (size_t xx = x >= 2 ? x - 2 : 0; xx <= std::min(x + 2, w - 1); ++xx) occupied.insert({xx, yy});
    if (stars.size() >= 5000) break;
  }
  StarField f;
  f.width = image.width;
  f.height = image.height;
  f.stars = std::move(stars);
  f.epoch = image.epoch;
  return f;
}

StarField detect(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline) {
  return detect_aperture(image, cancel, deadline, 5, false);
}
StarField detect_compact(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline) {
  return detect_aperture(image, cancel, deadline, 2, false);
}
StarField detect_local(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline) {
  return detect_aperture(image, cancel, deadline, 5, true);
}

}  // namespace skyshot::detect
