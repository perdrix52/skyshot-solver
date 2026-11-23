// GrayImage::preview and a minimal PNG encoder. Port of the preview part of src/image_io.rs.
#include <algorithm>

#include "skyshot/image_io.hpp"
#include "skyshot/parallel.hpp"

namespace skyshot::image_io {

std::pair<uint32_t, uint32_t> GrayImage::preview_size() const {
  double factor = fmax_r(double(std::max(width, height)) / double(PREVIEW_MAX_SIDE), 1.0);
  uint32_t w = std::max<uint32_t>(sat_cast<uint32_t>(std::round(double(width) / factor)), 1);
  uint32_t h = std::max<uint32_t>(sat_cast<uint32_t>(std::round(double(height) / factor)), 1);
  return {w, h};
}

std::vector<float> GrayImage::box_downsample(uint32_t w_, uint32_t h_) const {
  size_t sw = width, sh = height, w = w_, h = h_;
  auto bounds = [](size_t i, size_t src, size_t dst) {
    size_t a = i * src / dst;
    size_t b = std::clamp<size_t>((i + 1) * src / dst, a + 1, src);
    return std::make_pair(a, b);
  };
  std::vector<float> out(w * h, 0.0f);
  parallel_for(h, [&](size_t y) {
    auto [y0, y1] = bounds(y, sh, h);
    for (size_t x = 0; x < w; ++x) {
      auto [x0, x1] = bounds(x, sw, w);
      double sum = 0.0;
      uint32_t n = 0;
      for (size_t yy = y0; yy < y1; ++yy)
        for (size_t xx = x0; xx < x1; ++xx) {
          float p = pixels[yy * sw + xx];
          if (std::isfinite(p)) {
            sum += double(p);
            ++n;
          }
        }
      out[y * w + x] = n > 0 ? float(sum / double(n)) : std::numeric_limits<float>::quiet_NaN();
    }
  });
  return out;
}

static float quantile(std::vector<float>& v, double f) {
  size_t k = sat_cast<size_t>(std::round(double(v.size() - 1) * f));
  std::nth_element(v.begin(), v.begin() + long(k), v.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
  return v[k];
}

static float asinh_strength(float t, float target) {
  auto y = [&](float k) { return std::asinh(k * t) / std::asinh(k); };
  float lo = std::log(1e-3f), hi = std::log(1e5f);
  if (std::isnan(t) || t <= 0.0f || y(std::exp(hi)) <= target) return std::exp(hi);
  if (y(std::exp(lo)) >= target) return std::exp(lo);
  for (int i = 0; i < 40; ++i) {
    float mid = 0.5f * (lo + hi);
    if (y(std::exp(mid)) < target) lo = mid;
    else hi = mid;
  }
  return std::exp(0.5f * (lo + hi));
}

// ---- PNG ----
static uint32_t crc32_update(uint32_t crc, const uint8_t* d, size_t n) {
  static uint32_t table[256];
  static bool init = false;
  if (!init) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[i] = c;
    }
    init = true;
  }
  crc = ~crc;
  for (size_t i = 0; i < n; ++i) crc = table[(crc ^ d[i]) & 255] ^ (crc >> 8);
  return ~crc;
}

static void put_be32(std::vector<uint8_t>& o, uint32_t v) {
  for (int s = 24; s >= 0; s -= 8) o.push_back(uint8_t(v >> s));
}

static void put_chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
  put_be32(out, uint32_t(data.size()));
  std::vector<uint8_t> td(type, type + 4);
  td.insert(td.end(), data.begin(), data.end());
  out.insert(out.end(), td.begin(), td.end());
  put_be32(out, crc32_update(0, td.data(), td.size()));
}

std::vector<uint8_t> encode_png(uint32_t width, uint32_t height, int channels, const uint8_t* data) {
  std::vector<uint8_t> raw;
  raw.reserve(size_t(height) * (size_t(width) * size_t(channels) + 1));
  for (uint32_t y = 0; y < height; ++y) {
    raw.push_back(0);  // filter: none
    raw.insert(raw.end(), data + size_t(y) * width * size_t(channels), data + (size_t(y) + 1) * width * size_t(channels));
  }
  std::vector<uint8_t> z = {0x78, 0x01};
  size_t pos = 0;
  while (pos < raw.size() || raw.empty()) {
    size_t n = std::min<size_t>(65535, raw.size() - pos);
    bool last = pos + n >= raw.size();
    z.push_back(last ? 1 : 0);
    z.push_back(uint8_t(n & 255));
    z.push_back(uint8_t(n >> 8));
    z.push_back(uint8_t(~n & 255));
    z.push_back(uint8_t((~n >> 8) & 255));
    z.insert(z.end(), raw.begin() + long(pos), raw.begin() + long(pos + n));
    pos += n;
    if (last) break;
  }
  uint32_t a = 1, b = 0;
  for (uint8_t c : raw) {
    a = (a + c) % 65521;
    b = (b + a) % 65521;
  }
  put_be32(z, (b << 16) | a);
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
  std::vector<uint8_t> ihdr;
  put_be32(ihdr, width);
  put_be32(ihdr, height);
  const uint8_t color[5] = {0, 0, 4, 2, 6};  // index by channels: 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA
  ihdr.push_back(8);
  ihdr.push_back(color[std::clamp(channels, 1, 4)]);
  ihdr.push_back(0);
  ihdr.push_back(0);
  ihdr.push_back(0);
  put_chunk(out, "IHDR", ihdr);
  put_chunk(out, "IDAT", z);
  put_chunk(out, "IEND", {});
  return out;
}

std::vector<uint8_t> GrayImage::preview() const {
  auto [w, h] = preview_size();
  std::vector<float> small = box_downsample(w, h);
  size_t step = std::max<size_t>(small.size() / 200000, 1);
  std::vector<float> sample;
  for (size_t i = 0; i < small.size(); i += step)
    if (std::isfinite(small[i])) sample.push_back(small[i]);
  if (sample.empty()) throw Error("No finite pixels");
  float med = quantile(sample, 0.5);
  float floor = quantile(sample, 0.001);
  float top = quantile(sample, 0.9999);
  std::vector<float> dev;
  for (float x : sample) dev.push_back(std::fabs(x - med));
  float spread = 1.4826f * quantile(dev, 0.5);
  std::vector<float> diff;
  size_t row_step = std::max<size_t>(size_t(h) * w / 200000, 1);
  for (size_t y = 0; y < h; y += row_step)
    for (size_t x = 0; x + 1 < w; ++x) {
      float d = std::fabs(small[y * w + x + 1] - small[y * w + x]);
      if (std::isfinite(d)) diff.push_back(d);
    }
  float local = diff.empty() ? 0.0f : 1.4826f * quantile(diff, 0.5) / 1.41421356f;
  float sigma = (local > 0.0f && local < spread) ? std::sqrt(local * spread) : spread;
  float lo = fmax_r(med - 2.8f * sigma, floor);
  float hi = fmax_r(fmax_r(top, med + 25.0f * sigma), lo + 1e-6f);
  float k = asinh_strength((med - lo) / (hi - lo), 0.15f);
  float norm = 255.0f / std::asinh(k);
  std::vector<uint8_t> gray(small.size());
  parallel_for(h, [&](size_t y) {
    for (size_t x = 0; x < w; ++x) {
      float p = small[y * w + x];
      float t = std::isfinite(p) ? clampf((p - lo) / (hi - lo), 0.0f, 1.0f) : 0.0f;
      gray[y * w + x] = sat_cast<uint8_t>(clampf(std::round(std::asinh(k * t) * norm), 0.0f, 255.0f));
    }
  });
  return encode_png(w, h, 1, gray.data());
}

}  // namespace skyshot::image_io
