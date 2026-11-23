// Internal structures shared by the extractor, vetting and line removal.
#pragma once
#include <bit>
#include <optional>

#include "skyshot/extract.hpp"

namespace skyshot::extract_detail {

inline constexpr size_t TILE_NATIVE = 64;
constexpr float F32_MIN_POSITIVE = 1.17549435e-38f;

inline uint32_t key_of(float v) {
  auto b = std::bit_cast<uint32_t>(v);
  return (b >> 31) == 0 ? (b | 0x80000000u) : ~b;
}
inline float unkey(uint32_t k) {
  uint32_t b = (k >> 31) == 1 ? (k & 0x7fffffffu) : ~k;
  return std::bit_cast<float>(b);
}
inline double dist(double dx, double dy) { return std::sqrt(dx * dx + dy * dy); }
float noise_floor(float level, float lsb);

struct Plane {
  size_t w = 0, h = 0;
  const float* data = nullptr;
  std::vector<float> owned;
  float bilinear(float x, float y) const;
};

void bin_plane(const image_io::GrayImage& image, size_t b, Plane& plane);

struct Axis {
  size_t n = 1;
  float s = 1.0f;
  size_t len = 0;
  static Axis make(size_t len, size_t tile);
  std::pair<size_t, size_t> range(size_t t) const;
  size_t index(size_t c) const;
  std::pair<size_t, float> locate(float c) const;
  struct Lin {
    size_t i0, i1;
    float t;
  };
  Lin linear(float c) const;
  std::array<std::pair<uint32_t, float>, 6> cubic(float c) const;
};

struct TileLevel {
  float bg;
  std::optional<float> degenerate_sigma;
  float floor;
};

struct Grid {
  Axis ax, ay;
  size_t nx = 0, ny = 0, b = 1;
  std::vector<float> bg, sig_b, sig_n;
  float typical_sig_n = 0;
  std::vector<std::array<std::pair<uint32_t, float>, 6>> xcub;
  std::vector<std::array<float, 3>> xlin_t;  // unused placeholder (kept simple)
  struct XLin {
    uint32_t i0, i1;
    float t;
  };
  std::vector<XLin> xlin;

  static std::optional<Grid> build(const Plane& plane, const std::vector<float>& native, size_t w, size_t b, float lsb);
  void row_cubic_range(float yb, size_t c0, size_t c1, float* out) const;
  void row_cubic(float yb, float* out) const;
  float col_cubic(size_t xb, const float* colv) const;
  void row_sigma(float yb, float* out) const;
  float col_linear(size_t xb, const float* colv) const;
  struct At {
    float bg, sig_b, sig_n;
  };
  At at(float xb, float yb) const;
  At at_native(float x, float y) const;
  size_t tile_index(size_t xb, size_t yb) const;
};

struct Cand {
  uint32_t x, y;
  float v, bg, sig;
};

struct Trail {
  float pa, length;
};

struct Contrast {
  float height, contrast, floor;
};

struct Bg {
  float c = 0, gx = 0, gy = 0, x0 = 0, y0 = 0;
  float at(float x, float y) const { return c + gx * (x - x0) + gy * (y - y0); }
  Bg with_level(float level) const {
    Bg r = *this;
    r.c = level;
    return r;
  }
};

struct Moments {
  double sum = 0, x = 0, y = 0, xx = 0, yy = 0, xy = 0;
  bool clipped = false;
  struct Shape {
    double fwhm, elong, pa;
  };
  Shape shape(double f) const;
  std::pair<double, double> axes() const;
};

struct SecPt {
  bool has = false;
  uint32_t x = 0, y = 0;
};

struct Det {
  double x = 0, y = 0;
  uint32_t px = 0, py = 0;
  float h = 0, bg = 0;
  Bg plane;
  float sig_n = 0;
  double flux = 0, snr = 0, fwhm = 0, elong = 0, pa = 0;
  float ratio4 = 0, core = 0;
  uint32_t flags = 0;
  SecPt secondary[2];
  Star to_star(float trail_len) const;
};

struct Ctx {
  const float* img = nullptr;
  size_t w = 0, h = 0, b = 1;
  const Plane* plane = nullptr;
  const Grid* grid = nullptr;
  float lsb = 0, k = 0, sat_level = 0;
  bool local_background = false;
  float fwhm_frame = 0;
  std::optional<Trail> trail;

  float px(size_t x, size_t y) const { return img[y * w + x]; }
  std::array<float, 2> to_native(float xb, float yb) const;
  std::pair<Bg, float> plane_bg(size_t px, size_t py) const;
  std::optional<std::pair<size_t, size_t>> native_peak(const Cand& c) const;
  bool is_ridge(const Cand& c) const;
  bool is_compact_star(const Cand& c) const;
  std::optional<Contrast> contrast(const Cand& c) const;
  Moments moments(double cx, double cy, size_t hw, const Bg& bg, float thr) const;
  std::optional<std::pair<double, double>> xwin(double x0, double y0, size_t hw, double s, const Bg& bg) const;
  std::optional<float> annulus(double cx, double cy, double r_in, double r_out) const;
  bool separate(std::pair<double, double> a, std::pair<double, double> b, float bg, float sig_n) const;
  float bilinear_native(double x, double y) const;
  struct PlateauRes {
    double cx, cy;
    size_t n, n_top, n_max;
    Moments m;
  };
  PlateauRes plateau(size_t px, size_t py, float level, float top) const;
  std::optional<PlateauRes> plateau_in(size_t px, size_t py, float level, float top, size_t r) const;
  std::optional<Det> measure(size_t px, size_t py, bool compact, float sw_cap, bool deblend) const;
  std::optional<std::pair<double, double>> com(double x0, double y0, double r, const Bg& bg, float thr) const;
  void find_secondary(Det& det, size_t px, size_t py, float hpk, float bg, float sig) const;
};

struct ApertureResult {
  double flux;
  size_t n;
  bool clipped;
};
ApertureResult aperture(const float* img, size_t w, size_t h, double cx, double cy, double r, const Bg& bg);

struct FramePsf {
  float fwhm = 0;
  std::optional<Trail> trail;
};
FramePsf frame_psf(const Ctx& ctx, const std::vector<Cand>& cands);
void vet(const Ctx& ctx, std::vector<Det>& dets);
void flag_lines(const Ctx& ctx, std::vector<Det>& dets, const std::vector<std::array<float, 2>>& hints);
void merge_duplicates(const Ctx& ctx, std::vector<Det>& dets);
void merge_psf_lobes(const Ctx& ctx, std::vector<Det>& dets);

}  // namespace skyshot::extract_detail
