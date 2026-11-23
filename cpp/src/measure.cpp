// Forced photometry. Port of src/measure.rs.
#include "skyshot/measure.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

#include "extract_internal.hpp"
#include "skyshot/highpass.hpp"
#include "skyshot/coordinates.hpp"
#include "skyshot/parallel.hpp"

namespace skyshot::measure {

namespace ed = extract_detail;

constexpr size_t N_CAL_CANDIDATES = 200000;
constexpr uint64_t CAL_SEED = 0x5eedc0de0f00d5edull;
constexpr size_t N_PSF = 150;
constexpr double PSF_MIN_SNR = 30.0, PSF_AMP = 30.0, PSF_AMP_RELAXED = 15.0;
constexpr size_t PSF_MIN_FITS = 5, PSF_MIN_FITS_RELAXED = 3;
constexpr size_t MAX_REFS_MEASURED = 30000;
constexpr double NANd = std::numeric_limits<double>::quiet_NaN();

const char* status_name(Status s) {
  switch (s) {
    case Status::Recorded: return "recorded";
    case Status::Trace: return "trace";
    case Status::Below: return "below";
    case Status::NotAssessable: return "not_assessable";
  }
  std::unreachable();
}
const char* env_name(Env e) {
  switch (e) {
    case Env::Sky: return "sky";
    case Env::Faint: return "faint";
    default: return "bright";
  }
}
static const char* fwhm_source_name(FwhmSource s) {
  switch (s) {
    case FwhmSource::Given: return "given";
    case FwhmSource::Fit: return "fit";
    default: return "fit_relaxed";
  }
}

static std::optional<Env> env_of(double level, double sky, double sigma) {
  double ex = (level - sky) / sigma;
  if (!std::isfinite(ex)) return std::nullopt;
  if (ex < ENV_FAINT) return Env::Sky;
  if (ex < ENV_BRIGHT) return Env::Faint;
  return Env::Bright;
}

// ---------- content ----------
size_t apply_content(GrayImage& img, const Content& c) {
  size_t w = img.width, h = img.height;
  auto span = [&](const std::array<double, 4>& b) {
    auto cl = [](double v, size_t n) { return sat_cast<size_t>(clampd(v, 0.0, double(n))); };
    return std::array<size_t, 4>{cl(b[0], w), cl(b[1], h), cl(b[2], w), cl(b[3], h)};
  };
  std::vector<std::array<size_t, 4>> inc, exc;
  for (auto& b : c.include) inc.push_back(span(b));
  for (auto& b : c.exclude) exc.push_back(span(b));
  std::vector<uint8_t> keep(w);
  size_t n = 0;
  for (size_t y = 0; y < h; ++y) {
    std::fill(keep.begin(), keep.end(), uint8_t(inc.empty()));
    for (auto& b : inc)
      if (y >= b[1] && y < b[3] && b[0] < b[2]) std::fill(keep.begin() + long(b[0]), keep.begin() + long(b[2]), uint8_t(1));
    for (auto& b : exc)
      if (y >= b[1] && y < b[3] && b[0] < b[2]) std::fill(keep.begin() + long(b[0]), keep.begin() + long(b[2]), uint8_t(0));
    for (size_t x = 0; x < w; ++x) {
      float& p = img.pixels[y * w + x];
      if (!keep[x] && !std::isnan(p)) {
        p = std::numeric_limits<float>::quiet_NaN();
        ++n;
      }
    }
  }
  return n;
}

// ---------- statistics ----------
static double median(std::vector<double> v) {
  size_t n = v.size();
  if (n == 0) return NANd;
  std::sort(v.begin(), v.end(), [](double a, double b) { return total_cmp(a, b) < 0; });
  if (n % 2 == 1) return v[n / 2];
  return (v[n / 2 - 1] + v[n / 2]) / 2.0;
}
static double stdev(const std::vector<double>& v) {
  if (v.empty()) return NANd;
  double s = -0.0;
  for (double x : v) s += x;
  double m = s / double(v.size());
  double t = -0.0;
  for (double x : v) t += (x - m) * (x - m);
  return std::sqrt(t / double(v.size()));
}
static std::optional<double> robust_sd(const std::vector<double>& v) {
  if (v.size() < 10) return std::nullopt;
  double m = median(v);
  std::vector<double> d;
  for (double x : v) d.push_back(std::fabs(x - m));
  return 1.4826 * median(d);
}
static double percentile(std::vector<double> a, double q) {
  std::sort(a.begin(), a.end(), [](double x, double y) { return total_cmp(x, y) < 0; });
  if (a.empty()) return NANd;
  double p = q / 100.0 * double(a.size() - 1);
  size_t i = sat_cast<size_t>(std::floor(p));
  if (i + 1 >= a.size()) return a.back();
  return a[i] + (a[i + 1] - a[i]) * (p - double(i));
}
static std::tuple<double, double, size_t> clipped_median(const std::vector<double>& v, double floor) {
  std::vector<bool> keep(v.size(), true);
  std::vector<double> kept = v;
  for (int it = 0; it < 5; ++it) {
    double m = median(kept);
    std::vector<double> d;
    for (double x : kept) d.push_back(std::fabs(x - m));
    double mad = 1.4826 * median(d);
    double thr = 3.0 * fmax_r(mad, floor);
    std::vector<bool> nw(v.size());
    size_t cnt = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      nw[i] = std::fabs(v[i] - m) < thr;
      cnt += nw[i];
    }
    if (nw == keep || cnt < 8) break;
    keep = nw;
    kept.clear();
    for (size_t i = 0; i < v.size(); ++i)
      if (keep[i]) kept.push_back(v[i]);
  }
  return {median(kept), stdev(kept), kept.size()};
}
static std::pair<double, double> clipped_stats(const std::vector<double>& v, double floor) {
  std::vector<double> kept = v;
  for (int it = 0; it < 5; ++it) {
    if (kept.empty()) break;
    double m = median(kept);
    std::vector<double> d;
    for (double x : kept) d.push_back(std::fabs(x - m));
    double thr = 3.0 * fmax_r(1.4826 * median(d), floor);
    std::vector<double> nk;
    for (double x : kept)
      if (std::fabs(x - m) <= thr) nk.push_back(x);
    kept.swap(nk);
  }
  return {median(kept), stdev(kept)};
}
static double reps(std::vector<double> a) {
  std::sort(a.begin(), a.end(), [](double x, double y) { return total_cmp(x, y) < 0; });
  double sum = 0.0;
  size_t i = 0;
  while (i < a.size()) {
    size_t j = i + 1;
    while (j < a.size() && a[j] == a[i]) ++j;
    double k = double(j - i);
    if (j - i > 1) sum += k * (k - 1.0) * (2.0 * k + 5.0);
    i = j;
  }
  return sum;
}
struct TheilSen {
  double slope, lo, hi;
};
static std::optional<TheilSen> theil_sen(const std::vector<double>& x, const std::vector<double>& y) {
  size_t n = x.size();
  std::vector<double> s;
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      double dx = x[i] - x[j];
      if (dx > 0.0) s.push_back((y[i] - y[j]) / dx);
    }
  if (s.empty()) return std::nullopt;
  size_t nt = s.size();
  double slope = median(s);
  double nf = double(n);
  double sigsq = (nf * (nf - 1.0) * (2.0 * nf + 5.0) - reps(x) - reps(y)) / 18.0;
  double z = 1.959963984540054;
  double sigma = std::sqrt(fmax_r(sigsq, 0.0));
  size_t ru = std::min(sat_cast<size_t>(std::nearbyint((double(nt) + z * sigma) / 2.0)), nt - 1);
  int64_t rl_i = sat_cast<int64_t>(std::nearbyint((double(nt) - z * sigma) / 2.0)) - 1;
  size_t rl = size_t(std::max<int64_t>(rl_i, 0));
  std::sort(s.begin(), s.end(), [](double a, double b) { return total_cmp(a, b) < 0; });
  return TheilSen{slope, s[std::min(rl, nt - 1)], s[ru]};
}
static std::vector<double> ranks(const std::vector<double>& v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), size_t(0));
  std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
    int c = total_cmp(v[a], v[b]);
    return c != 0 ? c < 0 : a < b;
  });
  std::vector<double> r(v.size(), 0.0);
  size_t i = 0;
  while (i < idx.size()) {
    size_t j = i + 1;
    while (j < idx.size() && v[idx[j]] == v[idx[i]]) ++j;
    double avg = double(i + j + 1) / 2.0;
    for (size_t k = i; k < j; ++k) r[idx[k]] = avg;
    i = j;
  }
  return r;
}
static double spearman(const std::vector<double>& x, const std::vector<double>& y) {
  auto rx = ranks(x), ry = ranks(y);
  double n = double(rx.size());
  double sx = -0.0, sy = -0.0;
  for (double a : rx) sx += a;
  for (double b : ry) sy += b;
  double mx = sx / n, my = sy / n;
  double sxy = 0, sxx = 0, syy = 0;
  for (size_t i = 0; i < rx.size(); ++i) {
    sxy += (rx[i] - mx) * (ry[i] - my);
    sxx += (rx[i] - mx) * (rx[i] - mx);
    syy += (ry[i] - my) * (ry[i] - my);
  }
  return sxy / std::sqrt(sxx * syy);
}
template <size_t N>
static std::optional<std::array<double, N>> solve_small(std::array<std::array<double, N>, N> a, std::array<double, N> b) {
  for (size_t c = 0; c < N; ++c) {
    size_t p = c;
    for (size_t i = c + 1; i < N; ++i)
      if (total_cmp(std::fabs(a[i][c]), std::fabs(a[p][c])) >= 0) p = i;  // max_by keeps the last maximum
    double pivot = std::fabs(a[p][c]);
    if (std::isnan(pivot) || pivot <= 1e-300) return std::nullopt;
    std::swap(a[c], a[p]);
    std::swap(b[c], b[p]);
    auto top = a[c];
    for (size_t r = c + 1; r < N; ++r) {
      double f = a[r][c] / top[c];
      for (size_t k = c; k < N; ++k) a[r][k] -= f * top[k];
      b[r] -= f * b[c];
    }
  }
  std::array<double, N> x{};
  for (size_t c = N; c-- > 0;) {
    double s = b[c];
    for (size_t k = c + 1; k < N; ++k) s -= a[c][k] * x[k];
    x[c] = s / a[c][c];
  }
  for (double v : x)
    if (!std::isfinite(v)) return std::nullopt;
  return x;
}

struct SplitMix {
  uint64_t s;
  uint64_t next() {
    s += 0x9e3779b97f4a7c15ull;
    uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  double unit() { return double(next() >> 11) / double(uint64_t(1) << 53); }
};

// ---------- points in cells ----------
struct Points {
  std::vector<std::array<double, 3>> p;
  double x0 = 0, y0 = 0, cell = 1;
  size_t nx = 1, ny = 1;
  std::vector<uint32_t> start, items;

  Points() = default;
  Points(std::vector<std::array<double, 3>> pts, size_t w, size_t h, double margin, double cell_) : p(std::move(pts)), cell(cell_) {
    x0 = -margin;
    y0 = -margin;
    nx = std::max<size_t>(sat_cast<size_t>(std::ceil((double(w) + 2.0 * margin) / cell)), 1);
    ny = std::max<size_t>(sat_cast<size_t>(std::ceil((double(h) + 2.0 * margin) / cell)), 1);
    auto cell_of = [&](const std::array<double, 3>& q) -> std::optional<size_t> {
      double cx = (q[0] - x0) / cell, cy = (q[1] - y0) / cell;
      if (cx >= 0.0 && cy >= 0.0 && sat_cast<size_t>(cx) < nx && sat_cast<size_t>(cy) < ny) return sat_cast<size_t>(cy) * nx + sat_cast<size_t>(cx);
      return std::nullopt;
    };
    start.assign(nx * ny + 1, 0);
    for (auto& q : p)
      if (auto c = cell_of(q)) start[*c + 1] += 1;
    for (size_t c = 0; c < nx * ny; ++c) start[c + 1] += start[c];
    auto fill = start;
    items.assign(start[nx * ny], 0);
    for (size_t i = 0; i < p.size(); ++i)
      if (auto c = cell_of(p[i])) items[fill[*c]++] = uint32_t(i);
  }
  template <class F>
  void within(double x, double y, double r, F&& f) const {
    auto span = [&](double c, double o, size_t n) -> std::optional<std::pair<size_t, size_t>> {
      double a = std::floor((c - r - o) / cell), b = std::floor((c + r - o) / cell);
      if (!(b >= 0.0 && a < double(n))) return std::nullopt;
      return std::make_pair(sat_cast<size_t>(fmax_r(a, 0.0)), std::min(sat_cast<size_t>(b), n - 1));
    };
    auto sx = span(x, x0, nx);
    auto sy = span(y, y0, ny);
    if (!sx || !sy) return;
    double r2 = r * r;
    for (size_t cy = sy->first; cy <= sy->second; ++cy)
      for (size_t cx = sx->first; cx <= sx->second; ++cx) {
        size_t c = cy * nx + cx;
        for (size_t k = start[c]; k < start[c + 1]; ++k) {
          uint32_t i = items[k];
          const auto& q = p[i];
          double d2 = (q[0] - x) * (q[0] - x) + (q[1] - y) * (q[1] - y);
          if (d2 <= r2) f(size_t(i), d2);
        }
      }
  }
};

// ---------- estimators ----------
struct PlaneFit {
  double level, gx, gy, s;
  size_t n_keep;
};
static PlaneFit plane_fit(const std::vector<double>& dx, const std::vector<double>& dy, const std::vector<double>& v, double floor) {
  size_t n = v.size();
  std::vector<bool> keep(n, true);
  std::array<double, 3> coef = {median(v), 0.0, 0.0};
  double s = 0.0;
  for (int it = 0; it < 5; ++it) {
    size_t kc = 0;
    for (bool k : keep) kc += k;
    if (kc < 8) break;
    std::array<std::array<double, 3>, 3> a{};
    std::array<double, 3> b{};
    for (size_t i = 0; i < n; ++i) {
      if (!keep[i]) continue;
      double row[3] = {1.0, dx[i], dy[i]};
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) a[size_t(r)][size_t(c)] += row[r] * row[c];
        b[size_t(r)] += row[r] * v[i];
      }
    }
    if (auto c = solve_small<3>(a, b)) coef = *c;
    std::vector<double> res(n), rk;
    for (size_t i = 0; i < n; ++i) res[i] = v[i] - (coef[0] + coef[1] * dx[i] + coef[2] * dy[i]);
    for (size_t i = 0; i < n; ++i)
      if (keep[i]) rk.push_back(res[i]);
    s = stdev(rk);
    double m = median(rk);
    std::vector<double> d;
    for (double r : rk) d.push_back(std::fabs(r - m));
    double mad = 1.4826 * median(d);
    double thr = 3.0 * fmax_r(fmax_r(mad, 0.5 * s), floor);
    std::vector<bool> nw(n);
    for (size_t i = 0; i < n; ++i) nw[i] = std::fabs(res[i]) < thr;
    if (nw == keep) break;
    keep = nw;
  }
  std::vector<double> rk;
  for (size_t i = 0; i < n; ++i)
    if (keep[i]) rk.push_back(v[i] - (coef[0] + coef[1] * dx[i] + coef[2] * dy[i]));
  double shift = rk.empty() ? 0.0 : median(rk);
  return {coef[0] + shift, coef[1], coef[2], s, rk.size()};
}

static std::optional<std::pair<double, double>> centroid(const std::vector<std::array<double, 3>>& pix, double fwhm, double rmax) {
  double sw = fwhm / 2.3548;
  double sw2 = 2.0 * sw * sw;
  double cx = 0, cy = 0;
  for (int it = 0; it < 30; ++it) {
    double s = 0, sx = 0, sy = 0;
    for (auto& q : pix) {
      double ex = q[0] - cx, ey = q[1] - cy;
      double w = std::exp(-(ex * ex + ey * ey) / sw2) * q[2];
      s += w;
      sx += w * ex;
      sy += w * ey;
    }
    if (std::isnan(s) || s <= 0.0) return std::nullopt;
    double stx = clampd(2.0 * sx / s, -1.0, 1.0), sty = clampd(2.0 * sy / s, -1.0, 1.0);
    cx += stx;
    cy += sty;
    if (std::sqrt(cx * cx + cy * cy) > rmax) return std::nullopt;
    if (std::fabs(stx) < 1e-3 && std::fabs(sty) < 1e-3) break;
  }
  return std::make_pair(cx, cy);
}

struct Plain {
  bool edge = false, masked = false;
  double flux = 0, sigma = 0, bgsig = 0, level = 0, pix = 0;
  size_t n = 0;
  double peak = 0, sat_ann = 0;
  std::optional<std::pair<double, double>> off;
  double snr() const { return edge ? NANd : flux / sigma; }
};

struct Dog {
  float s1 = 0, s2 = 0;
  size_t m = 0;
  double c = 0, kappa = 0;
  static Dog make(double fwhm) {
    Dog d;
    d.s1 = fmax_r(float(fwhm / 2.355), 0.7f);
    d.s2 = 3.0f * d.s1;
    d.m = 3 * highpass::box_radius(d.s2);
    double ra = R_AP_DOG * fwhm;
    size_t half = d.m + sat_cast<size_t>(std::ceil(ra)) + 2;
    size_t n = 2 * half + 1;
    auto dog = [&](const std::vector<float>& src) {
      auto a = highpass::box_blur3(src, n, n, d.s1);
      auto b = highpass::box_blur3(src, n, n, d.s2);
      std::vector<double> o(a.size());
      for (size_t i = 0; i < a.size(); ++i) o[i] = double(a[i] - b[i]);
      return o;
    };
    std::vector<float> z(n * n, 0.0f);
    z[half * n + half] = 1.0f;
    auto k = dog(z);
    std::vector<float> ap(n * n, 0.0f);
    size_t n_ap = 0;
    double sp = fwhm / 2.3548;
    std::vector<double> psf(n * n, 0.0);
    for (size_t j = 0; j < n; ++j)
      for (size_t i = 0; i < n; ++i) {
        double dx = double(i) - double(half), dy = double(j) - double(half);
        double r2 = dx * dx + dy * dy;
        if (r2 <= ra * ra) {
          ap[j * n + i] = 1.0f;
          ++n_ap;
        }
        psf[j * n + i] = std::exp(-r2 / (2.0 * sp * sp));
      }
    auto wgt = dog(ap);
    double norm = -0.0;
    for (double v : psf) norm += v;
    double w2 = 0, k2 = 0, w2p = 0, wp = 0;
    for (size_t i = 0; i < n * n; ++i) {
      double p = psf[i] / norm;
      w2 += wgt[i] * wgt[i];
      k2 += k[i] * k[i];
      w2p += wgt[i] * wgt[i] * p;
      wp += wgt[i] * p;
    }
    d.c = std::sqrt(w2 / (k2 * double(n_ap)));
    d.kappa = w2p / wp;
    return d;
  }
};

// ---------- frame model ----------
struct FrameModelImpl {
  size_t w = 0, h = 0;
  ed::Grid grid;
  size_t bin = 1;
  double lsb = 0, fwhm = 0;
  FwhmSource fwhm_source = FwhmSource::Given;
  size_t n_psf = 0;
  double level_sky = 0, sigma_sky = 0, saturation = 0, sat_thr = 0;
  bool linear = false;
  std::optional<double> gain;
  Dog dog;
  Chi chi, chi_dog;
  PerEnv<size_t> n_cal;
  std::optional<ZeroPoint> zero_point;
  std::optional<double> m_lim;
  PerEnv<std::optional<double>> m5_empirical;
  std::array<std::vector<std::pair<double, double>>, 3> curve;
  double self_px = 0;
  Points stars, refs;
  size_t n_refs_measured = 0;

  bool photometric() const { return linear && (!zero_point || zero_point->linear); }
  bool is_flat(const Plain& a) const { return std::isnan(a.pix) || a.pix < fmax_r(FLAT_LSB * lsb, FLAT_SKY * sigma_sky); }
  bool is_bright(const Plain& a) const { return a.level >= level_sky + BRIGHT_LEVEL * (sat_thr - level_sky) || a.sat_ann > BRIGHT_CLIPPED; }
  double depth(double zp, double chi_) const;
  Plain plain(const GrayImage& img, double x, double y, bool with_centroid) const;
  std::optional<std::pair<double, double>> dog_photometry(const GrayImage& img, double x, double y) const;
  bool blended(const Target& t, double flux) const;
  Measurement measure_one(const GrayImage& img, const Target& t) const;
};

std::optional<double> Chi::get(Env e) const {
  std::optional<double> v = e == Env::Sky ? sky : e == Env::Faint ? faint : bright;
  return v ? v : all;
}

double FrameModelImpl::depth(double zp, double chi_) const {
  double f = fwhm;
  double n_a = PI * (R_AP * f) * (R_AP * f);
  double n_ann = PI * (R_OUT * R_OUT - R_IN * R_IN) * f * f;
  double bg2 = sigma_sky * sigma_sky * (n_a + n_a * n_a / n_ann);
  double k = SNR_LIMIT * SNR_LIMIT;
  double a = gain ? k / *gain : 0.0;
  double f5 = (a + std::sqrt(a * a + 4.0 * k * bg2 * chi_ * chi_)) / 2.0;
  return zp - 2.5 * std::log10(f5);
}

Plain FrameModelImpl::plain(const GrayImage& img, double x, double y, bool with_centroid) const {
  double f = fwhm;
  const float* px = img.pixels.data();
  Plain out;
  out.edge = true;
  out.flux = out.sigma = out.bgsig = out.level = out.pix = out.peak = out.sat_ann = NANd;
  double ix = std::floor(x + 0.5), iy = std::floor(y + 0.5);
  if (!(ix >= 0.0 && iy >= 0.0 && ix < double(w) && iy < double(h))) return out;
  out.masked = !std::isfinite(px[size_t(iy) * w + size_t(ix)]);
  double ra = R_AP * f, ri = R_IN * f, ro = R_OUT * f;
  double ra2 = ra * ra, ri2 = ri * ri, ro2 = ro * ro;
  std::vector<double> adx, ady, av;
  std::vector<std::array<double, 3>> cen;
  size_t total = 0;
  bool ap_bad = false;
  double peak = -INF;
  int64_t y0 = sat_cast<int64_t>(std::floor(y - ro)), y1 = sat_cast<int64_t>(std::ceil(y + ro));
  int64_t x0 = sat_cast<int64_t>(std::floor(x - ro)), x1 = sat_cast<int64_t>(std::ceil(x + ro));
  for (int64_t py = y0; py <= y1; ++py) {
    double dy = double(py) - y;
    for (int64_t qx = x0; qx <= x1; ++qx) {
      double dx = double(qx) - x;
      double d2 = dx * dx + dy * dy;
      if (d2 > ro2) continue;
      bool inside = qx >= 0 && py >= 0 && size_t(qx) < w && size_t(py) < h;
      double v = inside ? double(px[size_t(py) * w + size_t(qx)]) : NANd;
      if (d2 >= ri2) {
        ++total;
        if (std::isfinite(v)) {
          adx.push_back(dx);
          ady.push_back(dy);
          av.push_back(v);
        }
      }
      if (d2 <= ra2) {
        if (std::isfinite(v)) peak = fmax_r(peak, v);
        else ap_bad = true;
      }
      if (with_centroid && d2 <= ri2) cen.push_back({dx, dy, v});
    }
  }
  if (ap_bad || double(av.size()) < MIN_ANNULUS * double(total)) return out;
  PlaneFit fit = plane_fit(adx, ady, av, lsb);
  ed::Bg bg;
  bg.c = float(fit.level);
  bg.gx = float(fit.gx);
  bg.gy = float(fit.gy);
  bg.x0 = float(x);
  bg.y0 = float(y);
  auto ap = ed::aperture(px, w, h, x, y, ra, bg);
  if (ap.clipped || ap.n == 0) return out;
  double nf = double(ap.n);
  double s2 = fit.s * fit.s;
  double bgv = nf * s2 + nf * nf * s2 / double(std::max<size_t>(fit.n_keep, 1));
  double var = bgv + (gain ? fmax_r(ap.flux, 0.0) / *gain : 0.0);
  out.edge = false;
  out.flux = ap.flux;
  out.sigma = var > 0.0 ? std::sqrt(var) : NANd;
  out.bgsig = std::sqrt(bgv);
  out.level = fit.level;
  out.pix = fit.s;
  out.n = ap.n;
  out.peak = peak;
  size_t sat_n = 0;
  for (double v : av) sat_n += size_t(v >= sat_thr);
  out.sat_ann = double(sat_n) / double(av.size());
  if (with_centroid) {
    for (auto& c : cen) {
      double b = fit.level + fit.gx * c[0] + fit.gy * c[1];
      c[2] = std::isfinite(c[2]) ? fmax_r(c[2] - b, 0.0) : 0.0;
    }
    out.off = centroid(cen, f, 1.5 * R_IN * f);
  }
  return out;
}

std::optional<std::pair<double, double>> FrameModelImpl::dog_photometry(const GrayImage& img, double x, double y) const {
  int64_t wi = int64_t(w), hi = int64_t(h);
  double f = fwhm;
  double ro = R_OUT_DOG * f;
  int64_t rx0 = sat_cast<int64_t>(std::floor(x - ro)), rx1 = sat_cast<int64_t>(std::ceil(x + ro));
  int64_t ry0 = sat_cast<int64_t>(std::floor(y - ro)), ry1 = sat_cast<int64_t>(std::ceil(y + ro));
  int64_t m = int64_t(dog.m);
  int64_t px0 = std::max<int64_t>(rx0 - m, 0), px1 = std::min<int64_t>(rx1 + m, wi - 1);
  int64_t py0 = std::max<int64_t>(ry0 - m, 0), py1 = std::min<int64_t>(ry1 + m, hi - 1);
  if (px0 > px1 || py0 > py1) return std::nullopt;
  size_t pw = size_t(px1 - px0 + 1), ph = size_t(py1 - py0 + 1);
  std::vector<float> patch(pw * ph, 0.0f);
  std::vector<uint32_t> nan;
  for (size_t j = 0; j < ph; ++j) {
    size_t row = (size_t(py0) + j) * w + size_t(px0);
    for (size_t i = 0; i < pw; ++i) {
      float v = img.pixels[row + i];
      if (std::isfinite(v)) {
        patch[j * pw + i] = v;
      } else {
        if (nan.empty()) nan.assign((pw + 1) * (ph + 1), 0);
        nan[(j + 1) * (pw + 1) + i + 1] = 1;
        patch[j * pw + i] = grid.at_native(float(size_t(px0) + i), float(size_t(py0) + j)).bg;
      }
    }
  }
  if (!nan.empty()) {
    for (size_t j = 1; j <= ph; ++j)
      for (size_t i = 1; i <= pw; ++i) {
        size_t k = j * (pw + 1) + i;
        nan[k] += nan[k - 1] + nan[k - pw - 1] - nan[k - pw - 2];
      }
  }
  auto masked = [&](size_t i, size_t j) -> bool {
    if (nan.empty()) return false;
    size_t mu = dog.m;
    size_t a0 = i >= mu ? i - mu : 0, a1 = std::min(i + mu, pw - 1) + 1;
    size_t b0 = j >= mu ? j - mu : 0, b1 = std::min(j + mu, ph - 1) + 1;
    auto at = [&](size_t a, size_t b) -> int64_t { return int64_t(nan[b * (pw + 1) + a]); };
    return at(a1, b1) + at(a0, b0) - at(a0, b1) - at(a1, b0) > 0;
  };
  auto g1 = highpass::box_blur3(patch, pw, ph, dog.s1);
  auto g2 = highpass::box_blur3(patch, pw, ph, dog.s2);
  double ra2 = (R_AP_DOG * f) * (R_AP_DOG * f), ri2 = (R_IN_DOG * f) * (R_IN_DOG * f), ro2 = ro * ro;
  std::vector<double> ap, ann;
  size_t total = 0;
  for (int64_t qy = ry0; qy <= ry1; ++qy) {
    double dy = double(qy) - y;
    for (int64_t qx = rx0; qx <= rx1; ++qx) {
      double dx = double(qx) - x;
      double d2 = dx * dx + dy * dy;
      bool in_ap = d2 <= ra2, in_ann = d2 >= ri2 && d2 <= ro2;
      if (!in_ap && !in_ann) continue;
      double v;
      if (qx >= 0 && qy >= 0 && qx < wi && qy < hi) {
        size_t i = size_t(qx - px0), j = size_t(qy - py0);
        v = masked(i, j) ? NANd : double(g1[j * pw + i] - g2[j * pw + i]);
      } else {
        v = NANd;
      }
      if (in_ap) {
        if (!std::isfinite(v)) return std::nullopt;
        ap.push_back(v);
      }
      if (in_ann) {
        ++total;
        if (std::isfinite(v)) ann.push_back(v);
      }
    }
  }
  if (ap.empty() || double(ann.size()) < MIN_ANNULUS * double(total)) return std::nullopt;
  auto [lvl, sd, n_ann] = clipped_median(ann, lsb * 1e-3);
  double flux = -0.0;
  for (double v : ap) flux += v - lvl;
  double nb = double(ap.size());
  double c = dog.c;
  double bgv = c * c * sd * sd * (nb + nb * nb * FRAC_PI_2 / double(std::max<size_t>(n_ann, 1)));
  double var = bgv + (gain ? dog.kappa * fmax_r(flux, 0.0) / *gain : 0.0);
  if (var > 0.0) return std::make_pair(flux, std::sqrt(var));
  return std::nullopt;
}

bool FrameModelImpl::blended(const Target& t, double flux) const {
  double r = BLEND_RADIUS * fwhm;
  if (!refs.p.empty() && m_lim) {
    double m5 = *m_lim;
    std::vector<std::pair<double, size_t>> near;
    refs.within(t.x, t.y, r, [&](size_t j, double d2) { near.emplace_back(d2, j); });
    std::sort(near.begin(), near.end(), [](const auto& a, const auto& b) {
      int c = total_cmp(a.first, b.first);
      return c != 0 ? c < 0 : a.second < b.second;
    });
    size_t skip = (!near.empty() && std::sqrt(near[0].first) <= self_px) ? 1 : 0;
    double lim = fmin_r(t.mag.value_or(m5), m5 + BLEND_DEPTH);
    for (size_t i = skip; i < near.size(); ++i)
      if (refs.p[near[i].second][2] < lim) return true;
    return false;
  }
  bool b = false;
  stars.within(t.x, t.y, r, [&](size_t j, double d2) {
    if (std::sqrt(d2) > self_px && stars.p[j][2] > fmax_r(flux, 0.0)) b = true;
  });
  return b;
}

static std::vector<std::pair<double, double>> snr_curve(const std::vector<std::pair<double, double>>& pts) {
  std::vector<std::pair<double, double>> rows;
  double b0 = 8.0;
  while (b0 < 21.5) {
    std::vector<double> s;
    for (auto& p : pts)
      if (p.first >= b0 && p.first < b0 + 0.5) s.push_back(p.second);
    if (s.size() >= 8) rows.emplace_back(b0 + 0.25, median(s));
    b0 += 0.5;
  }
  return rows;
}
static std::optional<double> m5_from_curve(const std::vector<std::pair<double, double>>& rows) {
  for (size_t i = 0; i + 1 < rows.size(); ++i) {
    auto [m0, s0] = rows[i];
    auto [m1, s1] = rows[i + 1];
    if (!(s0 >= SNR_LIMIT && SNR_LIMIT > s1)) continue;
    double t = s1 > 0.0 ? (std::log(s0) - std::log(SNR_LIMIT)) / (std::log(s0) - std::log(s1)) : (s0 - SNR_LIMIT) / (s0 - s1);
    return m0 + t * (m1 - m0);
  }
  return std::nullopt;
}
static std::optional<double> predicted_snr(const std::vector<std::pair<double, double>>& rows, double mag) {
  std::vector<std::pair<double, double>> r;
  for (auto& x : rows)
    if (x.second > 0.0) r.emplace_back(x.first, std::log10(x.second));
  if (r.size() < 2 || !std::isfinite(mag)) return std::nullopt;
  size_t n = r.size();
  double y;
  if (mag <= r[0].first) {
    y = r[0].second;
  } else if (mag >= r[n - 1].first) {
    double slope = (r[n - 1].second - r[n - 2].second) / (r[n - 1].first - r[n - 2].first);
    y = r[n - 1].second + slope * (mag - r[n - 1].first);
  } else {
    size_t k = n - 2;
    for (size_t i = 0; i + 1 < n; ++i)
      if (mag < r[i + 1].first) {
        k = i;
        break;
      }
    auto a = r[k], b = r[k + 1];
    y = a.second + (b.second - a.second) * (mag - a.first) / (b.first - a.first);
  }
  return std::pow(10.0, y);
}

static std::optional<std::pair<double, double>> line_fit(const std::vector<double>& x, const std::vector<double>& y) {
  double n = double(x.size());
  if (x.size() < 2) return std::nullopt;
  double sx = -0.0, sy = -0.0;
  for (double a : x) sx += a;
  for (double b : y) sy += b;
  double mx = sx / n, my = sy / n;
  double sxy = 0, sxx = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    sxy += (x[i] - mx) * (y[i] - my);
    sxx += (x[i] - mx) * (x[i] - mx);
  }
  if (sxx > 0.0) return std::make_pair(sxy / sxx, my - sxy / sxx * mx);
  return std::nullopt;
}

std::optional<ZeroPoint> fit_zero_point(const std::vector<double>& cat, const std::vector<double>& inst) {
  size_t m = std::min(cat.size(), inst.size());
  std::vector<bool> ok(m);
  size_t nok = 0;
  for (size_t i = 0; i < m; ++i) {
    ok[i] = std::isfinite(cat[i]) && std::isfinite(inst[i]);
    nok += ok[i];
  }
  if (nok < 5) return std::nullopt;
  std::vector<double> d(m);
  for (size_t i = 0; i < m; ++i) d[i] = cat[i] - inst[i];
  auto pick = [&](const std::vector<double>& v, const std::vector<bool>& k) {
    std::vector<double> o;
    for (size_t i = 0; i < m; ++i)
      if (k[i]) o.push_back(v[i]);
    return o;
  };
  std::vector<bool> keep = ok;
  for (int it = 0; it < 8; ++it) {
    auto dk = pick(d, keep);
    double med = median(dk);
    for (double& x : dk) x = std::fabs(x - med);
    double mad = 1.4826 * median(dk);
    std::vector<bool> nw(m);
    bool any = false;
    for (size_t i = 0; i < m; ++i) {
      nw[i] = ok[i] && std::fabs(d[i] - med) < fmax_r(3.0 * mad, 0.05);
      any = any || nw[i];
    }
    if (nw == keep || !any) break;
    keep = nw;
  }
  auto dk = pick(d, keep);
  double zp = median(dk);
  for (double& x : dk) x = std::fabs(x - zp);
  double scatter = 1.4826 * median(dk);
  auto ck = pick(cat, keep);
  std::array<double, 2> mag_range = {INF, -INF};
  for (double c : ck) {
    mag_range[0] = fmin_r(mag_range[0], c);
    mag_range[1] = fmax_r(mag_range[1], c);
  }
  std::vector<bool> keep2 = ok;
  std::optional<std::pair<double, double>> fit;
  for (int it = 0; it < 8; ++it) {
    auto lf = line_fit(pick(cat, keep2), pick(inst, keep2));
    if (!lf) break;
    fit = lf;
    double b = lf->first, a = lf->second;
    std::vector<double> res(m);
    for (size_t i = 0; i < m; ++i) res[i] = inst[i] - (a + b * cat[i]);
    auto rk = pick(res, keep2);
    double mm = median(rk);
    for (double& x : rk) x = std::fabs(x - mm);
    double mad = 1.4826 * median(rk);
    std::vector<bool> nw(m);
    size_t cnt = 0;
    for (size_t i = 0; i < m; ++i) {
      nw[i] = ok[i] && std::fabs(res[i]) < fmax_r(3.0 * mad, 0.05);
      cnt += nw[i];
    }
    if (nw == keep2 || cnt < 5) break;
    keep2 = nw;
  }
  std::optional<double> se;
  if (fit) {
    double b = fit->first, a = fit->second;
    auto xs = pick(cat, keep2), ys = pick(inst, keep2);
    double n = double(xs.size());
    double sx = -0.0;
    for (double x : xs) sx += x;
    double mx = sx / n;
    double rss = -0.0, sxx = -0.0;
    for (size_t i = 0; i < xs.size(); ++i) rss += (ys[i] - (a + b * xs[i])) * (ys[i] - (a + b * xs[i]));
    for (double x : xs) sxx += (x - mx) * (x - mx);
    double s = std::sqrt(rss / fmax_r(n - 2.0, 1.0) / sxx);
    if (std::isfinite(s)) se = s;
  }
  ZeroPoint z;
  z.n = ck.size();
  z.zp = zp;
  z.scatter = scatter;
  if (fit) {
    z.slope = fit->first;
    z.intercept = fit->second;
  }
  z.slope_se = se;
  z.mag_range = mag_range;
  z.linear = fit && std::fabs(fit->first - 1.0) <= LINEAR_SLOPE_TOL;
  return z;
}

static std::tuple<size_t, size_t, size_t, size_t> tile_bounds(const ed::Grid& grid, size_t t, size_t w, size_t h) {
  size_t b = grid.b;
  auto xr = grid.ax.range(t % grid.nx);
  auto yr = grid.ay.range(t / grid.nx);
  return {std::min(xr.first * b, w), std::min(yr.first * b, h), std::min(xr.second * b, w), std::min(yr.second * b, h)};
}

static std::optional<double> ptc_gain(const float* px, size_t w, size_t h, const ed::Grid& grid, const std::vector<bool>& live, double level_sky,
                                      double saturation, double lsb) {
  size_t nt = grid.nx * grid.ny;
  std::vector<std::optional<std::pair<double, double>>> res(nt);
  parallel_for(nt, [&](size_t t) {
    double level = double(grid.bg[t]);
    if (!live[t] || std::isnan(level) || level >= level_sky + 0.8 * (saturation - level_sky)) return;
    auto [x0, y0, x1, y1] = tile_bounds(grid, t, w, h);
    std::vector<double> d;
    d.reserve(1100);
    size_t r = 0;
    for (size_t y = y0; y < y1; y += 2, ++r)
      for (size_t x = x0 + (r & 1); x < std::min(x1, w - 1); x += 2) {
        float a = px[y * w + x], b = px[y * w + x + 1];
        if (std::isfinite(a) && std::isfinite(b)) d.push_back(double(b - a));
      }
    if (d.size() < 64) return;
    double s = clipped_stats(d, lsb).second / SQRT_2;
    if (std::isfinite(s)) res[t] = std::make_pair(level - level_sky, s * s);
  });
  std::vector<std::pair<double, double>> pts;
  for (auto& r : res)
    if (r) pts.push_back(*r);
  if (pts.size() < 20) return std::nullopt;
  if (pts.size() > 1500) {
    size_t n = pts.size();
    std::vector<std::pair<double, double>> sub;
    for (size_t k = 0; k < 1500; ++k) sub.push_back(pts[k * n / 1500]);
    pts.swap(sub);
  }
  std::vector<double> x, y;
  for (auto& p : pts) {
    x.push_back(p.first);
    y.push_back(p.second);
  }
  auto ts = theil_sen(x, y);
  if (!ts) return std::nullopt;
  double rho = spearman(x, y);
  if (ts->slope > 0.0 && ts->lo > 0.0 && rho > 0.3) return 1.0 / ts->slope;
  return std::nullopt;
}

struct Gauss {
  double fwhm, a, x, y;
};
static std::optional<Gauss> gauss_fit(const std::vector<double>& cut, size_t n, double s0) {
  double c = double(n - 1) / 2.0;
  std::vector<double> border;
  for (size_t i = 0; i < n; ++i) {
    border.push_back(cut[i]);
    border.push_back(cut[(n - 1) * n + i]);
  }
  for (size_t i = 0; i < n; ++i) {
    border.push_back(cut[i * n]);
    border.push_back(cut[i * n + n - 1]);
  }
  double b0 = median(border);
  double mx = -INF;
  for (double v : cut) mx = fmax_r(mx, v);
  double a0 = mx - b0;
  if (std::isnan(a0) || a0 <= 0.0) return std::nullopt;
  const double lo[5] = {-INF, 0.0, c - 3.0, c - 3.0, 0.3};
  const double hi[5] = {INF, INF, c + 3.0, c + 3.0, 6.0 * s0 + 3.0};
  using P5 = std::array<double, 5>;
  auto clampp = [&](P5 p) {
    for (size_t k = 0; k < 5; ++k) p[k] = clampd(p[k], lo[k], hi[k]);
    return p;
  };
  auto cost = [&](const P5& p) {
    double s = 0.0;
    for (size_t j = 0; j < n; ++j)
      for (size_t i = 0; i < n; ++i) {
        double dx = double(i) - p[2], dy = double(j) - p[3];
        double e = std::exp(-(dx * dx + dy * dy) / (2.0 * p[4] * p[4]));
        double r = p[0] + p[1] * e - cut[j * n + i];
        s += r * r;
      }
    return s;
  };
  P5 p = clampp({b0, a0, c, c, s0});
  double cur = cost(p);
  double lambda = 1e-3;
  for (int it = 0; it < 200; ++it) {
    std::array<std::array<double, 5>, 5> jtj{};
    std::array<double, 5> jtr{};
    for (size_t j = 0; j < n; ++j)
      for (size_t i = 0; i < n; ++i) {
        double dx = double(i) - p[2], dy = double(j) - p[3];
        double s2 = p[4] * p[4];
        double e = std::exp(-(dx * dx + dy * dy) / (2.0 * s2));
        double r = p[0] + p[1] * e - cut[j * n + i];
        double ae = p[1] * e;
        double g[5] = {1.0, e, ae * dx / s2, ae * dy / s2, ae * (dx * dx + dy * dy) / (s2 * p[4])};
        for (size_t a = 0; a < 5; ++a) {
          for (size_t b = 0; b < 5; ++b) jtj[a][b] += g[a] * g[b];
          jtr[a] += g[a] * r;
        }
      }
    bool improved = false;
    while (lambda < 1e12) {
      auto m = jtj;
      for (size_t k = 0; k < 5; ++k) m[k][k] += lambda * fmax_r(jtj[k][k], 1e-12);
      std::array<double, 5> nb;
      for (size_t k = 0; k < 5; ++k) nb[k] = -jtr[k];
      auto d = solve_small<5>(m, nb);
      if (!d) {
        lambda *= 10.0;
        continue;
      }
      P5 q;
      for (size_t k = 0; k < 5; ++k) q[k] = p[k] + (*d)[k];
      q = clampp(q);
      double next = cost(q);
      if (next < cur) {
        bool done = cur - next <= 1e-12 * fmax_r(cur, 1e-300);
        p = q;
        cur = next;
        lambda = fmax_r(lambda / 10.0, 1e-12);
        improved = !done;
        break;
      }
      lambda *= 10.0;
    }
    if (!improved) break;
  }
  for (double v : p)
    if (!std::isfinite(v)) return std::nullopt;
  return Gauss{2.3548 * p[4], p[1], p[2] - c, p[3] - c};
}

static std::vector<double> psf_fits(const GrayImage& img, const Points& det, const Points& refs, double guess, double amp, double sigma_sky,
                                    double sat_thr, size_t max) {
  size_t w = img.width, h = img.height;
  bool by_mag = !refs.p.empty();
  const Points& src = by_mag ? refs : det;
  std::vector<size_t> order(src.p.size());
  std::iota(order.begin(), order.end(), size_t(0));
  if (by_mag)
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      int c = total_cmp(src.p[a][2], src.p[b][2]);
      return c != 0 ? c < 0 : a < b;
    });
  else
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      int c = total_cmp(src.p[b][2], src.p[a][2]);
      return c != 0 ? c < 0 : a < b;
    });
  size_t half = sat_cast<size_t>(std::ceil(2.5 * guess)) + 2;
  size_t n = 2 * half + 1;
  double s0 = guess / 2.3548;
  std::vector<double> out;
  auto fit_one = [&](size_t k) -> std::optional<double> {
    const auto& q = src.p[k];
    double ix = std::round(q[0]), iy = std::round(q[1]);
    double hf = double(half);
    if (!(ix >= hf && iy >= hf && ix + hf < double(w) && iy + hf < double(h))) return std::nullopt;
    bool isolated = true;
    src.within(q[0], q[1], 4.0 * guess, [&](size_t j, double) {
      bool brighter = by_mag ? src.p[j][2] < q[2] + 3.0 : src.p[j][2] > q[2] / 15.85;
      if (j != k && brighter) isolated = false;
    });
    if (!isolated) return std::nullopt;
    size_t uix = size_t(ix), uiy = size_t(iy);
    std::vector<double> cut;
    cut.reserve(n * n);
    for (size_t y = uiy - half; y <= uiy + half; ++y)
      for (size_t x = uix - half; x <= uix + half; ++x) {
        float v = img.pixels[y * w + x];
        if (!std::isfinite(v) || double(v) >= sat_thr) return std::nullopt;
        cut.push_back(double(v));
      }
    auto g = gauss_fit(cut, n, s0);
    if (!g) return std::nullopt;
    double off = std::hypot(g->x + double(uix) - q[0], g->y + double(uiy) - q[1]);
    if (g->a > amp * sigma_sky && off < guess) return g->fwhm;
    return std::nullopt;
  };
  for (size_t b0 = 0; b0 < order.size(); b0 += 256) {
    size_t bn = std::min<size_t>(256, order.size() - b0);
    std::vector<std::optional<double>> got(bn);
    parallel_for(bn, [&](size_t i) { got[i] = fit_one(order[b0 + i]); });
    for (auto& g : got)
      if (g && out.size() < max) out.push_back(*g);
    if (out.size() >= max) break;
  }
  return out;
}

// ---------- FrameModel ----------
FrameModel::FrameModel() = default;
FrameModel::~FrameModel() = default;
FrameModel::FrameModel(FrameModel&&) noexcept = default;
FrameModel& FrameModel::operator=(FrameModel&&) noexcept = default;
double FrameModel::fwhm() const { return impl_->fwhm; }
bool FrameModel::photometric() const { return impl_->photometric(); }

FrameModel FrameModel::create(const GrayImage& img, const SourceInfo& info, const std::vector<Star>& stars, const std::vector<RefStar>& refs_in,
                              std::optional<double> scale_arcsec, const FrameOptions& opts) {
  size_t w = img.width, h = img.height;
  if (w < 16 || h < 16 || img.pixels.size() < w * h) throw MeasureError("no_frame", "The image is too small to measure.");
  if (opts.fwhm && !(*opts.fwhm >= MIN_FWHM && *opts.fwhm <= MAX_FWHM))
    throw MeasureError("bad_request", "frame.fwhm must lie in [0.5, 64] px");
  const float* px = img.pixels.data();
  ExtractOptions eopts = ExtractOptions::for_source(info);
  size_t bin = bin_factor(w, h, eopts);
  float lsb_f = info.lsb > 0.0f ? info.lsb : detect_lattice(img.pixels);
  ed::Plane plane;
  ed::bin_plane(img, bin, plane);
  if (plane.w < 4 || plane.h < 4) throw MeasureError("no_frame", "The image is too small to measure.");
  auto grido = ed::Grid::build(plane, img.pixels, w, bin, lsb_f);
  if (!grido) throw MeasureError("no_frame", "The image has no usable pixels.");
  FrameModel fm;
  fm.impl_ = std::make_unique<FrameModelImpl>();
  FrameModelImpl& M = *fm.impl_;
  M.grid = std::move(*grido);
  const ed::Grid& grid = M.grid;
  double lsb = double(lsb_f);
  size_t nt = grid.nx * grid.ny;
  std::vector<uint8_t> live8(nt);
  parallel_for(nt, [&](size_t t) {
    auto [x0, y0, x1, y1] = tile_bounds(grid, t, w, h);
    size_t n = 0, fin = 0;
    for (size_t y = y0; y < y1; y += 2)
      for (size_t x = x0; x < x1; x += 2) {
        ++n;
        fin += size_t(std::isfinite(px[y * w + x]));
      }
    live8[t] = n > 0 && 2 * fin >= n;
  });
  std::vector<bool> live(live8.begin(), live8.end());
  std::vector<double> levels;
  for (size_t t = 0; t < nt; ++t)
    if (live[t] && std::isfinite(grid.bg[t])) levels.push_back(double(grid.bg[t]));
  if (levels.empty()) throw MeasureError("no_frame", "The image has no usable pixels.");
  double p30 = percentile(levels, 30.0);
  std::vector<double> lb, ls;
  for (size_t t = 0; t < nt; ++t)
    if (live[t] && double(grid.bg[t]) <= p30) {
      lb.push_back(double(grid.bg[t]));
      ls.push_back(double(grid.sig_n[t]));
    }
  double level_sky = median(lb), sigma_sky = median(ls);
  if (!(sigma_sky > 0.0 && std::isfinite(level_sky))) throw MeasureError("no_frame", "The image has no measurable sky noise.");
  float satf = -INF;
  for (size_t i = 0; i < w * h; ++i)
    if (std::isfinite(px[i])) satf = fmax_r(satf, px[i]);
  double saturation = double(satf);
  double sat_thr = level_sky + SAT_FRACTION * (saturation - level_sky);
  bool linear = opts.linear.value_or(info.format == image_io::SourceFormat::Fits);

  double guess;
  {
    std::vector<double> f;
    for (const Star& s : stars)
      if (s.snr > PSF_MIN_SNR && !s.saturated && std::isfinite(s.fwhm)) f.push_back(s.fwhm);
    double g = median(f);
    guess = std::isfinite(g) ? clampd(g, MIN_FWHM, MAX_FWHM) : 3.0;
  }
  auto reach = [](double f) { return fmax_r(fmax_r(R_OUT + R_AP, R_OUT_DOG), 4.0) * f + 8.0; };
  double margin = reach(fmax_r(guess, opts.fwhm.value_or(0.0)));
  double cell = fmax_r(2.0 * guess, 16.0);
  std::vector<std::array<double, 3>> dp, rp;
  for (const Star& s : stars)
    if (std::isfinite(s.x) && std::isfinite(s.y)) dp.push_back({s.x, s.y, s.flux});
  for (const RefStar& r : refs_in)
    if (std::isfinite(r.x) && std::isfinite(r.y) && std::isfinite(r.mag)) rp.push_back({r.x, r.y, r.mag});
  Points det(std::move(dp), w, h, margin, cell);
  Points refs(std::move(rp), w, h, margin, cell);

  double fwhm;
  FwhmSource fsrc;
  size_t n_psf;
  if (opts.fwhm) {
    fwhm = *opts.fwhm;
    fsrc = FwhmSource::Given;
    n_psf = 0;
  } else {
    auto fits = [&](double g, double amp) { return psf_fits(img, det, refs, g, amp, sigma_sky, sat_thr, N_PSF); };
    auto first = fits(guess, PSF_AMP);
    std::vector<double> list;
    if (first.size() >= PSF_MIN_FITS) {
      list = first;
      fsrc = FwhmSource::Fit;
    } else {
      double seed = first.empty() ? guess : median(first);
      list = fits(seed, PSF_AMP_RELAXED);
      fsrc = FwhmSource::FitRelaxed;
    }
    if (list.size() < PSF_MIN_FITS_RELAXED)
      throw MeasureError("no_psf", "Too few bright isolated stars to measure the frame FWHM; give frame.fwhm.");
    n_psf = list.size();
    double f = median(list);
    if (!(f >= MIN_FWHM && f <= MAX_FWHM)) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "The fitted frame FWHM (%.2f px) is out of range; give frame.fwhm.", f);
      throw MeasureError("no_psf", buf);
    }
    fwhm = f;
  }
  std::optional<double> gain;
  if (linear) gain = ptc_gain(px, w, h, grid, live, level_sky, saturation, lsb);
  double self_px = (scale_arcsec && *scale_arcsec > 0.0 && std::isfinite(*scale_arcsec)) ? SELF_ARCSEC / *scale_arcsec : SELF_FWHM * fwhm;
  M.w = w;
  M.h = h;
  M.bin = bin;
  M.lsb = lsb;
  M.fwhm = fwhm;
  M.fwhm_source = fsrc;
  M.n_psf = n_psf;
  M.level_sky = level_sky;
  M.sigma_sky = sigma_sky;
  M.saturation = saturation;
  M.sat_thr = sat_thr;
  M.linear = linear;
  M.gain = gain;
  M.dog = Dog::make(fwhm);
  M.self_px = self_px;
  M.stars = std::move(det);
  M.refs = std::move(refs);

  std::vector<size_t> measured;
  {
    for (size_t i = 0; i < M.refs.p.size(); ++i) {
      const auto& q = M.refs.p[i];
      if (q[0] >= 0.0 && q[1] >= 0.0 && q[0] <= double(w - 1) && q[1] <= double(h - 1)) measured.push_back(i);
    }
    std::sort(measured.begin(), measured.end(), [&](size_t a, size_t b) {
      int c = total_cmp(M.refs.p[a][2], M.refs.p[b][2]);
      return c != 0 ? c < 0 : a < b;
    });
    if (measured.size() > MAX_REFS_MEASURED) {
      size_t half2 = MAX_REFS_MEASURED / 2;
      size_t rest = measured.size() - half2, take = MAX_REFS_MEASURED - half2;
      std::vector<size_t> sample;
      for (size_t k = 0; k < take; ++k) sample.push_back(measured[half2 + k * rest / take]);
      measured.resize(half2);
      measured.insert(measured.end(), sample.begin(), sample.end());
    }
  }
  M.n_refs_measured = measured.size();
  std::vector<Plain> ref_plain(measured.size());
  parallel_for(measured.size(), [&](size_t k) { ref_plain[k] = M.plain(img, M.refs.p[measured[k]][0], M.refs.p[measured[k]][1], false); });
  std::vector<uint8_t> iso(measured.size());
  parallel_for(measured.size(), [&](size_t k) {
    size_t i = measured[k];
    auto q = M.refs.p[i];
    bool ok = true;
    M.refs.within(q[0], q[1], (R_OUT + R_AP) * fwhm, [&](size_t j, double) {
      if (j != i && M.refs.p[j][2] < q[2] + 2.5) ok = false;
    });
    iso[k] = ok;
  });
  std::vector<bool> usable(measured.size());
  for (size_t k = 0; k < measured.size(); ++k) {
    const Plain& a = ref_plain[k];
    usable[k] = !a.edge && a.peak < M.sat_thr && !M.is_flat(a) && !M.is_bright(a);
  }
  std::vector<double> zc, zi;
  for (size_t k = 0; k < measured.size(); ++k) {
    const Plain& a = ref_plain[k];
    if (iso[k] && usable[k] && a.snr() >= ZP_MIN_SNR && a.flux > 0.0) {
      zc.push_back(M.refs.p[measured[k]][2]);
      zi.push_back(-2.5 * std::log10(a.flux));
    }
  }
  M.zero_point = fit_zero_point(zc, zi);
  std::optional<double> m5_formula;
  if (M.zero_point) m5_formula = M.depth(M.zero_point->zp, 1.0);

  SplitMix rng{CAL_SEED};
  std::array<std::vector<Vec2>, 3> picked;
  for (size_t it = 0; it < N_CAL_CANDIDATES; ++it) {
    bool all = true;
    for (auto& p : picked) all = all && p.size() >= N_CAL;
    if (all) break;
    double x = rng.unit() * double(w - 1), y = rng.unit() * double(h - 1);
    size_t ix = sat_cast<size_t>(x + 0.5), iy = sat_cast<size_t>(y + 0.5);
    if (!std::isfinite(px[std::min(iy, h - 1) * w + std::min(ix, w - 1)])) continue;
    double bg = double(M.grid.at_native(float(x), float(y)).bg);
    if (auto e = env_of(bg, level_sky, sigma_sky))
      if (picked[std::to_underlying(*e)].size() < N_CAL) picked[std::to_underlying(*e)].push_back({x, y});
  }
  std::vector<Vec2> cal;
  for (auto& p : picked) cal.insert(cal.end(), p.begin(), p.end());
  std::vector<Plain> cal_a(cal.size());
  std::vector<std::optional<std::pair<double, double>>> cal_d(cal.size());
  parallel_for(cal.size(), [&](size_t i) {
    cal_a[i] = M.plain(img, cal[i][0], cal[i][1], false);
    if (!cal_a[i].edge) cal_d[i] = M.dog_photometry(img, cal[i][0], cal[i][1]);
  });
  std::array<std::vector<double>, 3> snr_a, snr_b;
  for (size_t i = 0; i < cal.size(); ++i) {
    double x = cal[i][0], y = cal[i][1];
    const Plain& a = cal_a[i];
    bool empty = true;
    M.stars.within(x, y, CAL_EMPTY * fwhm, [&](size_t, double) { empty = false; });
    if (m5_formula) {
      M.refs.within(x, y, CAL_EMPTY * fwhm, [&](size_t j, double) {
        if (M.refs.p[j][2] < *m5_formula + 1.0) empty = false;
      });
    }
    double snr = a.snr();
    if (!(empty && !a.edge && std::isfinite(snr) && !M.is_flat(a) && !M.is_bright(a))) continue;
    auto e = env_of(a.level, level_sky, sigma_sky);
    if (!e) continue;
    snr_a[std::to_underlying(*e)].push_back(snr);
    if (cal_d[i]) {
      double sb = cal_d[i]->first / cal_d[i]->second;
      if (std::isfinite(sb)) snr_b[std::to_underlying(*e)].push_back(sb);
    }
  }
  auto chi_of = [&](const std::array<std::vector<double>, 3>& v) {
    auto env = [&](size_t k) { return v[k].size() >= CAL_MIN_ENV ? robust_sd(v[k]) : std::nullopt; };
    std::vector<double> all;
    for (auto& x : v) all.insert(all.end(), x.begin(), x.end());
    Chi c;
    c.sky = env(0);
    c.faint = env(1);
    c.bright = env(2);
    if (all.size() >= CAL_MIN) c.all = robust_sd(all);
    return c;
  };
  M.chi = chi_of(snr_a);
  M.chi_dog = chi_of(snr_b);
  M.n_cal = PerEnv<size_t>{snr_a[0].size(), snr_a[1].size(), snr_a[2].size()};
  if (M.zero_point) {
    if (auto c = M.chi.get(Env::Sky)) M.m_lim = M.depth(M.zero_point->zp, *c);
  }
  std::array<std::vector<std::pair<double, double>>, 3> pts;
  for (size_t k = 0; k < measured.size(); ++k) {
    const Plain& a = ref_plain[k];
    if (!iso[k] || !usable[k]) continue;
    auto e = env_of(a.level, level_sky, sigma_sky);
    double snr = a.snr();
    if (!e) continue;
    if (auto c = M.chi.get(*e); c && std::isfinite(snr)) pts[std::to_underlying(*e)].emplace_back(M.refs.p[measured[k]][2], snr / *c);
  }
  for (size_t e = 0; e < 3; ++e) M.curve[e] = snr_curve(pts[e]);
  M.m5_empirical.sky = m5_from_curve(M.curve[0]);
  M.m5_empirical.faint = m5_from_curve(M.curve[1]);
  M.m5_empirical.bright = m5_from_curve(M.curve[2]);
  return fm;
}

ModelSummary FrameModel::summary() const {
  const FrameModelImpl& M = *impl_;
  ModelSummary s;
  s.width = uint32_t(M.w);
  s.height = uint32_t(M.h);
  s.fwhm = M.fwhm;
  s.fwhm_source = M.fwhm_source;
  s.n_psf = M.n_psf;
  s.bin = M.bin;
  s.lsb = M.lsb;
  s.level_sky = M.level_sky;
  s.sigma_sky = M.sigma_sky;
  s.saturation = M.saturation;
  s.sat_thr = M.sat_thr;
  s.linear = M.linear;
  s.photometric = M.photometric();
  s.gain = M.gain;
  s.chi = M.chi;
  s.chi_dog = M.chi_dog;
  s.n_cal = M.n_cal;
  s.zero_point = M.zero_point;
  s.m_lim = M.m_lim;
  s.m5_empirical = M.m5_empirical;
  auto c = [&](size_t k) {
    std::vector<std::array<double, 2>> v;
    for (auto& p : M.curve[k]) v.push_back({p.first, p.second});
    return v;
  };
  s.snr_curve = {c(0), c(1), c(2)};
  s.n_stars = M.stars.p.size();
  s.n_refs = M.refs.p.size();
  s.n_refs_measured = M.n_refs_measured;
  s.self_px = M.self_px;
  return s;
}

Measurement FrameModelImpl::measure_one(const GrayImage& img, const Target& t) const {
  Plain a = plain(img, t.x, t.y, true);
  std::optional<std::pair<double, double>> d;
  if (!a.edge) d = dog_photometry(img, t.x, t.y);
  std::optional<Env> env;
  if (!a.edge) env = env_of(a.level, level_sky, sigma_sky);
  std::optional<double> c, c_dog;
  if (env) {
    c = chi.get(*env);
    c_dog = chi_dog.get(*env);
  }
  double snr = a.snr();
  std::optional<double> snr_cal;
  if (c) snr_cal = snr / *c;
  std::optional<double> snr_dog;
  if (d && c_dog) snr_dog = d->first / d->second / *c_dog;
  uint32_t fl = 0;
  if (!photometric()) fl |= flags::NONLINEAR;
  Status status = Status::Below;
  bool rec_snr = snr_cal && *snr_cal >= SNR_RECORDED;
  bool confirmed = snr_dog && *snr_dog >= SNR_CONFIRM;
  if (snr_cal && *snr_cal >= SNR_TRACE) status = Status::Trace;
  if (rec_snr && !confirmed) fl |= flags::UNCONFIRMED;
  if (rec_snr && confirmed) {
    bool near = a.off && std::sqrt(a.off->first * a.off->first + a.off->second * a.off->second) / fwhm < OFFSET_MAX;
    if (near) {
      status = Status::Recorded;
    } else {
      status = Status::Below;
      fl |= flags::OFFSET;
    }
  }
  if (!a.edge && !d) fl |= flags::NO_DOG;
  if (a.edge) {
    fl |= flags::NEAR_EDGE;
  } else {
    if (is_flat(a) || !std::isfinite(snr)) fl |= flags::NO_DATA;
    if (is_bright(a)) fl |= flags::BRIGHT_BG;
    if (a.peak >= sat_thr) fl |= flags::SATURATED;
    if (!c) fl |= flags::UNCALIBRATED;
  }
  if (a.masked) fl |= flags::NO_DATA;
  if (blended(t, a.edge ? 0.0 : a.flux)) fl |= flags::BLENDED;
  if (fl & flags::NOT_ASSESSABLE) status = Status::NotAssessable;
  if (status == Status::Recorded && env && t.mag) {
    auto p = predicted_snr(curve[std::to_underlying(*env)], *t.mag);
    if (p && *p < SUSPECT_SNR) fl |= flags::SUSPECT;
  }
  auto fin = [](double v) -> std::optional<double> { return std::isfinite(v) ? std::optional<double>(v) : std::nullopt; };
  Measurement m;
  m.x = t.x;
  m.y = t.y;
  if (!a.edge) {
    m.flux = fin(a.flux);
    m.sigma = fin(a.sigma);
    m.bg = fin(a.level);
  }
  m.snr = fin(snr);
  if (snr_cal) m.snr_cal = fin(*snr_cal);
  if (snr_dog) m.snr_dog = fin(*snr_dog);
  if (a.off) {
    m.dx = a.off->first;
    m.dy = a.off->second;
  }
  m.n_pix = a.n;
  m.env = env;
  m.status = status;
  m.flags = fl;
  return m;
}

Measurement FrameModel::measure_one(const GrayImage& img, const Target& t) const { return impl_->measure_one(img, t); }

std::vector<Measurement> measure(const GrayImage& img, const FrameModel& model, const std::vector<Target>& targets) {
  std::vector<Measurement> out(targets.size());
  parallel_for(targets.size(), [&](size_t i) { out[i] = model.measure_one(img, targets[i]); });
  return out;
}

// ---------- JSON ----------
static Json opt_num(const std::optional<double>& v) { return v ? Json(*v) : Json(); }

Json Content::to_json() const {
  Json o = Json::object();
  Json inc = Json::array(), exc = Json::array();
  for (auto& b : include) inc.push(Json::from_arr(b));
  for (auto& b : exclude) exc.push(Json::from_arr(b));
  o.set("include", std::move(inc)).set("exclude", std::move(exc));
  return o;
}
static void check_fields(const Json& j, std::initializer_list<const char*> allowed, const char* what) {
  if (!j.is_object()) throw Error(std::string("invalid ") + what);
  for (auto& kv : j.members()) {
    bool ok = false;
    for (const char* a : allowed) ok = ok || kv.first == a;
    if (!ok) throw Error("unknown field `" + kv.first + "` in " + what);
  }
}
static std::vector<std::array<double, 4>> boxes_from(const Json* j) {
  std::vector<std::array<double, 4>> out;
  if (!j || j->is_null()) return out;
  if (!j->is_array()) throw Error("invalid box list");
  for (auto& b : j->items()) {
    if (!b.is_array() || b.items().size() != 4) throw Error("invalid box");
    std::array<double, 4> a;
    for (size_t i = 0; i < 4; ++i) {
      if (!b.items()[i].is_number()) throw Error("invalid box");
      a[i] = b.items()[i].as_f64();
    }
    out.push_back(a);
  }
  return out;
}
Content Content::from_json(const Json& j) {
  check_fields(j, {"include", "exclude"}, "content");
  Content c;
  c.include = boxes_from(j.get("include"));
  c.exclude = boxes_from(j.get("exclude"));
  return c;
}
Json FrameOptions::to_json() const {
  Json o = Json::object();
  o.set("fwhm", opt_num(fwhm));
  o.set("linear", linear ? Json(*linear) : Json());
  o.set("content", content ? content->to_json() : Json());
  return o;
}
FrameOptions FrameOptions::from_json(const Json& j) {
  check_fields(j, {"fwhm", "linear", "content"}, "frame");
  FrameOptions f;
  if (auto* v = j.get("fwhm"); v && !v->is_null()) {
    if (!v->is_number()) throw Error("invalid frame.fwhm");
    f.fwhm = v->as_f64();
  }
  if (auto* v = j.get("linear"); v && !v->is_null()) {
    if (!v->is_bool()) throw Error("invalid frame.linear");
    f.linear = v->as_bool();
  }
  if (auto* v = j.get("content"); v && !v->is_null()) f.content = Content::from_json(*v);
  return f;
}
Json Point::to_json() const {
  Json o = Json::object();
  if (x) o.set("x", *x);
  if (y) o.set("y", *y);
  if (ra) o.set("ra", *ra);
  if (dec) o.set("dec", *dec);
  if (mag) o.set("mag", *mag);
  return o;
}
Point Point::from_json(const Json& j) {
  check_fields(j, {"x", "y", "ra", "dec", "mag"}, "point");
  Point p;
  auto get = [&](const char* k, std::optional<double>& dst) {
    if (auto* v = j.get(k); v && !v->is_null()) {
      if (!v->is_number()) throw Error(std::string("invalid ") + k);
      dst = v->as_f64();
    }
  };
  get("x", p.x);
  get("y", p.y);
  get("ra", p.ra);
  get("dec", p.dec);
  get("mag", p.mag);
  return p;
}
Request Request::from_json(const Json& j) {
  check_fields(j, {"wcs", "targets", "refs", "frame"}, "request");
  Request r;
  if (auto* v = j.get("wcs"); v && !v->is_null()) r.wcs = Wcs::from_json(*v);
  auto pts = [&](const char* k, std::vector<Point>& dst) {
    if (auto* v = j.get(k)) {
      if (!v->is_array()) throw Error(std::string("invalid ") + k);
      for (auto& p : v->items()) dst.push_back(Point::from_json(p));
    }
  };
  pts("targets", r.targets);
  pts("refs", r.refs);
  if (auto* v = j.get("frame")) r.frame = FrameOptions::from_json(*v);
  return r;
}
Json ZeroPoint::to_json() const {
  Json o = Json::object();
  o.set("n", uint64_t(n)).set("zp", zp).set("scatter", scatter).set("slope", opt_num(slope)).set("slope_se", opt_num(slope_se));
  o.set("intercept", opt_num(intercept)).set("mag_range", Json::from_arr(mag_range)).set("linear", linear);
  return o;
}
Json Chi::to_json() const {
  Json o = Json::object();
  o.set("sky", opt_num(sky)).set("faint", opt_num(faint)).set("bright", opt_num(bright)).set("all", opt_num(all));
  return o;
}
Json ModelSummary::to_json() const {
  Json o = Json::object();
  o.set("width", width).set("height", height).set("fwhm", fwhm).set("fwhm_source", fwhm_source_name(fwhm_source));
  o.set("n_psf", uint64_t(n_psf)).set("bin", uint64_t(bin)).set("lsb", lsb).set("level_sky", level_sky).set("sigma_sky", sigma_sky);
  o.set("saturation", saturation).set("sat_thr", sat_thr).set("linear", linear).set("photometric", photometric).set("gain", opt_num(gain));
  o.set("chi", chi.to_json()).set("chi_dog", chi_dog.to_json());
  Json nc = Json::object();
  nc.set("sky", uint64_t(n_cal.sky)).set("faint", uint64_t(n_cal.faint)).set("bright", uint64_t(n_cal.bright));
  o.set("n_cal", std::move(nc));
  o.set("zero_point", zero_point ? zero_point->to_json() : Json());
  o.set("m_lim", opt_num(m_lim));
  Json m5 = Json::object();
  m5.set("sky", opt_num(m5_empirical.sky)).set("faint", opt_num(m5_empirical.faint)).set("bright", opt_num(m5_empirical.bright));
  o.set("m5_empirical", std::move(m5));
  auto curve = [](const std::vector<std::array<double, 2>>& v) {
    Json a = Json::array();
    for (auto& p : v) a.push(Json::from_arr(p));
    return a;
  };
  Json sc = Json::object();
  sc.set("sky", curve(snr_curve.sky)).set("faint", curve(snr_curve.faint)).set("bright", curve(snr_curve.bright));
  o.set("snr_curve", std::move(sc));
  o.set("n_stars", uint64_t(n_stars)).set("n_refs", uint64_t(n_refs)).set("n_refs_measured", uint64_t(n_refs_measured)).set("self_px", self_px);
  return o;
}
Json Measurement::to_json() const {
  Json r = Json::array();
  r.push(x).push(y).push(opt_num(flux)).push(opt_num(sigma)).push(opt_num(snr)).push(opt_num(snr_cal)).push(opt_num(snr_dog));
  r.push(opt_num(dx)).push(opt_num(dy)).push(opt_num(bg)).push(uint64_t(n_pix));
  r.push(env ? Json(env_name(*env)) : Json()).push(status_name(status)).push(flags);
  return r;
}
Json Response::to_json() const {
  Json o = Json::object();
  o.set("model", model.to_json());
  Json c = Json::object();
  c.set("recorded", uint64_t(counts.recorded)).set("trace", uint64_t(counts.trace)).set("below", uint64_t(counts.below));
  c.set("not_assessable", uint64_t(counts.not_assessable));
  o.set("counts", std::move(c));
  Json cols = Json::array();
  for (const char* n : COLUMNS) cols.push(n);
  o.set("columns", std::move(cols));
  Json rs = Json::array();
  for (auto& r : rows) rs.push(r);
  o.set("rows", std::move(rs));
  return o;
}

// ---------- request handling ----------
static void check_point(const Point& p, const std::string& what, bool need_mag, bool wcs) {
  auto bad = [&](const char* m) { throw MeasureError("bad_request", what + ": " + m); };
  bool xy = p.x && p.y && !p.ra && !p.dec, rd = !p.x && !p.y && p.ra && p.dec;
  if (xy) {
    if (!(std::fabs(*p.x) < 1e9 && std::fabs(*p.y) < 1e9)) bad("x and y must be finite pixel positions");
  } else if (rd) {
    if (!wcs) bad("ra/dec positions need the request's wcs");
    if (!(std::isfinite(*p.ra) && *p.dec >= -90.0 && *p.dec <= 90.0)) bad("ra must be finite and dec within [-90, 90]");
  } else {
    bad("give either x and y or ra and dec");
  }
  if (p.mag) {
    if (!std::isfinite(*p.mag) || std::fabs(*p.mag) > 100.0) bad("mag out of range");
  } else if (need_mag) {
    bad("reference stars need mag");
  }
}
static std::optional<Vec2> pixel_of(const Point& p, const Wcs* wcs) {
  if (p.x && p.y) return Vec2{*p.x, *p.y};
  if (p.ra && p.dec && wcs) return coordinates::sky_to_pixel(*wcs, *p.ra, *p.dec);
  return std::nullopt;
}

Response run(GrayImage& img, const SourceInfo& info, const Request& req, Session& session,
             const std::function<std::vector<Star>(const GrayImage&)>& stars) {
  if (req.targets.size() > MAX_TARGETS) throw MeasureError("bad_request", "at most 200000 targets per request");
  if (req.refs.size() > MAX_REFS) throw MeasureError("bad_request", "at most 500000 reference stars per request");
  bool has_wcs = req.wcs.has_value();
  for (size_t i = 0; i < req.targets.size(); ++i) check_point(req.targets[i], "targets[" + std::to_string(i) + "]", false, has_wcs);
  for (size_t i = 0; i < req.refs.size(); ++i) check_point(req.refs[i], "refs[" + std::to_string(i) + "]", true, has_wcs);
  if (req.wcs) {
    const Wcs& w = *req.wcs;
    double det = w.cd[0][0] * w.cd[1][1] - w.cd[0][1] * w.cd[1][0];
    bool finite = std::isfinite(w.ra) && std::isfinite(w.dec) && std::isfinite(w.crpix[0]) && std::isfinite(w.crpix[1]) && std::isfinite(det);
    if (!finite || det == 0.0 || (w.sip && w.sip->order > 9)) throw MeasureError("bad_request", "wcs: not a usable WCS");
    if (w.sip) {
      size_t terms = Sip::terms(w.sip->order).size();
      if (w.sip->a.size() != terms || w.sip->b.size() != terms) throw MeasureError("bad_request", "wcs: SIP coefficients do not match the order");
    }
  }
  if (req.frame.content) {
    const Content& c = *req.frame.content;
    bool ok = true;
    for (auto& b : c.include)
      for (double v : b) ok = ok && std::isfinite(v);
    for (auto& b : c.exclude)
      for (double v : b) ok = ok && std::isfinite(v);
    if (!ok || c.include.size() + c.exclude.size() > 1024) throw MeasureError("bad_request", "frame.content: at most 1024 finite boxes");
    if (session.content) {
      if (!(*session.content == c))
        throw MeasureError("content_conflict", "A different content mask was applied to this frame; open it again.");
    } else {
      apply_content(img, c);
      session.content = c;
      session.model.reset();
    }
  }
  const Wcs* wcs = req.wcs ? &*req.wcs : nullptr;
  Json keyj = Json::array();
  keyj.push(req.frame.to_json());
  Json refsj = Json::array();
  for (auto& r : req.refs) refsj.push(r.to_json());
  keyj.push(std::move(refsj)).push(wcs ? wcs->to_json() : Json());
  std::string key = keyj.dump(false);
  if (!session.model || session.model_key != key) {
    session.model.reset();
    std::vector<RefStar> refs;
    for (auto& p : req.refs)
      if (auto px = pixel_of(p, wcs)) refs.push_back({(*px)[0], (*px)[1], p.mag.value_or(NANd)});
    std::optional<double> scale;
    if (wcs) scale = coordinates::scale_arcsec(*wcs, img.width, img.height);
    auto found = stars(img);
    session.model = std::make_unique<FrameModel>(FrameModel::create(img, info, found, refs, scale, req.frame));
    session.model_key = key;
  }
  const FrameModel& model = *session.model;
  std::vector<std::optional<Target>> targets;
  std::vector<Target> valid;
  for (auto& p : req.targets) {
    auto px = pixel_of(p, wcs);
    if (px) {
      targets.push_back(Target{(*px)[0], (*px)[1], p.mag});
      valid.push_back(*targets.back());
    } else {
      targets.push_back(std::nullopt);
    }
  }
  auto measured = measure(img, model, valid);
  size_t mi = 0;
  Response resp;
  for (auto& t : targets) {
    if (t) {
      const Measurement& m = measured[mi++];
      switch (m.status) {
        case Status::Recorded: resp.counts.recorded++; break;
        case Status::Trace: resp.counts.trace++; break;
        case Status::Below: resp.counts.below++; break;
        default: resp.counts.not_assessable++; break;
      }
      resp.rows.push_back(m.to_json());
    } else {
      resp.counts.not_assessable++;
      uint32_t fl = flags::NEAR_EDGE;
      if (!model.photometric()) fl |= flags::NONLINEAR;
      Json r = Json::array();
      for (int i = 0; i < 10; ++i) r.push(Json());
      r.push(uint64_t(0)).push(Json()).push(status_name(Status::NotAssessable)).push(fl);
      resp.rows.push_back(std::move(r));
    }
  }
  resp.model = model.summary();
  return resp;
}

}  // namespace skyshot::measure
