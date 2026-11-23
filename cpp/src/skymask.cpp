#include "skyshot/skymask.hpp"

#include <algorithm>

#include "skyshot/linalg.hpp"
#include "skyshot/parallel.hpp"

namespace skyshot {

// ---------- SkyMask ----------
SkyMask SkyMask::open(uint32_t width, uint32_t height, uint32_t cols, uint32_t rows) {
  cols = std::max<uint32_t>(cols, 1);
  rows = std::max<uint32_t>(rows, 1);
  SkyMask m;
  m.cols = cols;
  m.rows = rows;
  m.cell = {double(width) / double(cols), double(height) / double(rows)};
  m.origin = {0.0, 0.0};
  m.blocked.assign(size_t(cols) * rows, false);
  return m;
}

std::optional<std::string> SkyMask::validate() const {
  uint64_t cells = uint64_t(cols) * rows;
  if (cols == 0 || rows == 0 || cells > SKYMASK_MAX_CELLS)
    return "mask grid must have 1.." + std::to_string(SKYMASK_MAX_CELLS) + " cells";
  size_t n = size_t(cols) * rows;
  if (blocked.size() < n || blocked.size() >= n + 8) return std::string("mask size does not match cols x rows");
  for (double c : cell)
    if (!(std::isfinite(c) && c > 0.0)) return std::string("mask cell size and origin must be finite and positive");
  for (double o : origin)
    if (!std::isfinite(o)) return std::string("mask cell size and origin must be finite and positive");
  return std::nullopt;
}

std::optional<size_t> SkyMask::index(double x, double y) const {
  double i = std::floor((x - origin[0]) / cell[0]);
  double j = std::floor((y - origin[1]) / cell[1]);
  if (i < 0.0 || j < 0.0 || i >= double(cols) || j >= double(rows)) return std::nullopt;
  return sat_cast<size_t>(j) * cols + sat_cast<size_t>(i);
}

bool SkyMask::is_blocked(double x, double y) const {
  auto k = index(x, y);
  return k && blocked[*k];
}

size_t SkyMask::n_blocked() const {
  size_t n = 0;
  for (bool b : blocked) n += size_t(b);
  return n;
}

bool SkyMask::is_landscape() const {
  size_t n = size_t(cols) * rows;
  if (n == 0) return false;
  size_t c = 0;
  for (size_t i = 0; i < std::min(n, blocked.size()); ++i) c += size_t(blocked[i]);
  return double(c) >= LANDSCAPE_FRACTION * double(n);
}

double SkyMask::blocked_area(double x0, double y0, double x1, double y1) const {
  double a = 0.0;
  for (size_t j = 0; j < rows; ++j) {
    double cy0 = origin[1] + double(j) * cell[1];
    double oy = fmin_r(cy0 + cell[1], y1) - fmax_r(cy0, y0);
    if (oy <= 0.0) continue;
    for (size_t i = 0; i < cols; ++i) {
      if (!blocked[j * cols + i]) continue;
      double cx0 = origin[0] + double(i) * cell[0];
      double ox = fmin_r(cx0 + cell[0], x1) - fmax_r(cx0, x0);
      if (ox > 0.0) a += ox * oy;
    }
  }
  return a;
}

std::array<double, 4> SkyMask::open_bbox(double width, double height) const {
  double x0 = std::numeric_limits<double>::max(), y0 = x0, x1 = std::numeric_limits<double>::lowest(), y1 = x1;
  for (size_t j = 0; j < rows; ++j)
    for (size_t i = 0; i < cols; ++i) {
      if (blocked[j * cols + i]) continue;
      double cx = origin[0] + double(i) * cell[0];
      double cy = origin[1] + double(j) * cell[1];
      x0 = fmin_r(x0, cx);
      y0 = fmin_r(y0, cy);
      x1 = fmax_r(x1, cx + cell[0]);
      y1 = fmax_r(y1, cy + cell[1]);
    }
  if (x0 > x1) return {0.0, 0.0, width, height};
  return {fmax_r(x0, 0.0), fmax_r(y0, 0.0), fmin_r(x1, width), fmin_r(y1, height)};
}

double SkyMask::open_fraction(double x0, double y0, double x1, double y1) const {
  double area = (x1 - x0) * (y1 - y0);
  if (area <= 0.0) return 0.0;
  return clampd(1.0 - blocked_area(x0, y0, x1, y1) / area, 0.0, 1.0);
}

SkyMask SkyMask::shifted(double x0, double y0) const {
  SkyMask m = *this;
  m.origin = {origin[0] - x0, origin[1] - y0};
  return m;
}

SkyMask SkyMask::trimmed() const {
  SkyMask m = *this;
  m.blocked.resize(std::min(m.blocked.size(), size_t(cols) * rows));
  return m;
}

Json SkyMask::to_json() const {
  Json o = Json::object();
  o.set("cols", cols).set("rows", rows).set("cell", Json::from_arr(cell)).set("origin", Json::from_arr(origin));
  static const char* HEX = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < blocked.size(); i += 8) {
    unsigned byte = 0;
    for (size_t k = 0; k < 8 && i + k < blocked.size(); ++k) byte |= unsigned(blocked[i + k]) << k;
    s.push_back(HEX[byte >> 4]);
    s.push_back(HEX[byte & 15]);
  }
  o.set("blocked", s);
  return o;
}

SkyMask SkyMask::from_json(const Json& j) {
  SkyMask m;
  auto get_u32 = [&](const char* k) {
    const Json* v = j.get(k);
    auto u = v ? v->as_u64() : std::nullopt;
    if (!u || *u > UINT32_MAX) throw Error(std::string("invalid mask field ") + k);
    return uint32_t(*u);
  };
  auto get2 = [&](const char* k, bool required) -> Vec2 {
    const Json* v = j.get(k);
    if (!v) {
      if (required) throw Error(std::string("missing mask field ") + k);
      return {0, 0};
    }
    if (!v->is_array() || v->items().size() != 2 || !v->items()[0].is_number() || !v->items()[1].is_number())
      throw Error(std::string("invalid mask field ") + k);
    return {v->items()[0].as_f64(), v->items()[1].as_f64()};
  };
  m.cols = get_u32("cols");
  m.rows = get_u32("rows");
  m.cell = get2("cell", true);
  m.origin = get2("origin", false);
  const Json* b = j.get("blocked");
  if (!b || !b->is_string()) throw Error("missing mask field blocked");
  const std::string& s = b->as_str();
  if (s.size() % 2 != 0 || s.size() > 2 * ((SKYMASK_MAX_CELLS + 7) / 8)) throw Error("bad mask length");
  auto hv = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw Error("bad mask digit");
  };
  for (size_t i = 0; i < s.size(); i += 2) {
    int byte = (hv(s[i]) << 4) | hv(s[i + 1]);
    for (int k = 0; k < 8; ++k) m.blocked.push_back(((byte >> k) & 1) == 1);
  }
  return m;
}

// ---------- estimate ----------
namespace skymask {
namespace {

constexpr double GRID = 64.0;
constexpr double MIN_BLOCKED = 0.02;
constexpr double MAX_BLOCKED = 0.85;
constexpr double MIN_GAIN = 8.0;
constexpr float NANF = std::numeric_limits<float>::quiet_NaN();

struct Cells {
  size_t cols, rows;
  std::vector<float> med, tex, stars;
  bool dead(size_t k, float sigma, float black) const {
    return std::fabs(med[k]) <= sigma && tex[k] < 0.25f * sigma && (med[k] - black) <= sigma && stars[k] == 0.0f;
  }
};

float median_f(std::vector<float>& v) {
  if (v.empty()) return NANF;
  size_t k = v.size() / 2;
  std::nth_element(v.begin(), v.begin() + long(k), v.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
  return v[k];
}

float percentile(const std::vector<float>& v, double q) {
  std::vector<float> w;
  for (float x : v)
    if (std::isfinite(x)) w.push_back(x);
  if (w.empty()) return NANF;
  size_t k = sat_cast<size_t>(std::round(double(w.size() - 1) * q));
  std::nth_element(w.begin(), w.begin() + long(k), w.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
  return w[k];
}

Cells measure(const image_io::GrayImage& gray, const StarField& field, double cell, size_t cols, size_t rows) {
  size_t w = gray.width, h = gray.height;
  size_t step = std::max<size_t>(sat_cast<size_t>(std::floor(cell / 14.0)), 1);
  size_t n = cols * rows;
  Cells c;
  c.cols = cols;
  c.rows = rows;
  c.med.assign(n, NANF);
  c.tex.assign(n, NANF);
  parallel_for(rows, [&](size_t j) {
    size_t y0 = sat_cast<size_t>(double(j) * cell);
    size_t y1 = std::min(sat_cast<size_t>(double(j + 1) * cell), h);
    std::vector<float> vals, diffs;
    vals.reserve(256);
    diffs.reserve(256);
    for (size_t i = 0; i < cols; ++i) {
      size_t x0 = sat_cast<size_t>(double(i) * cell);
      size_t x1 = std::min(sat_cast<size_t>(double(i + 1) * cell), w);
      vals.clear();
      diffs.clear();
      for (size_t y = y0; y < y1; y += step) {
        const float* row = gray.pixels.data() + y * w;
        for (size_t x = x0; x < x1; x += step) {
          float p = row[x];
          if (!std::isfinite(p)) continue;
          vals.push_back(p);
          if (x + 2 < x1 && std::isfinite(row[x + 2])) diffs.push_back(std::fabs(row[x + 2] - p));
        }
      }
      c.med[j * cols + i] = median_f(vals);
      c.tex[j * cols + i] = median_f(diffs);
    }
  });
  std::vector<float> mid;
  {
    size_t skip = std::min<size_t>(field.stars.size(), 30);
    size_t taken = 0;
    for (size_t i = skip; i < field.stars.size() && taken < 2000; ++i, ++taken) {
      const Star& s = field.stars[i];
      if (std::isfinite(s.fwhm) && s.fwhm > 0.0) mid.push_back(float(s.fwhm));
    }
  }
  float psf = percentile(mid, 0.25);
  c.stars.assign(n, 0.0f);
  for (const Star& s : field.stars) {
    bool round = std::isfinite(s.elongation) && s.elongation <= 1.5;
    bool sized = !std::isfinite(psf) || (float(s.fwhm) >= 0.5f * psf && float(s.fwhm) <= 1.8f * psf);
    if (!(round && sized && (s.flags & star_flags::REJECT) == 0)) continue;
    double i = std::floor(s.x / cell), j = std::floor(s.y / cell);
    if (i >= 0.0 && j >= 0.0 && sat_cast<size_t>(i) < cols && sat_cast<size_t>(j) < rows)
      c.stars[sat_cast<size_t>(j) * cols + sat_cast<size_t>(i)] += 1.0f;
  }
  return c;
}

std::optional<std::array<double, 6>> fit_surface(const Cells& c, const std::vector<bool>& use) {
  auto basis = [&](size_t i, size_t j) {
    double x = 2.0 * (double(i) + 0.5) / double(c.cols) - 1.0;
    double y = 2.0 * (double(j) + 0.5) / double(c.rows) - 1.0;
    return std::array<double, 6>{1.0, x, y, x * x, x * y, y * y};
  };
  std::vector<bool> keep = use;
  std::optional<std::array<double, 6>> coef;
  for (int pass = 0; pass < 3; ++pass) {
    double ata[6][6] = {};
    double atb[6] = {};
    int n = 0;
    for (size_t j = 0; j < c.rows; ++j)
      for (size_t i = 0; i < c.cols; ++i) {
        size_t k = j * c.cols + i;
        if (!keep[k] || !std::isfinite(c.med[k])) continue;
        auto b = basis(i, j);
        for (int r = 0; r < 6; ++r) {
          atb[r] += b[size_t(r)] * double(c.med[k]);
          for (int s = 0; s < 6; ++s) ata[r][s] += b[size_t(r)] * b[size_t(s)];
        }
        ++n;
      }
    if (n < 12) return coef;
    linalg::Mat a(6, 6);
    for (size_t r = 0; r < 6; ++r)
      for (size_t s = 0; s < 6; ++s) a(r, s) = ata[r][s];
    auto l = linalg::cholesky(a);
    if (!l) return std::nullopt;
    auto x = linalg::cholesky_solve(*l, std::vector<double>(atb, atb + 6));
    std::array<double, 6> cf = {x[0], x[1], x[2], x[3], x[4], x[5]};
    coef = cf;
    auto model_at = [&](size_t k) {
      auto b = basis(k % c.cols, k / c.cols);
      double s = -0.0;
      for (size_t r = 0; r < 6; ++r) s += cf[r] * b[r];
      return s;
    };
    std::vector<float> res;
    for (size_t k = 0; k < c.rows * c.cols; ++k)
      if (keep[k] && std::isfinite(c.med[k])) res.push_back(float(std::fabs(double(c.med[k]) - model_at(k))));
    float sig = 1.4826f * fmax_r(percentile(res, 0.5), 1e-12f);
    for (size_t k = 0; k < keep.size(); ++k) {
      if (keep[k] && std::isfinite(c.med[k])) {
        double r = std::fabs(double(c.med[k]) - model_at(k));
        if (r > 3.0 * double(sig)) keep[k] = false;
      }
    }
  }
  return coef;
}

std::pair<std::vector<bool>, double> skyline(const std::vector<double>& g, size_t cols, size_t rows) {
  const double JUMP = 0.35, JUMP_CAP = 2.0;
  auto col_cost = [&](size_t i) {
    std::vector<double> above(rows + 1, 0.0), below(rows + 1, 0.0);
    for (size_t j = 0; j < rows; ++j) above[j + 1] = above[j] + fmax_r(g[j * cols + i], 0.0);
    for (size_t j = rows; j-- > 0;) below[j] = below[j + 1] + fmax_r(-g[j * cols + i], 0.0);
    std::vector<double> out(rows + 1);
    for (size_t b = 0; b <= rows; ++b) out[b] = above[b] + below[b];
    return out;
  };
  std::vector<double> cost = col_cost(0);
  std::vector<std::vector<size_t>> back(cols, std::vector<size_t>(rows + 1, 0));
  for (size_t i = 1; i < cols; ++i) {
    auto cc = col_cost(i);
    std::vector<double> next(rows + 1, INF);
    for (size_t b = 0; b <= rows; ++b) {
      for (size_t p = 0; p < cost.size(); ++p) {
        double jump = fmin_r(JUMP * std::fabs(double(b) - double(p)), JUMP_CAP);
        double v = cost[p] + jump;
        if (v < next[b]) {
          next[b] = v;
          back[i][b] = p;
        }
      }
      next[b] += cc[b];
    }
    cost = next;
  }
  size_t b = rows;
  {
    bool first = true;
    for (size_t x = 0; x <= rows; ++x)
      if (first || total_cmp(cost[x], cost[b]) < 0) {
        b = x;
        first = false;
      }
  }
  double total = cost[b];
  std::vector<bool> blocked(cols * rows, false);
  for (size_t i = cols; i-- > 0;) {
    for (size_t j = b; j < rows; ++j) blocked[j * cols + i] = true;
    b = back[i][b];
  }
  return {blocked, total};
}

struct Rot {
  size_t c2, r2;
  std::vector<size_t> map;
};
Rot rotation(size_t cols, size_t rows, size_t o) {
  Rot r;
  if (o % 2 == 0) {
    r.c2 = cols;
    r.r2 = rows;
  } else {
    r.c2 = rows;
    r.r2 = cols;
  }
  r.map.resize(r.c2 * r.r2);
  for (size_t k = 0; k < r.c2 * r.r2; ++k) {
    size_t i2 = k % r.c2, j2 = k / r.c2, i, j;
    switch (o) {
      case 0: i = i2; j = j2; break;
      case 1: i = cols - 1 - j2; j = i2; break;
      case 2: i = cols - 1 - i2; j = rows - 1 - j2; break;
      default: i = j2; j = rows - 1 - i2; break;
    }
    r.map[k] = j * cols + i;
  }
  return r;
}

std::vector<bool> skylines(const std::vector<double>& g, size_t cols, size_t rows) {
  double none = 0.0;
  for (double v : g) none += fmax_r(v, 0.0);
  std::vector<std::pair<double, std::vector<bool>>> found;
  for (size_t o = 0; o < 4; ++o) {
    Rot rot = rotation(cols, rows, o);
    std::vector<double> gr(rot.map.size());
    for (size_t k = 0; k < rot.map.size(); ++k) gr[k] = g[rot.map[k]];
    auto [b, cost] = skyline(gr, rot.c2, rot.r2);
    size_t ground_edge = 0, sky_edge = 0, ground_hidden = 0, sky_hidden = 0;
    for (size_t k = (rot.r2 - 1) * rot.c2; k < b.size(); ++k) ground_edge += size_t(b[k]);
    for (size_t k = 0; k < rot.c2; ++k) sky_edge += size_t(b[k]);
    for (size_t k = (rot.r2 - 1) * rot.c2; k < gr.size(); ++k) ground_hidden += size_t(gr[k] > 0.0);
    for (size_t k = 0; k < rot.c2; ++k) sky_hidden += size_t(gr[k] > 0.0);
    bool coherent = ground_edge * 5 >= rot.c2 * 3 && sky_edge * 10 <= rot.c2 && ground_hidden * 2 >= rot.c2 && sky_hidden * 2 <= rot.c2;
    std::vector<bool> blocked(cols * rows, false);
    for (size_t k2 = 0; k2 < rot.map.size(); ++k2) blocked[rot.map[k2]] = b[k2];
    found.emplace_back(coherent ? none - cost : 0.0, std::move(blocked));
  }
  std::stable_sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return total_cmp(b.first, a.first) < 0; });
  if (found[0].first >= MIN_GAIN) return std::move(found[0].second);
  return std::vector<bool>(cols * rows, false);
}

void flood_dead(const Cells& c, float sigma, float black, std::vector<bool>& blocked) {
  size_t cols = c.cols, rows = c.rows;
  auto dead = [&](size_t k) { return c.dead(k, sigma, black); };
  std::vector<size_t> stack;
  for (size_t k = 0; k < cols * rows; ++k) {
    size_t i = k % cols, j = k / cols;
    if ((i == 0 || j == 0 || i + 1 == cols || j + 1 == rows) && dead(k)) stack.push_back(k);
  }
  std::vector<bool> seen(cols * rows, false);
  while (!stack.empty()) {
    size_t k = stack.back();
    stack.pop_back();
    if (seen[k]) continue;
    seen[k] = true;
    blocked[k] = true;
    size_t i = k % cols, j = k / cols;
    auto push = [&](size_t ii, size_t jj) {
      size_t kk = jj * cols + ii;
      if (!seen[kk] && dead(kk)) stack.push_back(kk);
    };
    if (i > 0) push(i - 1, j);
    if (i + 1 < cols) push(i + 1, j);
    if (j > 0) push(i, j - 1);
    if (j + 1 < rows) push(i, j + 1);
  }
}

}  // namespace

std::shared_ptr<const SkyMask> estimate(const image_io::GrayImage& gray, const StarField& field) {
  double w = double(gray.width), h = double(gray.height);
  if (w < 64.0 || h < 64.0 || gray.pixels.size() < size_t(gray.width) * gray.height) return nullptr;
  double cell = fmax_r(std::ceil(fmax_r(w, h) / GRID), 4.0);
  size_t cols = sat_cast<size_t>(std::ceil(w / cell)), rows = sat_cast<size_t>(std::ceil(h / cell));
  Cells c = measure(gray, field, cell, cols, rows);
  size_t n = cols * rows;
  std::vector<float> nz;
  for (float t : c.tex)
    if (t > 0.0f) nz.push_back(t);
  float sigma = percentile(nz, 0.2);
  if (!std::isfinite(sigma) || sigma <= 0.0f) return nullptr;
  std::vector<float> dens(n);
  for (size_t k = 0; k < n; ++k) {
    int64_t i = int64_t(k % cols), j = int64_t(k / cols);
    float s = 0.0f;
    for (int64_t dj = -1; dj <= 1; ++dj)
      for (int64_t di = -1; di <= 1; ++di) {
        int64_t ii = i + di, jj = j + dj;
        if (ii >= 0 && jj >= 0 && size_t(ii) < cols && size_t(jj) < rows) s += c.stars[size_t(jj) * cols + size_t(ii)];
      }
    dens[k] = s;
  }
  std::vector<float> starry;
  for (float d : dens)
    if (d > 0.0f) starry.push_back(d);
  float dens_sky = fmax_r(percentile(starry, 0.5), 1.0f);
  std::vector<bool> sky_like(n);
  for (size_t k = 0; k < n; ++k) sky_like[k] = c.tex[k] <= 2.0f * sigma && dens[k] >= 0.5f * dens_sky;
  auto surf = fit_surface(c, sky_like);
  auto model = [&](size_t k) -> double {
    if (!surf) return NaN;
    double x = 2.0 * (double(k % cols) + 0.5) / double(cols) - 1.0;
    double y = 2.0 * (double(k / cols) + 0.5) / double(rows) - 1.0;
    double b[6] = {1.0, x, y, x * x, x * y, y * y};
    double s = -0.0;
    for (size_t r = 0; r < 6; ++r) s += (*surf)[r] * b[r];
    return s;
  };
  std::vector<float> resid;
  for (size_t k = 0; k < n; ++k)
    if (sky_like[k]) resid.push_back(float(std::fabs(double(c.med[k]) - model(k))));
  float sig_level = fmax_r(1.4826f * percentile(resid, 0.5), sigma);
  float black = percentile(c.med, 0.01);
  std::vector<double> g(n);
  for (size_t k = 0; k < n; ++k) {
    float t = c.tex[k] / sigma;
    double s = clampd((double(t) - 2.0) / 2.0, 0.0, 2.0);
    if (surf && std::isfinite(c.med[k])) {
      double dark = (model(k) - double(c.med[k])) / double(sig_level);
      s += clampd((dark - 3.0) / 3.0, 0.0, 2.0);
    }
    if (c.dead(k, sigma, black)) s += 1.5;
    double stellar = double(clampf(dens[k] / dens_sky, 0.0f, 1.5f));
    s *= fmax_r(1.0 - stellar, 0.0);
    s -= stellar;
    g[k] = s - 0.15;
  }
  std::vector<bool> blocked = skylines(g, cols, rows);
  flood_dead(c, sigma, black, blocked);
  std::vector<bool> grown(n);
  for (size_t k = 0; k < n; ++k) {
    size_t i = k % cols, j = k / cols;
    bool near = (i > 0 && blocked[k - 1]) || (i + 1 < cols && blocked[k + 1]) || (j > 0 && blocked[k - cols]) ||
                (j + 1 < rows && blocked[k + cols]);
    grown[k] = blocked[k] || (near && g[k] > 0.0);
  }
  double nb = 0;
  for (bool b : grown) nb += b ? 1.0 : 0.0;
  if (nb < MIN_BLOCKED * double(n) || nb > MAX_BLOCKED * double(n)) return nullptr;
  auto m = std::make_shared<SkyMask>();
  m->cols = uint32_t(cols);
  m->rows = uint32_t(rows);
  m->cell = {cell, cell};
  m->origin = {0.0, 0.0};
  m->blocked = std::move(grown);
  return m;
}

}  // namespace skymask
}  // namespace skyshot
