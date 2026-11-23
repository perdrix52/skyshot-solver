// Port of tests/measure.rs: synthetic tests for forced photometry.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>

#include "skyshot/measure.hpp"
#include "skyshot/pipeline.hpp"

using namespace skyshot;
using namespace skyshot::measure;

extern int failures, checks;
#define CHECK(cond)                                                        \
  do {                                                                     \
    ++checks;                                                              \
    if (!(cond)) {                                                         \
      ++failures;                                                          \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                      \
  } while (0)

namespace {
constexpr double FWHM = 3.0;
constexpr double TAU = 6.283185307179586;

struct Synth {
  size_t w, h;
  std::vector<double> px;
  std::mt19937_64 rng;
  Synth(size_t w_, size_t h_, uint64_t seed) : w(w_), h(h_), px(w_ * h_, 0.0), rng(seed) {}
  double unit() { return std::uniform_real_distribution<double>(0.0, 1.0)(rng); }
  double range(double a, double b) { return a + (b - a) * unit(); }
  double gauss() {
    double u1 = std::max(unit(), 1e-12), u2 = unit();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(TAU * u2);
  }
  void background(double c, double gx, double gy) {
    for (size_t y = 0; y < h; ++y)
      for (size_t x = 0; x < w; ++x) px[y * w + x] += c + gx * double(x) + gy * double(y);
  }
  void noise(double sigma, std::optional<double> gain) {
    for (auto& v : px) {
      double var = sigma * sigma + (gain ? std::max(v, 0.0) / *gain : 0.0);
      v += std::sqrt(var) * gauss();
    }
  }
  void smooth(int k) {
    int r = k / 2;
    auto src = px;
    for (int y = 0; y < int(h); ++y)
      for (int x = 0; x < int(w); ++x) {
        double s = 0, n = 0;
        for (int dy = -r; dy <= r; ++dy)
          for (int dx = -r; dx <= r; ++dx) {
            int xx = x + dx, yy = y + dy;
            if (xx >= 0 && yy >= 0 && xx < int(w) && yy < int(h)) {
              s += src[size_t(yy) * w + size_t(xx)];
              n += 1;
            }
          }
        px[size_t(y) * w + size_t(x)] = s / n;
      }
  }
  void star(double x, double y, double flux, double fwhm) {
    double s = fwhm / 2.3548;
    int r = int(std::ceil(6.0 * s));
    double norm = flux / (TAU * s * s) / 25.0;
    for (int yy = std::max(int(y) - r, 0); yy <= std::min(int(y) + r, int(h) - 1); ++yy)
      for (int xx = std::max(int(x) - r, 0); xx <= std::min(int(x) + r, int(w) - 1); ++xx) {
        double v = 0;
        for (int sy = 0; sy < 5; ++sy)
          for (int sx = 0; sx < 5; ++sx) {
            double dx = xx - 0.4 + 0.2 * sx - x, dy = yy - 0.4 + 0.2 * sy - y;
            v += std::exp(-(dx * dx + dy * dy) / (2.0 * s * s));
          }
        px[size_t(yy) * w + size_t(xx)] += v * norm;
      }
  }
  image_io::GrayImage image(std::optional<double> clip = std::nullopt) const {
    image_io::GrayImage g;
    g.width = uint32_t(w);
    g.height = uint32_t(h);
    for (double v : px) g.pixels.push_back(float(clip ? std::min(v, *clip) : v));
    return g;
  }
};

image_io::SourceInfo fits_info() {
  image_io::SourceInfo i;
  i.format = image_io::SourceFormat::Fits;
  i.bitpix = -32;
  return i;
}
std::vector<Star> detect(const image_io::GrayImage& img, const image_io::SourceInfo& info) {
  auto f = pipeline::attempt_field(img, info, "default");
  return f ? f->stars : std::vector<Star>{};
}
double flux_for_snr(double snr, double sigma) {
  double n = PI * FWHM * FWHM, n_ann = PI * (16.0 - 6.25) * FWHM * FWHM;
  return snr * sigma * std::sqrt(n + n * n / n_ann) / 0.937;
}
struct B {
  double x, y, f;
};
std::vector<B> bright_grid(Synth& s, double flux) {
  std::vector<B> out;
  int k = 0;
  for (int gy = 0; gy < 5; ++gy)
    for (int gx = 0; gx < 6; ++gx) {
      double jx = s.range(-12, 12), jy = s.range(-12, 12);
      double x = 40.0 + 100.0 * gx + jx, y = 45.0 + 100.0 * gy + jy;
      double f = flux * (1.0 + 0.1 * (k % 7));
      s.star(x, y, f, FWHM);
      out.push_back({x, y, f});
      ++k;
    }
  return out;
}
Target target(double x, double y) { return Target{x, y, std::nullopt}; }
FrameModel model(const image_io::GrayImage& img, const image_io::SourceInfo& info, const std::vector<RefStar>& refs, const FrameOptions& opts) {
  return FrameModel::create(img, info, detect(img, info), refs, std::nullopt, opts);
}
bool in(double v, double a, double b) { return v >= a && v < b; }

void known_sources() {
  Synth s(620, 520, 1);
  s.background(1000.0, 0.3, -0.2);
  auto bright = bright_grid(s, 60000.0);
  std::vector<std::pair<double, double>> rec = {{90.3, 95.6}, {290.7, 195.2}, {490.1, 295.8}, {190.5, 395.4}};
  std::vector<std::pair<double, double>> empty = {{390, 395}, {490, 95}, {290, 495}, {590, 195}};
  std::vector<std::pair<double, double>> tr;
  for (int k = 0; k < 30 && tr.size() < 12; ++k) {
    std::pair<double, double> p{90.4 + 100.0 * (k % 6), 95.2 + 100.0 * (k / 6)};
    bool ok = true;
    for (auto& q : rec) ok = ok && std::hypot(p.first - q.first, p.second - q.second) > 5.0;
    for (auto& q : empty) ok = ok && std::hypot(p.first - q.first, p.second - q.second) > 5.0;
    if (ok) tr.push_back(p);
  }
  double f_rec = flux_for_snr(25.0, 10.0);
  for (auto& p : rec) s.star(p.first + 0.4, p.second - 0.3, f_rec, FWHM);
  double f_tr = flux_for_snr(4.5, 10.0);
  for (auto& p : tr) s.star(p.first, p.second, f_tr, FWHM);
  s.noise(10.0, std::nullopt);
  auto img = s.image();
  auto info = fits_info();
  auto m = model(img, info, {}, FrameOptions{});
  auto sum = m.summary();
  std::printf("  fwhm %.3f sigma_sky %.2f chi.all %.3f\n", sum.fwhm, sum.sigma_sky, sum.chi.all.value_or(-1));
  CHECK(std::fabs(sum.fwhm - 3.08) < 0.15);
  CHECK(std::fabs(sum.sigma_sky - 10.0) < 1.0);
  CHECK(sum.chi.all && in(*sum.chi.all, 0.85, 1.2));
  CHECK(sum.linear && !sum.gain);
  std::vector<Target> t;
  for (auto& p : rec) t.push_back(target(p.first, p.second));
  for (auto& p : tr) t.push_back(target(p.first, p.second));
  for (auto& p : empty) t.push_back(target(p.first, p.second));
  auto out = measure::measure(img, m, t);
  for (size_t i = 0; i < 4; ++i) {
    auto& r = out[i];
    CHECK(r.status == Status::Recorded);
    if (r.flux) CHECK(in(*r.flux / f_rec, 0.85, 1.0));
    CHECK(r.dx && r.dy && std::fabs(*r.dx - 0.4) < 0.12 && std::fabs(*r.dy + 0.3) < 0.12);
    CHECK(r.snr_dog && *r.snr_dog > 3.0);
    CHECK((r.flags & flags::NOT_ASSESSABLE) == 0);
  }
  size_t traces = 0;
  std::vector<double> snr;
  size_t n_rec = 0;
  for (size_t i = 4; i < 4 + tr.size(); ++i) {
    traces += out[i].status == Status::Trace;
    n_rec += out[i].status == Status::Recorded;
    if (out[i].snr_cal) snr.push_back(*out[i].snr_cal);
  }
  std::printf("  traces %zu of %zu\n", traces, tr.size());
  CHECK(traces >= 8);
  // the Rust test (ChaCha8 noise) sees none; at SNR 4.5 a source reaching 6 is a 1.5 sigma draw, so allow a couple
  CHECK(n_rec <= 2);
  std::sort(snr.begin(), snr.end());
  CHECK(!snr.empty() && in(snr[snr.size() / 2], 3.5, 5.5));
  for (size_t i = 4 + tr.size(); i < out.size(); ++i) {
    CHECK(out[i].status == Status::Below);
    CHECK(out[i].snr_cal && std::fabs(*out[i].snr_cal) < 3.0);
  }
  std::vector<Target> tb;
  for (auto& b : bright) tb.push_back(target(b.x, b.y));
  for (auto& r : measure::measure(img, m, tb)) CHECK(r.dx && r.dy && std::hypot(*r.dx, *r.dy) < 0.05);
}

void correlated_noise() {
  Synth s(640, 560, 2);
  s.background(500.0, 0.0, 0.0);
  bright_grid(s, 400000.0);
  s.noise(30.0, std::nullopt);
  s.smooth(3);
  auto img = s.image();
  auto m = model(img, fits_info(), {}, FrameOptions{});
  auto chi = m.summary().chi.sky;
  CHECK(chi && *chi > 2.0);
  std::mt19937_64 rng(99);
  auto u = [&](double a, double b) { return a + (b - a) * std::uniform_real_distribution<double>(0, 1)(rng); };
  std::vector<Target> empty;
  for (int i = 0; i < 3000; ++i) {
    Target t = target(u(20, 620), u(20, 540));
    double dx = rem_euclid(t.x - 40.0, 100.0), dy = rem_euclid(t.y - 45.0, 100.0);
    if (std::hypot(std::min(dx, 100.0 - dx), std::min(dy, 100.0 - dy)) > 35.0) empty.push_back(t);
  }
  auto out = measure::measure(img, m, empty);
  std::vector<const Measurement*> ok;
  for (auto& r : out)
    if (r.status != Status::NotAssessable) ok.push_back(&r);
  CHECK(ok.size() > 1000);
  size_t raw = 0, cal_hits = 0;
  std::vector<double> cal;
  for (auto* r : ok) {
    raw += *r->snr >= 6.0;
    cal_hits += r->status == Status::Recorded;
    cal.push_back(*r->snr_cal);
  }
  std::sort(cal.begin(), cal.end());
  double med = cal[cal.size() / 2];
  std::vector<double> dev;
  for (double v : cal) dev.push_back(std::fabs(v - med));
  std::sort(dev.begin(), dev.end());
  double sd = 1.4826 * dev[dev.size() / 2];
  std::printf("  chi.sky %.2f spread %.3f raw %zu cal %zu of %zu\n", chi.value_or(-1), sd, raw, cal_hits, ok.size());
  CHECK(in(sd, 0.85, 1.2));
  CHECK(double(raw) > 0.005 * double(ok.size()));
  CHECK(double(cal_hits) <= 0.002 * double(ok.size()));
}

void saturation_edges_masks() {
  Synth s(600, 500, 3);
  s.background(800.0, 0.1, 0.1);
  auto bright = bright_grid(s, 40000.0);
  s.star(300.0, 250.0, 3000000.0, FWHM);
  s.noise(8.0, std::nullopt);
  auto img = s.image(20000.0);
  for (size_t y = 330; y < 380; ++y)
    for (size_t x = 120; x < 170; ++x) img.pixels[y * 600 + x] = std::nanf("");
  auto info = fits_info();
  auto m = model(img, info, {}, FrameOptions{});
  std::vector<Target> t = {target(300, 250), target(1, 250), target(-40, -40), target(145, 355), target(145, 384.5), target(bright[7].x, bright[7].y)};
  auto out = measure::measure(img, m, t);
  CHECK(out[0].status == Status::NotAssessable);
  CHECK(out[0].flags & flags::SATURATED);
  for (size_t i = 1; i < 3; ++i) {
    CHECK(out[i].status == Status::NotAssessable);
    CHECK(out[i].flags & flags::NEAR_EDGE);
    CHECK(!out[i].flux && !out[i].snr);
  }
  CHECK(out[3].status == Status::NotAssessable);
  CHECK(out[3].flags & flags::NO_DATA);
  CHECK(out[4].status != Status::Recorded);
  CHECK(out[5].status == Status::Recorded);
}

void blends_zero_point() {
  Synth s(620, 520, 4);
  s.background(1200.0, -0.2, 0.25);
  auto bright = bright_grid(s, 50000.0);
  double bx = bright[8].x + 3.0, by = bright[8].y;
  s.star(bx, by, flux_for_snr(5.0, 9.0), FWHM);
  s.noise(9.0, std::nullopt);
  auto img = s.image();
  auto info = fits_info();
  std::vector<RefStar> refs;
  for (auto& b : bright) refs.push_back({b.x, b.y, 25.0 - 2.5 * std::log10(b.f)});
  refs.push_back({500.0, 500.0, 21.0});
  auto m = model(img, info, refs, FrameOptions{});
  auto sum = m.summary();
  CHECK(sum.zero_point.has_value());
  if (sum.zero_point) {
    auto& z = *sum.zero_point;
    std::printf("  zp %.3f scatter %.4f linear %d m_lim %.2f\n", z.zp, z.scatter, int(z.linear), sum.m_lim.value_or(-1));
    CHECK(std::fabs(z.zp - 24.93) < 0.05);
    CHECK(z.scatter < 0.05 && z.linear && sum.photometric);
  }
  CHECK(sum.m_lim && std::fabs(*sum.m_lim - 18.9) < 0.4);
  auto out = measure::measure(img, m, {Target{bx, by, 19.0}, Target{90.0, 95.0, 19.0}});
  CHECK(out[0].status == Status::NotAssessable);
  CHECK(out[0].flags & flags::BLENDED);
  CHECK((out[1].flags & flags::BLENDED) == 0);
  auto m2 = model(img, info, {}, FrameOptions{});
  auto out2 = measure::measure(img, m2, {target(bx, by)});
  CHECK(out2[0].flags & flags::BLENDED);
}

void photon_transfer() {
  Synth s(640, 640, 5);
  s.background(500.0, 4.0, 3.0);
  bright_grid(s, 80000.0);
  s.noise(4.0, 2.0);
  auto img = s.image();
  auto m = model(img, fits_info(), {}, FrameOptions{});
  auto g = m.summary().gain;
  std::printf("  gain %.3f\n", g.value_or(-1));
  CHECK(g && std::fabs(*g - 2.0) < 0.3);
  image_io::SourceInfo png;
  png.format = image_io::SourceFormat::Png;
  png.bitpix = 16;
  auto m2 = model(img, png, {}, FrameOptions{});
  CHECK(!m2.summary().gain && !m2.photometric());
  auto r = measure::measure(img, m2, {target(40.0, 45.0)})[0];
  CHECK(r.flags & flags::NONLINEAR);
  FrameOptions lin;
  lin.linear = true;
  CHECK(model(img, png, {}, lin).summary().gain.has_value());
}

std::pair<image_io::GrayImage, image_io::SourceInfo> request_frame() {
  Synth s(400, 360, 6);
  s.background(300.0, 0.0, 0.0);
  for (int k = 0; k < 12; ++k) {
    double jx = s.range(-10, 10), jy = s.range(-10, 10);
    s.star(40.0 + 80.0 * (k % 5) + jx, 40.0 + 90.0 * (k / 5) + jy, 30000.0, FWHM);
  }
  s.star(200.0, 300.0, flux_for_snr(20.0, 5.0), FWHM);
  s.noise(5.0, std::nullopt);
  return {s.image(), fits_info()};
}
Point px(double x, double y) {
  Point p;
  p.x = x;
  p.y = y;
  return p;
}

void request_path() {
  auto [img, info] = request_frame();
  Request req;
  req.targets = {px(200, 300), px(100, 300), px(-5, 3)};
  auto det = [&](const image_io::GrayImage& g) { return detect(g, info); };
  Session session;
  auto r1 = run(img, info, req, session, det);
  auto r2 = run(img, info, req, session, [](const image_io::GrayImage&) -> std::vector<Star> { throw std::runtime_error("the model was rebuilt"); });
  CHECK(r1.to_json().dump(false) == r2.to_json().dump(false));
  Session s2;
  auto r3 = run(img, info, req, s2, det);
  CHECK(r1.to_json().dump(false) == r3.to_json().dump(false));
  CHECK(r1.rows.size() == 3);
  auto st = [&](size_t i) { return r1.rows[i].items()[12].as_str(); };
  CHECK(st(0) == "recorded");
  CHECK(st(1) == "below");
  CHECK(st(2) == "not_assessable");
  CHECK(r1.counts.recorded + r1.counts.below + r1.counts.not_assessable == 3);
  Request given = req;
  given.frame.fwhm = 3.2;
  auto r4 = run(img, info, given, session, det);
  CHECK(r4.model.fwhm == 3.2);
}

template <class F>
std::string code_of(F&& f) {
  try {
    f();
  } catch (const MeasureError& e) {
    return e.code;
  }
  return "";
}

void request_errors() {
  auto [img0, info] = request_frame();
  auto det = [&](const image_io::GrayImage& g) { return detect(g, info); };
  auto code = [&](const Request& req) {
    auto im = img0;
    Session s;
    return code_of([&] { run(im, info, req, s, det); });
  };
  Point sky;
  sky.ra = 10.0;
  sky.dec = 41.0;
  Request a1;
  a1.targets = {sky};
  CHECK(code(a1) == "bad_request");
  Request a2;
  a2.refs = {px(10, 10)};
  CHECK(code(a2) == "bad_request");
  Request a3;
  Point half;
  half.x = 1.0;
  a3.targets = {half};
  CHECK(code(a3) == "bad_request");
  Request a4;
  a4.frame.fwhm = 500.0;
  CHECK(code(a4) == "bad_request");
  Session session;
  Request a;
  a.targets = {px(200, 300)};
  Content c;
  c.include = {{0, 0, 400, 360}};
  c.exclude = {{0, 0, 50, 50}};
  a.frame.content = c;
  auto im = img0;
  run(im, info, a, session, det);
  CHECK(std::isnan(im.pixels[10 * 400 + 10]) && std::isfinite(im.pixels[100 * 400 + 100]));
  Request b = a;
  b.frame.content->exclude.clear();
  CHECK(code_of([&] { run(im, info, b, session, det); }) == "content_conflict");
  Synth flat(200, 200, 7);
  flat.background(100.0, 0.0, 0.0);
  flat.noise(3.0, std::nullopt);
  auto fi = flat.image();
  Request req;
  req.targets = {px(100, 100)};
  auto none = [](const image_io::GrayImage&) { return std::vector<Star>{}; };
  Session s3;
  CHECK(code_of([&] { run(fi, info, req, s3, none); }) == "no_psf");
  Request given = req;
  given.frame.fwhm = 3.0;
  Session s4;
  auto out = run(fi, info, given, s4, none);
  CHECK(out.rows[0].items()[12].as_str() == "below");
  for (const char* bad : {"{", "{\"targets\": 5}", "{\"targets\": [{\"x\": 1, \"y\": 2, \"z\": 3}]}", "{\"frame\": {\"fwhm\": \"wide\"}}",
                          "[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"}) {
    bool threw = false;
    try {
      Request::from_json(parse_json(bad));
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }
  image_io::GrayImage tiny;
  tiny.width = 8;
  tiny.height = 8;
  tiny.pixels.assign(64, 1.0f);
  Session s5;
  CHECK(code_of([&] { run(tiny, info, given, s5, none); }) == "no_frame");
}

void zero_point_helper() {
  std::vector<double> cat, inst;
  for (int i = 0; i < 60; ++i) {
    cat.push_back(9.0 + 0.1 * i);
    inst.push_back(0.7 * cat.back() - 17.0 + (i % 13 == 0 ? 1.5 : 0.0));
  }
  auto z = fit_zero_point(cat, inst);
  CHECK(z.has_value());
  if (z) {
    CHECK(z->slope && std::fabs(*z->slope - 0.7) < 1e-6);
    CHECK(!z->linear);
    CHECK(z->n < 60 && z->n > 4);
  }
}
}  // namespace

void run_measure_tests() {
  struct T {
    const char* name;
    void (*fn)();
  } tests[] = {{"measure: known sources", known_sources},
               {"measure: correlated noise", correlated_noise},
               {"measure: saturation/edges/masks", saturation_edges_masks},
               {"measure: blends and zero point", blends_zero_point},
               {"measure: photon transfer gain", photon_transfer},
               {"measure: request path", request_path},
               {"measure: request errors", request_errors},
               {"measure: zero point helper", zero_point_helper}};
  for (auto& t : tests) {
    int before = failures;
    std::printf("[ RUN  ] %s\n", t.name);
    try {
      t.fn();
    } catch (const std::exception& e) {
      ++failures;
      std::fprintf(stderr, "EXCEPTION in %s: %s\n", t.name, e.what());
    }
    std::printf("[ %s ] %s\n", failures == before ? " OK " : "FAIL", t.name);
  }
}
