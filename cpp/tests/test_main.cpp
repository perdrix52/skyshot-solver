// Integration and unit tests. Mirrors tests/cli.rs plus checks of the numerical building blocks.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>

#include "../examples/support.hpp"
#include "skyshot/compat.hpp"
#include "skyshot/engine.hpp"
#include "skyshot/extract.hpp"
#include "skyshot/fingerprint.hpp"
#include "skyshot/geometry.hpp"
#include "skyshot/image_io.hpp"
#include "skyshot/learned.hpp"
#include "skyshot/select.hpp"
#include "skyshot/sha256.hpp"
#include "skyshot/sphere.hpp"

#ifndef SKYSHOT_CLI
#define SKYSHOT_CLI "skyshot"
#endif

using namespace skyshot;
namespace fs = std::filesystem;

int failures = 0, checks = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++checks;                                                                \
    if (!(cond)) {                                                           \
      ++failures;                                                            \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
    }                                                                        \
  } while (0)

struct TempDir {
  fs::path path;
  TempDir() {
    std::random_device rd;
    path = fs::temp_directory_path() / ("skyshot-test-" + std::to_string(rd()) + std::to_string(rd()));
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

static int run_cli(const std::string& args, const fs::path& stdout_file) {
  std::string cmd = std::string("\"") + SKYSHOT_CLI + "\" " + args + " > \"" + stdout_file.string() + "\" 2> NUL";
#ifdef _WIN32
  cmd = "\"" + cmd + "\"";
#endif
  int rc = std::system(cmd.c_str());
  return rc;
}

static void test_json_numbers() {
  CHECK(format_f64(1.0) == "1.0");
  CHECK(format_f64(0.1) == "0.1");
  CHECK(format_f64(0.25) == "0.25");
  CHECK(format_f64(1e21) == "1e21");
  CHECK(format_f64(1.5e-7) == "1.5e-7");
  CHECK(format_f64(123456789.0) == "123456789.0");
  CHECK(format_f64(1e16) == "1e16");
  CHECK(format_f64(1e15) == "1000000000000000.0");
  CHECK(format_f64(0.0001) == "0.0001");
  CHECK(format_f64(0.00001) == "0.00001");
  CHECK(format_f64(0.000001) == "1e-6");
  CHECK(format_f64(-2.5) == "-2.5");
  Json j = Json::object();
  j.set("a", 1).set("b", Json::array().push(1.5).push(Json())).set("c", "x\"y");
  CHECK(j.dump(false) == "{\"a\":1,\"b\":[1.5,null],\"c\":\"x\\\"y\"}");
  CHECK(j.dump(true) == "{\n  \"a\": 1,\n  \"b\": [\n    1.5,\n    null\n  ],\n  \"c\": \"x\\\"y\"\n}");
  Json p = parse_json("{\"x\": [1, 2.5e3, -4], \"s\": \"\\u00e9\", \"t\": true}");
  CHECK(p.get("x")->items()[1].as_f64() == 2500.0);
  CHECK(p.get("x")->items()[2].as_i64().value() == -4);
  CHECK(p.get("s")->as_str() == "\xC3\xA9");
  bool threw = false;
  try {
    parse_json("{\"x\": ");
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
}

static void test_sha256() {
  CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
}

static void test_sphere_and_select() {
  for (int k = 0; k < 2000; ++k) {
    double ra = std::fmod(k * 37.13, 360.0);
    double dec = to_degrees(std::asin((k * 0.61803 - std::floor(k * 0.61803)) * 2.0 - 1.0));
    Vec3 v = sphere::unit(ra, dec);
    auto fc = sphere::face_coords(v);
    Vec3 w = sphere::face_vector(fc.face, fc.alpha, fc.beta);
    CHECK(sphere::angle(v, w) < 1e-12);
  }
  // cap cells cover brute force
  sphere::CubeGrid g(64);
  std::vector<uint32_t> out;
  for (int k = 0; k < 100; ++k) {
    double ra = std::fmod(k * 77.7, 360.0);
    double dec = to_degrees(std::asin((k * 0.414 - std::floor(k * 0.414)) * 2.0 - 1.0));
    Vec3 c = sphere::unit(ra, dec);
    static const double radii[4] = {0.002, 0.01, 0.05, 0.2};
    double radius = radii[k % 4];
    g.cap_cells(c, radius, out);
    auto [e, n] = sphere::tangent_basis(c);
    for (int t = 0; t < 100; ++t) {
      double ang = t * 2.399;
      double rr = radius * std::sqrt((t + 0.5) / 100.0);
      Vec3 p = sphere::ungnomonic({std::tan(rr) * std::cos(ang), std::tan(rr) * std::sin(ang)}, c, e, n);
      if (sphere::angle(p, c) <= radius) CHECK(std::find(out.begin(), out.end(), g.cell(p)) != out.end());
    }
  }
  // dominance matches brute force
  uint64_t s = 7;
  auto rnd = [&] {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return double(s >> 11) / double(uint64_t(1) << 53);
  };
  std::vector<Vec2> pts;
  for (int i = 0; i < 3000; ++i) pts.push_back({rnd() * 4000.0, rnd() * 3000.0});
  auto d = select::dominance_px(pts, 4000.0, 3000.0);
  for (size_t i = 0; i < pts.size(); i += 37) {
    double best = INF;
    for (size_t j = 0; j < i; ++j) best = std::min(best, std::hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]));
    CHECK(std::fabs(d[i] - best) < 1e-9 || (std::isinf(d[i]) && std::isinf(best)));
  }
  // knn vs brute force
  std::vector<Vec2> q;
  for (int i = 0; i < 2000; ++i) q.push_back({rnd() * 1000.0, rnd() * 800.0});
  std::vector<uint32_t> ids;
  for (uint32_t i = 0; i < 2000; i += 3) ids.push_back(i);
  auto grid = select::Grid2::build(q, ids, 1000.0, 800.0, 2.0);
  std::vector<std::pair<double, uint32_t>> res;
  for (uint32_t t = 0; t < 100; ++t) {
    uint32_t skip = t * 7 % 2000;
    grid.knn(q, q[skip], 6, skip, res);
    std::vector<std::pair<double, uint32_t>> bf;
    for (uint32_t i : ids)
      if (i != skip) bf.emplace_back(std::pow(q[i][0] - q[skip][0], 2) + std::pow(q[i][1] - q[skip][1], 2), i);
    std::sort(bf.begin(), bf.end());
    for (int u = 0; u < 6; ++u) CHECK(std::fabs(res[size_t(u)].first - bf[size_t(u)].first) < 1e-9);
    auto nn = grid.nearest_within(q, q[skip], 1e9, skip);
    CHECK(nn && std::fabs(nn->first - bf[0].first) < 1e-9);
  }
}

static void test_fingerprint() {
  using fingerprint::P2;
  P2 a{0, 0}, r{1, 0};
  std::array<P2, 3> n = {P2{1.31, 0.77}, P2{-0.9, 1.7}, P2{0.4, -2.1}};
  auto k0 = fingerprint::canonical_key(a, r, n);
  CHECK(k0.has_value());
  struct T {
    double ang, s;
    bool mirror;
  };
  int i = 0;
  for (T t : {T{0.3, 2.0, false}, T{2.9, 0.01, true}, T{-1.2, 340.0, true}}) {
    auto rot = [&](P2 p) {
      double y = t.mirror ? -p[1] : p[1];
      double c = std::cos(t.ang), sn = std::sin(t.ang);
      return P2{t.s * (c * p[0] - sn * y) + 10.0 * i, t.s * (sn * p[0] + c * y) - 3.0};
    };
    auto k = fingerprint::canonical_key(rot(a), rot(r), {rot(n[0]), rot(n[1]), rot(n[2])});
    CHECK(k && k->key == k0->key && k->flip == (k0->flip != t.mirror));
    ++i;
  }
  // mix32 is bijective on a sample
  std::set<uint32_t> seen;
  bool unique = true;
  for (uint32_t k = 0; k < 200000; ++k) unique = unique && seen.insert(fingerprint::mix32(k * 2654435761u)).second;
  CHECK(unique);
}

static void test_learned_math() {
  struct R {
    double x, v;
  };
  for (R r : {R{0.0, 1.0}, R{0.5, 0.4795001221869535}, R{1.0, 0.15729920705028513}, R{2.0, 0.004677734981047266}, R{3.0, 2.209049699858544e-5},
              R{5.0, 1.5374597944280348e-12}}) {
    double e = learned::erfc(r.x);
    CHECK(std::fabs((e - r.v) / r.v) < 1e-12);
  }
  uint64_t st = 0x123456789abcdef0ull;
  auto rnd = [&] {
    st ^= st << 13;
    st ^= st >> 7;
    st ^= st << 17;
    return double(st >> 11) / double(uint64_t(1) << 53);
  };
  for (int t = 0; t < 50; ++t) {
    std::array<float, 32> l;
    for (auto& x : l) x = float((rnd() - 0.5) * 2.0 * (1.0 + 30.0 * rnd()));
    std::vector<learned::Probe> got;
    learned::branch_probes(l, 64, 0.0, false, got);
    std::vector<size_t> idx(32);
    for (size_t i = 0; i < 32; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      return std::fabs(l[a]) != std::fabs(l[b]) ? std::fabs(l[a]) < std::fabs(l[b]) : a < b;
    });
    std::vector<double> all;
    for (uint32_t m = 0; m < (1u << 12); ++m) {
      double sum = 0;
      for (int i = 0; i < 12; ++i)
        if ((m >> i) & 1) sum += double(std::fabs(l[idx[size_t(i)]]));
      all.push_back(sum);
    }
    std::sort(all.begin(), all.end());
    CHECK(got.size() == 64);
    for (size_t i = 0; i < got.size() && i < 64; ++i) CHECK(std::fabs(got[i].cost - all[i]) < 1e-9);
    std::set<uint32_t> keys;
    for (auto& p : got) keys.insert(p.key);
    CHECK(keys.size() == 64);
  }
}

static void test_wcs() {
  Wcs w;
  w.ra = 163.0;
  w.dec = 71.5;
  w.crpix = {999.5, 666.0};
  w.cd = {{{-0.05, -0.007}, {0.007, -0.05}}};
  w.proj = Proj::Arc;
  struct Case {
    Vec2 px, want;
  };
  for (Case c : {Case{{-1.0, -1.0}, {267.53348, 35.736778}}, Case{{1998.0, 1331.0}, {106.222256, 20.253008}},
                 Case{{499.0, 899.0}, {199.957084, 49.257924}}}) {
    Vec2 s = w.pixel_to_sky(c.px[0], c.px[1]);
    CHECK(std::fabs(s[0] - c.want[0]) < 2e-6 && std::fabs(s[1] - c.want[1]) < 2e-6);
    auto back = w.sky_to_pixel(s[0], s[1]);
    CHECK(back && std::hypot((*back)[0] - c.px[0], (*back)[1] - c.px[1]) < 1e-6);
  }
  Sip sip;
  sip.order = 2;
  sip.a = {1e-4, 0.0, 0.0};
  sip.b = {0.0, 0.0, -1e-4};
  w.sip = sip;
  Vec2 s = w.pixel_to_sky(1499.0, 99.0);
  CHECK(std::fabs(s[0] - 35.674462) < 2e-6 && std::fabs(s[1] - 63.59072) < 2e-6);
  CHECK(w.fits_header().find("'RA---ARC-SIP'") != std::string::npos);
  CHECK(w.fits_header().size() % 2880 == 0);
  // STG / cylindrical round trips
  for (Proj p : {Proj::Tan, Proj::Stg, Proj::Car, Proj::Mer, Proj::Cyp, Proj::Arc}) {
    Wcs t;
    t.ra = 140;
    t.dec = 30;
    t.crpix = {100, 100};
    t.cd = {{{-0.05, 0.001}, {0.002, 0.05}}};
    t.proj = p;
    for (Vec2 px : {Vec2{10, 20}, Vec2{150, 180}, Vec2{100, 100}}) {
      Vec2 sky = t.pixel_to_sky(px[0], px[1]);
      auto back = t.sky_to_pixel(sky[0], sky[1]);
      CHECK(back && std::hypot((*back)[0] - px[0], (*back)[1] - px[1]) < 1e-6);
    }
  }
}

static std::vector<uint8_t> make_fits(const StarField& field, const std::vector<double>& amp, double sigma_psf,
                                      const std::vector<std::string>& extra = {}) {
  size_t w = field.width, h = field.height;
  std::vector<double> img(w * h, 1000.0);
  std::mt19937_64 rng(5);
  std::normal_distribution<double> noise(0.0, 4.0);
  for (auto& v : img) v += noise(rng);
  for (size_t i = 0; i < field.stars.size(); ++i) {
    const Star& s = field.stars[i];
    int x0 = int(s.x) - 8, y0 = int(s.y) - 8;
    for (int y = std::max(0, y0); y < std::min<int>(int(h), y0 + 17); ++y)
      for (int x = std::max(0, x0); x < std::min<int>(int(w), x0 + 17); ++x) {
        double r2 = (x - s.x) * (x - s.x) + (y - s.y) * (y - s.y);
        img[size_t(y) * w + size_t(x)] += amp[i] * std::exp(-r2 / (2 * sigma_psf * sigma_psf));
      }
  }
  std::vector<std::string> cards = {"SIMPLE  =                    T", "BITPIX  =                   16", "NAXIS   =                    2",
                                    "NAXIS1  = " + std::string(20 - std::to_string(w).size(), ' ') + std::to_string(w),
                                    "NAXIS2  = " + std::string(20 - std::to_string(h).size(), ' ') + std::to_string(h),
                                    "BZERO   =                32768"};
  cards.insert(cards.end(), extra.begin(), extra.end());
  cards.push_back("END");
  std::vector<uint8_t> out;
  for (auto& c : cards) {
    std::string pad = c + std::string(80 - c.size(), ' ');
    out.insert(out.end(), pad.begin(), pad.end());
  }
  out.resize((out.size() + 2879) / 2880 * 2880, ' ');
  for (double v : img) {
    int32_t iv = int32_t(std::lround(v)) - 32768;
    iv = std::clamp(iv, -32768, 32767);
    uint16_t u = uint16_t(int16_t(iv));
    out.push_back(uint8_t(u >> 8));
    out.push_back(uint8_t(u));
  }
  out.resize((out.size() + 2879) / 2880 * 2880, 0);
  return out;
}

static std::vector<uint8_t> make_tiff(uint32_t w, uint32_t h, int spp, const std::vector<uint8_t>& pixels, int compression, int orientation) {
  // little-endian baseline TIFF, one strip
  std::vector<uint8_t> strip = pixels;
  if (compression == 32773) {  // PackBits: literal runs of up to 128 bytes
    strip.clear();
    for (size_t i = 0; i < pixels.size(); i += 128) {
      size_t n = std::min<size_t>(128, pixels.size() - i);
      strip.push_back(uint8_t(n - 1));
      strip.insert(strip.end(), pixels.begin() + long(i), pixels.begin() + long(i + n));
    }
  }
  std::vector<uint8_t> out = {'I', 'I', 42, 0, 8, 0, 0, 0};
  struct E {
    uint16_t tag, type;
    uint32_t count, value;
  };
  uint32_t strip_off = 8 + 2 + 12 * 11 + 4;
  std::vector<E> es = {{256, 4, 1, w},          {257, 4, 1, h},
                       {258, 3, 1, 8},          {259, 3, 1, uint32_t(compression)},
                       {262, 3, 1, spp == 1 ? 1u : 2u}, {273, 4, 1, strip_off},
                       {274, 3, 1, uint32_t(orientation)}, {277, 3, 1, uint32_t(spp)},
                       {278, 4, 1, h},          {279, 4, 1, uint32_t(strip.size())},
                       {284, 3, 1, 1}};
  auto p16 = [&](uint16_t v) {
    out.push_back(uint8_t(v));
    out.push_back(uint8_t(v >> 8));
  };
  auto p32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i)));
  };
  p16(uint16_t(es.size()));
  for (auto& e : es) {
    p16(e.tag);
    p16(e.type);
    p32(e.count);
    p32(e.value);
  }
  p32(0);
  out.insert(out.end(), strip.begin(), strip.end());
  return out;
}

static void test_image_formats() {
  // PNG (8-bit gray and RGB), through our encoder and stb_image
  std::vector<uint8_t> g(16 * 8);
  for (size_t i = 0; i < g.size(); ++i) g[i] = uint8_t(i * 2);
  auto png = image_io::encode_png(16, 8, 1, g.data());
  auto lg = image_io::decode_bytes(png.data(), png.size());
  CHECK(lg.info.format == image_io::SourceFormat::Png && lg.info.bitpix == 8);
  CHECK(lg.image.width == 16 && lg.image.height == 8);
  bool same = true;
  for (size_t i = 0; i < g.size(); ++i) same = same && lg.image.pixels[i] == float(g[i]) * (1.0f / 255.0f);
  CHECK(same);
  std::vector<uint8_t> rgb(16 * 8 * 3);
  for (size_t i = 0; i < rgb.size(); ++i) rgb[i] = uint8_t(i * 7);
  auto png3 = image_io::encode_png(16, 8, 3, rgb.data());
  auto l3 = image_io::decode_bytes(png3.data(), png3.size());
  const float B8 = 1.0f / 255.0f;
  float expect = (0.2126f * float(rgb[0]) + 0.7152f * float(rgb[1]) + 0.0722f * float(rgb[2])) * B8;
  CHECK(l3.image.pixels[0] == expect);
  // TIFF: uncompressed gray, PackBits RGB, and EXIF-style orientation 6 (rotate 90 CW)
  std::vector<uint8_t> tg(8 * 4);
  for (size_t i = 0; i < tg.size(); ++i) tg[i] = uint8_t(i * 5);
  auto t1 = make_tiff(8, 4, 1, tg, 1, 1);
  auto d1 = image_io::decode_bytes(t1.data(), t1.size());
  CHECK(d1.info.format == image_io::SourceFormat::Tiff && d1.image.width == 8 && d1.image.height == 4);
  CHECK(d1.image.pixels[9] == float(tg[9]) * B8);
  std::vector<uint8_t> trgb(8 * 4 * 3);
  for (size_t i = 0; i < trgb.size(); ++i) trgb[i] = uint8_t(i * 3 + 1);
  auto t2 = make_tiff(8, 4, 3, trgb, 32773, 1);
  auto d2 = image_io::decode_bytes(t2.data(), t2.size());
  float e2 = (0.2126f * float(trgb[3]) + 0.7152f * float(trgb[4]) + 0.0722f * float(trgb[5])) * B8;
  CHECK(d2.image.pixels[1] == e2);
  auto t3 = make_tiff(8, 4, 1, tg, 1, 6);
  auto d3 = image_io::decode_bytes(t3.data(), t3.size());
  CHECK(d3.image.width == 4 && d3.image.height == 8);
  // rotate 90 CW: source (x, y) -> (h - 1 - y, x)
  CHECK(d3.image.pixels[size_t(2) * 4 + (4 - 1 - 1)] == float(tg[1 * 8 + 2]) * B8);
  // FITS hints from capture-style cards
  std::vector<std::string> cards = {"SIMPLE  =                    T", "BITPIX  =                    8", "NAXIS   =                    2",
                                    "NAXIS1  =                   16", "NAXIS2  =                   16", "RA      = 83.8", "DEC     = -5.4",
                                    "PIXSCALE= 1.2", "DATE-OBS= '2024-03-01T22:10:00'", "ROWORDER= 'BOTTOM-UP'", "END"};
  std::vector<uint8_t> f;
  for (auto& c : cards) {
    std::string pad = c + std::string(80 - c.size(), ' ');
    f.insert(f.end(), pad.begin(), pad.end());
  }
  f.resize(2880, ' ');
  f.resize(2880 + 2880, 0);
  auto fi = image_io::decode_bytes(f.data(), f.size());
  CHECK(fi.info.hint && fi.info.hint->scale_arcsec && *fi.info.hint->scale_arcsec == 1.2);
  CHECK(fi.info.hint && fi.info.hint->position && (*fi.info.hint->position)[0] == 83.8 && (*fi.info.hint->position)[1] == -5.4);
  CHECK(fi.image.epoch && *fi.image.epoch == 2024.5);
  CHECK(image_io::fits_row_order(f.data(), f.size()).value_or("") == "BOTTOM-UP");
  bool threw = false;
  try {
    image_io::decode_bytes(f.data(), 3000);
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
  threw = false;
  try {
    const char* junk = "not an image at all";
    image_io::decode_bytes(reinterpret_cast<const uint8_t*>(junk), 19);
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
  // preview
  image_io::GrayImage big;
  big.width = 4000;
  big.height = 3000;
  big.pixels.assign(size_t(4000) * 3000, 1000.0f);
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.0f, 4.0f);
  for (auto& v : big.pixels) v += nd(rng);
  big.pixels[5] = std::numeric_limits<float>::quiet_NaN();
  CHECK(big.preview_size() == std::make_pair(1600u, 1200u));
  auto pv = big.preview();
  auto pd = image_io::decode_bytes(pv.data(), pv.size());
  CHECK(pd.image.width == 1600 && pd.image.height == 1200);
  std::vector<float> med = pd.image.pixels;
  std::nth_element(med.begin(), med.begin() + long(med.size() / 2), med.end());
  CHECK(std::fabs(med[med.size() / 2] - 0.15f) < 0.05f);
}

static void test_synthetic_sky() {
  TempDir tmp;
  fs::path index = tmp.path / "index";
  auto [field, expected] = support::synthetic_catalog(index);
  Engine engine = Engine::open(index);
  solve::SolveConfig cfg;
  cfg.deadline_ms = 10000.0;
  for (int i = 0; i < 2; ++i) {
    auto result = engine.solve_sources(field, cfg);
    CHECK(result.solution.status == "solved");
    CHECK(std::fabs(result.solution.ra - expected.ra) < 0.001);
    CHECK(std::fabs(result.solution.dec - expected.dec) < 0.001);
  }
  // CLI behaviour
  fs::path input = tmp.path / "stars.json";
  { std::ofstream(input, std::ios::binary) << field.to_json().dump(false); }
  fs::path wcs = tmp.path / "solution.wcs";
  fs::path out = tmp.path / "out.json";
  int rc = run_cli("solve \"" + input.string() + "\" --index \"" + index.string() + "\" --deadline-ms 10000 --wcs \"" + wcs.string() + "\"", out);
  CHECK(rc == 0);
  Json v = parse_json(read_file_string(out.string()));
  CHECK(v.get("schema_version")->as_u64().value() == 1);
  CHECK(v.get("solution")->get("status")->as_str() == "solved");
  CHECK(read_file_string(wcs.string()).find("RA---TAN") != std::string::npos);
  auto before = read_file_bytes(wcs.string());
  rc = run_cli("solve \"" + input.string() + "\" --index \"" + index.string() + "\" --wcs \"" + wcs.string() + "\"", out);
  CHECK(rc == 2);
  CHECK(read_file_bytes(wcs.string()) == before);
  rc = run_cli("batch \"" + input.string() + "\" \"" + (tmp.path / "missing.fits").string() + "\" \"" + input.string() + "\" --index \"" +
                   index.string() + "\" --deadline-ms 10000",
               out);
  CHECK(rc == 2);
  {
    std::istringstream is(read_file_string(out.string()));
    std::string line;
    std::vector<Json> rows;
    while (std::getline(is, line))
      if (!line.empty()) rows.push_back(parse_json(line));
    CHECK(rows.size() == 3);
    if (rows.size() == 3) {
      CHECK(rows[0].get("result")->get("solution")->get("status")->as_str() == "solved");
      CHECK(rows[1].get("error") && rows[1].get("error")->is_string());
      CHECK(rows[2].get("result")->get("solution")->get("status")->as_str() == "solved");
    }
  }
  StarField empty = field;
  empty.stars.clear();
  { std::ofstream(input, std::ios::binary) << empty.to_json().dump(false); }
  rc = run_cli("solve \"" + input.string() + "\" --index \"" + index.string() + "\"", out);
  CHECK(rc == 1);
  CHECK(parse_json(read_file_string(out.string())).get("solution")->get("status")->as_str() == "unsolved");
  solve::SolveConfig bad = cfg;
  bad.deadline_ms = NaN;
  bool threw = false;
  try {
    engine.solve_sources(field, bad);
  } catch (const Error&) {
    threw = true;
  }
  CHECK(threw);
  // the builder never replaces an existing directory
  fs::path marker = tmp.path / "untouched";
  { std::ofstream(marker, std::ios::binary) << "keep"; }
  rc = run_cli("build --d80 \"" + (tmp.path / "missing").string() + "\" --out \"" + tmp.path.string() + "\"", out);
  CHECK(rc == 2);
  CHECK(read_file_string(marker.string()) == "keep");
}

static void test_image_pipeline() {
  TempDir tmp;
  fs::path index = tmp.path / "index";
  auto [field, expected] = support::synthetic_catalog(index);
  std::vector<double> amp;
  for (size_t i = 0; i < field.stars.size(); ++i) {
    // flux = 10^(-0.4 mag): brighter stars first (mag 6 + 0.01 * i); amplitude 30000 at mag 6
    double mag = -2.5 * std::log10(field.stars[i].flux);
    amp.push_back(30000.0 * std::pow(10.0, -0.4 * (mag - 6.0)));
  }
  auto fits = make_fits(field, amp, 1.6);
  fs::path img = tmp.path / "frame.fits";
  { std::ofstream(img, std::ios::binary).write(reinterpret_cast<const char*>(fits.data()), std::streamsize(fits.size())); }
  auto loaded = image_io::decode_bytes(fits.data(), fits.size());
  CHECK(loaded.image.width == 1000 && loaded.image.height == 750);
  CHECK(loaded.info.format == image_io::SourceFormat::Fits && loaded.info.bitpix == 16);
  CHECK(std::fabs(loaded.image.pixels[0] - 1000.0f) < 40.0f);
  StarField det = extract(loaded.image, ExtractOptions::for_source(loaded.info));
  CHECK(det.stars.size() > 100);
  // detections agree with the truth for the bright stars
  size_t matched = 0, bright = 0;
  for (size_t i = 0; i < field.stars.size() && i < 120; ++i) {
    ++bright;
    for (const Star& d : det.stars)
      if (std::hypot(d.x - field.stars[i].x, d.y - field.stars[i].y) < 0.4) {
        ++matched;
        break;
      }
  }
  std::printf("  extraction: %zu detections, %zu/%zu bright stars within 0.4 px\n", det.stars.size(), matched, bright);
  CHECK(matched * 10 >= bright * 8);
  Engine engine = Engine::open(index);
  solve::SolveConfig cfg;
  cfg.deadline_ms = 10000.0;
  auto result = engine.solve_path(img, cfg);
  std::printf("  image solve: %s, %zu stars, rms %.3f px, %.1f ms\n", result.solution.status.c_str(), result.n_stars, result.solution.rms_px,
              result.timings.total_ms);
  CHECK(result.solution.status == "solved");
  CHECK(std::fabs(result.solution.ra - expected.ra) < 0.01);
  CHECK(std::fabs(result.solution.dec - expected.dec) < 0.01);
  CHECK(std::fabs(result.solution.scale_arcsec - 3.0) < 0.02);
  // a FITS header with a good and with a wrong position hint: a wrong hint must never prevent a solve
  for (const char* dec : {"-5.4", "40.0"}) {
    auto hinted = make_fits(field, amp, 1.6, {"RA      = 83.8", std::string("DEC     = ") + dec, "PIXSCALE= 3.0"});
    fs::path himg = tmp.path / "hinted.fits";
    { std::ofstream(himg, std::ios::binary).write(reinterpret_cast<const char*>(hinted.data()), std::streamsize(hinted.size())); }
    auto hr = engine.solve_path(himg, cfg);
    std::printf("  hinted (dec %s): %s, hint_used=%d fallback=%d\n", dec, hr.solution.status.c_str(), int(hr.solution.stats.hint_used),
                int(hr.solution.stats.hint_fallback));
    CHECK(hr.solution.status == "solved");
    CHECK(hr.solution.stats.hint_used);
    CHECK(std::fabs(hr.solution.ra - expected.ra) < 0.01);
  }
}

void run_measure_tests();

int main() {
  struct T {
    const char* name;
    std::function<void()> fn;
  } tests[] = {{"json numbers", test_json_numbers}, {"sha256", test_sha256},         {"sphere and select", test_sphere_and_select},
               {"fingerprint", test_fingerprint}, {"learned math", test_learned_math},   {"wcs", test_wcs}, {"image formats", test_image_formats},               {"synthetic sky (engine, CLI, builder)", test_synthetic_sky},
               {"image pipeline", test_image_pipeline}};
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
  run_measure_tests();
  std::printf("%d checks, %d failures\n", checks, failures);
  return failures ? 1 : 0;
}
