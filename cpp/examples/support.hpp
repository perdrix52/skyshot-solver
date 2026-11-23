// Deterministic artificial sky for examples and tests; no external catalogue. Port of examples/support/mod.rs.
#pragma once
#include <filesystem>

#include "skyshot/builder.hpp"
#include "skyshot/geometry.hpp"

namespace support {

inline std::pair<skyshot::StarField, skyshot::Wcs> synthetic_catalog(const std::filesystem::path& dir,
                                                                     std::optional<std::filesystem::path> coder = std::nullopt) {
  using namespace skyshot;
  if (!std::filesystem::create_directory(dir)) throw Error("create_directory failed: " + dir.string());
  Wcs wcs;
  wcs.ra = 83.8;
  wcs.dec = -5.4;
  wcs.crpix = {499.5, 374.5};
  wcs.cd = {{{-3.0 / 3600.0, 0.0}, {0.0, 3.0 / 3600.0}}};
  uint64_t seed = 0x9e3779b97f4a7c15ull;
  auto rnd = [&] {
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    return double(seed >> 11) / double(uint64_t(1) << 53);
  };
  std::vector<Vec3> exact;
  std::vector<double> dist;
  std::vector<float> mags;
  StarField field;
  field.width = 1000;
  field.height = 750;
  for (int i = 0; i < 800; ++i) {
    double x = rnd() * 1600.0 - 300.0, y = rnd() * 1350.0 - 300.0;
    Vec2 s = wcs.pixel_to_sky(x, y);
    Vec3 v = sphere::unit(s[0], s[1]);
    double d = PI;
    for (const Vec3& b : exact) d = std::min(d, sphere::angle(v, b));
    exact.push_back(v);
    dist.push_back(d);
    float mag = 6.0f + float(i) * 0.01f;
    mags.push_back(mag);
    if (x >= 0.0 && x < 1000.0 && y >= 0.0 && y < 750.0) {
      Star st;
      st.x = x;
      st.y = y;
      st.flux = std::pow(10.0, -0.4 * double(mag));
      st.sigma = 0.2;
      field.stars.push_back(st);
    }
  }
  builder::BuildConfig cfg;
  cfg.out = dir;
  cfg.dump_levels = false;
  cfg.caps = {};
  cfg.coder = coder;
  builder::write_index(cfg, std::chrono::steady_clock::now(), exact, dist, mags, exact.size(),
                       "synthetic-demo-v1; artificial stars, not a real sky catalogue", std::nullopt);
  return {field, wcs};
}

}  // namespace support
