#include "skyshot/d80.hpp"

#include <fstream>

#include "skyshot/geometry.hpp"

namespace skyshot::d80 {

static const size_t RINGS[18] = {1, 3, 9, 15, 21, 27, 33, 38, 43, 48, 52, 56, 60, 63, 65, 67, 68, 69};

void decode_raw(const uint8_t* data, size_t len, std::vector<RawStar>& out) {
  if (!(len >= 110 && data[109] == 5 && (len - 110) % 5 == 0)) throw Error("Invalid 1476-5 data");
  bool has_band = false;
  int32_t dec9 = 0;
  int16_t mag10 = 0;
  for (size_t i = 110; i + 5 <= len; i += 5) {
    const uint8_t* r = data + i;
    if (r[0] == 255 && r[1] == 255 && r[2] == 255) {
      dec9 = int32_t(r[3]) - 128;
      mag10 = int16_t(int16_t(r[4]) - 16);
      has_band = true;
      continue;
    }
    if (!has_band) throw Error("Missing D80 section header");
    uint32_t ra = uint32_t(r[0]) | (uint32_t(r[1]) << 8) | (uint32_t(r[2]) << 16);
    int32_t dec = int32_t(r[3]) + int32_t(r[4]) * 256 + dec9 * 65536;
    out.push_back(RawStar{ra, dec, mag10});
  }
}

D80 D80::open(const std::filesystem::path& directory) {
  D80 d;
  for (size_t ring = 0; ring < 36; ++ring) {
    size_t count = RINGS[std::min<size_t>(ring, 35 - ring)];
    double low = fmax_r(-90.0 + (double(ring) - 0.5) * 180.0 / 35.0, -90.0);
    double high = fmin_r(-90.0 + (double(ring) + 0.5) * 180.0 / 35.0, 90.0);
    for (size_t sector = 0; sector < count; ++sector) {
      uint64_t id = uint64_t((ring + 1) * 100 + sector + 1);
      char name[32];
      std::snprintf(name, sizeof name, "d80_%04llu.1476", (unsigned long long)id);
      std::filesystem::path path = directory / name;
      std::ifstream f(path, std::ios::binary);
      if (!f) throw Error("Missing D80 tile " + path.string());
      uint8_t header[110];
      f.read(reinterpret_cast<char*>(header), 110);
      if (f.gcount() != 110) throw Error("failed to fill whole buffer");
      f.seekg(0, std::ios::end);
      uint64_t len = uint64_t(f.tellg());
      if (!(header[109] == 5 && len > 110 && (len - 110) % 5 == 0)) throw Error("Unsupported D80 tile " + path.string());
      std::string description(reinterpret_cast<const char*>(header), 109);
      if (description.find("GAIA DR3") == std::string::npos || description.find("Epoch=2025") == std::string::npos)
        throw Error("Expected Gaia DR3 D80 epoch 2025");
      d.bytes += len;
      d.tiles.push_back(Tile{path, id, (double(sector) + 0.5) * 360.0 / double(count), 180.0 / double(count), low, high});
    }
  }
  return d;
}

std::vector<std::filesystem::path> D80::tile_paths() const {
  std::vector<std::filesystem::path> out;
  for (auto& t : tiles) out.push_back(t.path);
  return out;
}

void decode_records(const uint8_t* data, size_t len, uint64_t tile, std::vector<CatalogStar>& out) {
  if (!(len >= 110 && data[109] == 5 && (len - 110) % 5 == 0)) throw Error("Invalid 1476-5 data");
  bool has_band = false;
  int32_t dec9 = 0;
  float mag = 0;
  size_t index = 0;
  for (size_t i = 110; i + 5 <= len; i += 5, ++index) {
    const uint8_t* r = data + i;
    if (r[0] == 255 && r[1] == 255 && r[2] == 255) {
      dec9 = int32_t(r[3]) - 128;
      mag = (float(r[4]) - 16.0f) / 10.0f;
      has_band = true;
      continue;
    }
    if (!has_band) throw Error("Missing D80 section header");
    double ra = double(uint32_t(r[0]) | (uint32_t(r[1]) << 8) | (uint32_t(r[2]) << 16)) * 360.0 / 16777215.0;
    double dec = double(int32_t(r[3]) + int32_t(r[4]) * 256 + dec9 * 65536) * 90.0 / 8388607.0;
    if (!(std::fabs(dec) <= 90.00002)) throw Error("Invalid D80 declination");
    CatalogStar s;
    s.source_id = (tile << 32) | uint64_t(index);
    s.ra = ra;
    s.dec = clampd(dec, -90.0, 90.0);
    s.mag = mag;
    s.pmra = 0.0f;
    s.pmdec = 0.0f;
    s.epoch = 2025.0f;
    out.push_back(s);
  }
}

std::vector<std::pair<uint32_t, CatalogStar>> D80::nearby(double ra, double dec, double radius) const {
  if (!(std::isfinite(ra) && std::isfinite(dec) && std::fabs(dec) <= 90.0 && std::isfinite(radius) && radius > 0.0 && radius <= 30.0))
    throw Error("Invalid catalog cone");
  double ra_radius = (std::fabs(dec) + radius >= 90.0)
                         ? 180.0
                         : to_degrees(std::asin(clampd(std::sin(to_radians(radius)) / std::cos(to_radians(dec)), -1.0, 1.0)));
  Vec3 direction = geometry::unit(ra, dec);
  std::vector<std::pair<uint32_t, CatalogStar>> result;
  double cos_r = std::cos(to_radians(radius));
  for (const Tile& tile : tiles) {
    double delta_ra = rem_euclid(ra - tile.ra + 180.0, 360.0) - 180.0;
    if (tile.low > dec + radius || tile.high < dec - radius || std::fabs(delta_ra) > ra_radius + tile.half_ra) continue;
    auto data = read_file_bytes(tile.path.string());
    std::vector<CatalogStar> stars;
    decode_records(data.data(), data.size(), tile.id, stars);
    for (const CatalogStar& star : stars) {
      if (std::fabs(star.dec - dec) > radius) continue;
      double delta = rem_euclid(star.ra - ra + 180.0, 360.0) - 180.0;
      if (std::fabs(delta) > ra_radius) continue;
      Vec3 u = geometry::unit(star.ra, star.dec);
      double dotp = direction[0] * u[0] + direction[1] * u[1] + direction[2] * u[2];
      if (dotp >= cos_r) result.emplace_back(uint32_t(result.size()), star);
    }
  }
  return result;
}

}  // namespace skyshot::d80
