// Reader for the documented ASTAP 1476-5 catalogue format (Gaia DR3, epoch 2025). Port of src/d80.rs.
#pragma once
#include <filesystem>

#include "common.hpp"
#include "types.hpp"

namespace skyshot::d80 {

struct RawStar {
  uint32_t ra;
  int32_t dec;
  int16_t mag10;
  double ra_deg() const { return double(ra) * 360.0 / 16777215.0; }
  double dec_deg() const { return clampd(double(dec) * 90.0 / 8388607.0, -90.0, 90.0); }
  float mag() const { return float(mag10) / 10.0f; }
};

// Decode one tile into raw records. Throws skyshot::Error on invalid data.
void decode_raw(const uint8_t* data, size_t len, std::vector<RawStar>& out);

struct Tile {
  std::filesystem::path path;
  uint64_t id;
  double ra, half_ra, low, high;
};

class D80 {
 public:
  std::vector<Tile> tiles;
  uint64_t bytes = 0;
  static D80 open(const std::filesystem::path& directory);
  std::vector<std::filesystem::path> tile_paths() const;
  // Catalogue stars within `radius` degrees of (ra, dec), as (running index, star). Throws on an invalid cone.
  std::vector<std::pair<uint32_t, CatalogStar>> nearby(double ra, double dec, double radius) const;
};

// Decode one tile into catalogue stars (source_id = tile << 32 | index).
void decode_records(const uint8_t* data, size_t len, uint64_t tile, std::vector<CatalogStar>& out);

}  // namespace skyshot::d80
