// Core data types. Port of src/types.rs and src/pixelgrid.rs.
#pragma once
#include <memory>

#include "common.hpp"
#include "json.hpp"

namespace skyshot {

namespace star_flags {
inline constexpr uint32_t SATURATED = 1u << 0;
inline constexpr uint32_t HOT_PIXEL = 1u << 1;
inline constexpr uint32_t BLENDED = 1u << 2;
inline constexpr uint32_t NEAR_EDGE = 1u << 3;
inline constexpr uint32_t TRAILED = 1u << 4;
inline constexpr uint32_t ON_LINE = 1u << 5;
inline constexpr uint32_t RIDGE = 1u << 6;
inline constexpr uint32_t ELONGATED = 1u << 7;
inline constexpr uint32_t EXTENDED = 1u << 8;
inline constexpr uint32_t REJECT = HOT_PIXEL | ON_LINE | RIDGE | ELONGATED | EXTENDED;
}  // namespace star_flags

struct Star {
  double x = 0.0, y = 0.0, flux = 0.0;
  double sigma = 0.25;
  double snr = 20.0;
  double fwhm = 3.0;
  double elongation = 1.0;
  bool saturated = false;
  double peak = 0.0;
  double pa = 0.0;
  uint32_t flags = 0;

  Json to_json() const;
  static Star from_json(const Json& j);
};

struct CatalogStar {
  uint64_t source_id = 0;
  double ra = 0.0, dec = 0.0;
  float mag = 0.0f;
  float pmra = 0.0f, pmdec = 0.0f;
  float epoch = 2016.0f;
};

struct StarField {
  uint32_t width = 0, height = 0;
  std::vector<Star> stars;
  std::optional<double> epoch;

  Json to_json() const;
  static StarField from_json(const Json& j);
};

struct Sip {
  uint32_t order = 0;
  std::vector<double> a, b;

  static std::vector<std::pair<uint32_t, uint32_t>> terms(uint32_t order);
  std::pair<double, double> eval(double u, double v) const;
  struct EvalJac {
    double fa, fb;
    double j[2][2];
  };
  EvalJac eval_jacobian(double u, double v) const;
  std::array<std::array<double, 2>, 2> jacobian(double u, double v) const;

  Json to_json() const;
  static Sip from_json(const Json& j);
};

enum class Proj { Tan, Arc, Stg, Car, Mer, Cyp };
inline bool is_tan(Proj p) { return p == Proj::Tan; }
inline bool is_cylindrical(Proj p) { return p == Proj::Car || p == Proj::Mer || p == Proj::Cyp; }
const char* proj_code(Proj p);

struct Refraction {
  Vec2 zenith{0, 0};
  double scale = 0, rms_px = 0, rms_refracted_px = 0, min_alt_deg = 0;
  size_t n_pairs = 0;
  Json to_json() const;
};

struct PixelGrid {
  size_t cols = 0, rows = 0;
  Vec2 origin{0, 0}, step{0, 0};
  std::vector<Vec2> offsets;

  // Returns an error message when the grid is malformed.
  std::optional<std::string> validate() const;
  Vec2 offset(Vec2 p) const;
  Vec2 forward(Vec2 p) const;
  std::optional<Vec2> inverse(Vec2 p) const;
  Json to_json() const;
  static PixelGrid from_json(const Json& j);
};

struct Wcs {
  double ra = 0, dec = 0;
  Vec2 crpix{0, 0};
  std::array<std::array<double, 2>, 2> cd{{{0, 0}, {0, 0}}};
  std::optional<Sip> sip;
  Proj proj = Proj::Tan;
  std::optional<Vec2> native_pole;
  std::shared_ptr<const PixelGrid> local_grid;

  Vec2 correct_pixel(Vec2 p) const;
  std::optional<Vec2> uncorrect_pixel(Vec2 p) const;
  double width_deg(uint32_t width, uint32_t height) const;
  double arc_deg(Vec2 a, Vec2 b) const;
  double determinant() const;
  Vec2 pixel_to_sky(double x, double y) const;
  std::optional<Vec2> sky_to_pixel_fast(double ra, double dec) const;
  std::optional<Vec2> sky_to_pixel(double ra, double dec) const;
  double scale_arcsec() const;
  double rotation_deg() const;
  std::string fits_header() const;

  Json to_json() const;
  static Wcs from_json(const Json& j);
};

}  // namespace skyshot
