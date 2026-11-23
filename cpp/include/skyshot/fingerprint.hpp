// Per-star fingerprint (key spec v1). Port of src/fingerprint.rs.
#pragma once
#include "common.hpp"

namespace skyshot::fingerprint {

inline constexpr double RHO_STEP = 0.12;
inline constexpr uint32_t RHO_BINS = 20;
inline constexpr uint32_t THETA_BINS = 60;
inline constexpr uint32_t THETA1_BINS = THETA_BINS / 2;
inline constexpr double THETA_STEP = TAU / double(THETA_BINS);
inline constexpr double RHO_TOL_FLOOR = 0.03;
inline constexpr double THETA_TOL_FLOOR = 1.5 * PI / 180.0;
inline constexpr size_t MAX_VARIANTS = 96;

using P2 = Vec2;

inline uint32_t rho_bin(double r) {
  return std::min(sat_cast<uint32_t>(fmax_r(std::floor(r / RHO_STEP), 0.0)), RHO_BINS - 1);
}
inline uint32_t t1_bin(double t) {
  return std::min(sat_cast<uint32_t>(fmax_r(std::floor(t / THETA_STEP), 0.0)), THETA1_BINS - 1);
}
inline uint32_t t_bin(double t) {
  return sat_cast<uint32_t>(std::floor(rem_euclid(t, TAU) / THETA_STEP)) % THETA_BINS;
}

inline uint32_t pack(const std::array<uint32_t, 6>& b) {
  return b[0] | (b[1] << 5) | (b[2] << 10) | (b[3] << 15) | (b[4] << 20) | (b[5] << 26);
}
inline std::array<uint32_t, 6> unpack(uint32_t k) {
  return {k & 31, (k >> 5) & 31, (k >> 10) & 31, (k >> 15) & 31, (k >> 20) & 63, (k >> 26) & 63};
}

struct BasisCoords {
  std::array<std::pair<double, double>, 3> c;
  double len;
};
std::optional<BasisCoords> basis_coords(P2 a, P2 r, const std::array<P2, 3>& nbrs);

struct CanonicalKey {
  uint32_t key;
  bool flip;
  double len;
};
std::optional<CanonicalKey> canonical_key(P2 a, P2 r, const std::array<P2, 3>& nbrs);

struct KeyVariant {
  uint32_t key;
  bool flip;
  bool operator==(const KeyVariant& o) const { return key == o.key && flip == o.flip; }
  bool operator<(const KeyVariant& o) const { return key != o.key ? key < o.key : flip < o.flip; }
};

std::optional<double> image_gate(P2 a, P2 r, const std::array<P2, 3>& nbrs, double sigma_px);
std::optional<double> image_keys(P2 a, P2 r, const std::array<P2, 3>& nbrs, double sigma_px,
                                 std::vector<KeyVariant>& out);

inline uint32_t mix32(uint32_t h) {
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

}  // namespace skyshot::fingerprint
