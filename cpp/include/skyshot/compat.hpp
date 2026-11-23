// Output contracts for capture tools (ASTAP .ini, astrometry.net .wcs/.solved). Port of src/compat.rs.
#pragma once
#include <filesystem>

#include "solve.hpp"
#include "types.hpp"

namespace skyshot::compat {

struct Cdelt {
  double cdelt1, cdelt2, crota1, crota2;
};
Cdelt cdelt_crota(const std::array<std::array<double, 2>, 2>& cd);

struct Outcome {
  bool solved = false;
  std::optional<Wcs> wcs;
  uint32_t width = 0, height = 0;
  size_t n_matches = 0;
  double rms_px = NaN;
  static Outcome from(const solve::Solution& s);
};

bool write_astap(const std::filesystem::path& base, const Outcome& sol, const std::string& cmdline, bool write_wcs);
std::string astap_ini(const Outcome& sol, const std::string& cmdline);
bool write_solve_field(const std::filesystem::path& base, const Outcome& sol);
std::string header(const Wcs& w, const Outcome& sol);

}  // namespace skyshot::compat
