#include "skyshot/compat.hpp"

#include <cstdio>
#include <fstream>

namespace skyshot::compat {

Cdelt cdelt_crota(const std::array<std::array<double, 2>, 2>& cd) {
  double det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
  double sign = det < 0.0 ? -1.0 : 1.0;
  double cdelt1 = sign * std::sqrt(cd[0][0] * cd[0][0] + cd[1][0] * cd[1][0]);
  double cdelt2 = std::sqrt(cd[0][1] * cd[0][1] + cd[1][1] * cd[1][1]);
  double crota1 = to_degrees(std::atan2(sign * cd[1][0], sign * cd[0][0]));
  double crota2 = to_degrees(std::atan2(-cd[0][1], cd[1][1]));
  return {cdelt1, cdelt2, crota1, crota2};
}

Outcome Outcome::from(const solve::Solution& s) {
  Outcome o;
  o.solved = s.status == "solved" && s.wcs.has_value();
  if (o.solved) o.wcs = s.wcs;
  o.width = s.width;
  o.height = s.height;
  o.n_matches = s.n_matches;
  o.rms_px = s.rms_px;
  return o;
}

static std::string fmt_e(double v) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.12E", v);
  std::string s(buf);
  size_t e = s.find('E');
  if (e == std::string::npos) return s;
  return s.substr(0, e) + "E" + std::to_string(std::atoi(s.c_str() + e + 1));
}

static std::filesystem::path with_ext(const std::filesystem::path& base, const char* ext) {
  std::filesystem::path p = base;
  p.replace_extension(ext);
  return p;
}

static void write_file(const std::filesystem::path& p, const std::string& data) {
  std::ofstream f(p, std::ios::binary);
  if (!f) throw Error("write " + p.string());
  f.write(data.data(), std::streamsize(data.size()));
}

bool write_astap(const std::filesystem::path& base, const Outcome& sol, const std::string& cmdline, bool write_wcs) {
  write_file(with_ext(base, "ini"), astap_ini(sol, cmdline));
  if (sol.solved && write_wcs && sol.wcs) write_file(with_ext(base, "wcs"), header(*sol.wcs, sol));
  return sol.solved;
}

std::string astap_ini(const Outcome& sol, const std::string& cmdline) {
  std::string ini;
  ini += std::string("PLTSOLVD=") + (sol.solved ? "T" : "F") + "\n";
  if (sol.solved && sol.wcs) {
    const Wcs& w = *sol.wcs;
    auto c = cdelt_crota(w.cd);
    ini += "CRPIX1=" + fmt_e(w.crpix[0] + 1.0) + "\n";
    ini += "CRPIX2=" + fmt_e(w.crpix[1] + 1.0) + "\n";
    ini += "CRVAL1=" + fmt_e(w.ra) + "\n";
    ini += "CRVAL2=" + fmt_e(w.dec) + "\n";
    ini += "CDELT1=" + fmt_e(c.cdelt1) + "\n";
    ini += "CDELT2=" + fmt_e(c.cdelt2) + "\n";
    ini += "CROTA1=" + fmt_e(c.crota1) + "\n";
    ini += "CROTA2=" + fmt_e(c.crota2) + "\n";
    ini += "CD1_1=" + fmt_e(w.cd[0][0]) + "\n";
    ini += "CD1_2=" + fmt_e(w.cd[0][1]) + "\n";
    ini += "CD2_1=" + fmt_e(w.cd[1][0]) + "\n";
    ini += "CD2_2=" + fmt_e(w.cd[1][1]) + "\n";
    char buf[128];
    std::snprintf(buf, sizeof buf, "WARNING=Skyshot Solver blind solve: %zu matches, rms %.2f px\n", sol.n_matches, sol.rms_px);
    ini += buf;
  } else {
    ini += "ERROR=No solution found\n";
  }
  ini += "CMDLINE=" + cmdline + "\n";
  return ini;
}

bool write_solve_field(const std::filesystem::path& base, const Outcome& sol) {
  if (sol.solved && sol.wcs) {
    write_file(with_ext(base, "wcs"), header(*sol.wcs, sol));
    write_file(with_ext(base, "solved"), std::string(1, '\x01'));
    return true;
  }
  return false;
}

static std::string pad_right(const std::string& s, size_t w) { return s.size() >= w ? s : s + std::string(w - s.size(), ' '); }
static std::string pad_left(const std::string& s, size_t w) { return s.size() >= w ? s : std::string(w - s.size(), ' ') + s; }

std::string header(const Wcs& w, const Outcome& sol) {
  std::string base = w.fits_header();
  std::vector<std::string> cards;
  for (size_t i = 0; i < base.size(); i += 80) {
    std::string c = base.substr(i, 80);
    std::string t = trim(c);
    if (!t.empty() && t != "END") cards.push_back(c);
  }
  cards.push_back(pad_right("IMAGEW", 8) + "= " + pad_left(std::to_string(sol.width), 20));
  cards.push_back(pad_right("IMAGEH", 8) + "= " + pad_left(std::to_string(sol.height), 20));
  cards.push_back(pad_right("PLTSOLVD", 8) + "= " + pad_left("T", 20));
  char buf[128];
  std::snprintf(buf, sizeof buf, "COMMENT Skyshot Solver matches=%zu rms_px=%.3f", sol.n_matches, sol.rms_px);
  cards.push_back(buf);
  cards.push_back("END");
  std::string out;
  for (auto& c : cards) out += pad_right(c.substr(0, std::min<size_t>(c.size(), 80)), 80);
  size_t n = (out.size() + 2879) / 2880 * 2880;
  out.append(n - out.size(), ' ');
  return out;
}

}  // namespace skyshot::compat
