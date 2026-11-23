#include "skyshot/hints.hpp"

#include "skyshot/types.hpp"

namespace skyshot {

std::optional<SolveHint> SolveHint::merge(const std::optional<SolveHint>& primary,
                                          const std::optional<SolveHint>& fallback) {
  SolveHint p, f;
  if (primary) {
    if (auto s = primary->sanitized()) p = *s;
  }
  if (fallback) {
    if (auto s = fallback->sanitized()) f = *s;
  }
  SolveHint m;
  m.position = p.position ? p.position : f.position;
  m.scale_arcsec = p.scale_arcsec ? p.scale_arcsec : f.scale_arcsec;
  return m.sanitized();
}

std::optional<SolveHint> SolveHint::sanitized() const {
  SolveHint h;
  if (position) {
    double ra = (*position)[0], dec = (*position)[1];
    if (std::isfinite(ra) && ra >= 0.0 && ra <= 360.0 && std::isfinite(dec) && dec >= -90.0 && dec <= 90.0)
      h.position = Vec2{rem_euclid(ra, 360.0), dec};
  }
  if (scale_arcsec) {
    double s = *scale_arcsec;
    if (std::isfinite(s) && s > 0.0 && s <= 3600.0) h.scale_arcsec = s;
  }
  if (h.position || h.scale_arcsec) return h;
  return std::nullopt;
}

std::optional<double> SolveHint::fov_deg(uint32_t width, uint32_t height) const {
  auto s = sanitized();
  if (!s || !s->scale_arcsec) return std::nullopt;
  double f = *s->scale_arcsec * double(std::max(width, height)) / 3600.0;
  if (f >= 0.001 && f < 180.0) return f;
  return std::nullopt;
}

std::optional<SolveHint> SolveHint::for_frame(uint32_t width, uint32_t height) const {
  auto h = sanitized();
  if (!h) return std::nullopt;
  if (!h->fov_deg(width, height)) h->scale_arcsec.reset();
  return h->sanitized();
}

namespace {

std::optional<std::string> get(const FitsHeader& h, const std::string& k) {
  auto it = h.find(k);
  if (it == h.end()) return std::nullopt;
  return it->second;
}

std::string replace_d(std::string s) {
  for (auto& c : s)
    if (c == 'D' || c == 'd') c = 'E';
  return s;
}

std::optional<double> number(const FitsHeader& h, const std::string& key) {
  auto v = get(h, key);
  if (!v) return std::nullopt;
  auto p = parse_f64(replace_d(trim(*v)));
  if (p && std::isfinite(*p)) return p;
  return std::nullopt;
}

std::optional<double> coordinate(const std::string& s, bool ra, bool hours) {
  std::vector<std::string> parts;
  std::string cur;
  std::string t = trim(s);
  auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f'; };
  for (char c : t) {
    if (c == ':' || is_ws(c)) {
      if (!cur.empty()) parts.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) parts.push_back(cur);
  if (parts.empty()) return std::nullopt;
  auto fo = parse_f64(replace_d(parts[0]));
  if (!fo) return std::nullopt;
  double first = *fo;
  if (!std::isfinite(first)) return std::nullopt;
  double value;
  if (parts.size() == 1) {
    value = first * ((ra && hours) ? 15.0 : 1.0);
  } else {
    if (parts.size() < 2 || parts.size() > 3) return std::nullopt;
    auto mo = parse_f64(parts[1]);
    if (!mo) return std::nullopt;
    double sec = 0.0;
    if (parts.size() > 2) {
      auto so = parse_f64(parts[2]);
      if (!so) return std::nullopt;
      sec = *so;
    }
    double m = *mo;
    if (!(m >= 0.0 && m < 60.0) || !(sec >= 0.0 && sec < 60.0)) return std::nullopt;
    value = signum(first) * (std::fabs(first) + m / 60.0 + sec / 3600.0) * (ra ? 15.0 : 1.0);
  }
  double limit = ra ? 360.0 : 90.0;
  if (std::isfinite(value) && std::fabs(value) <= limit && (!ra || value >= 0.0)) return value;
  return std::nullopt;
}

std::optional<double> unit_scale(const FitsHeader& h, int axis) {
  auto v = get(h, "CUNIT" + std::to_string(axis));
  std::string u = v ? ascii_lower(trim(*v)) : "deg";
  if (u == "deg" || u == "degree" || u == "degrees") return 1.0;
  if (u == "rad") return 180.0 / PI;
  if (u == "arcmin") return 1.0 / 60.0;
  if (u == "arcsec") return 1.0 / 3600.0;
  return std::nullopt;
}

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

bool wcs_axes(const FitsHeader& h) {
  auto a = get(h, "CTYPE1"), b = get(h, "CTYPE2");
  if (!a && !b) return true;
  if (a && b) return starts_with(*a, "RA---") && starts_with(*b, "DEC--");
  return false;
}

std::optional<std::array<std::array<double, 2>, 2>> matrix(const FitsHeader& h) {
  if (!wcs_axes(h)) return std::nullopt;
  auto u1 = unit_scale(h, 1);
  if (!u1) return std::nullopt;
  auto u2 = unit_scale(h, 2);
  if (!u2) return std::nullopt;
  double units[2] = {*u1, *u2};
  std::array<std::array<double, 2>, 2> cd{};
  bool has_cd = false;
  for (const char* k : {"CD1_1", "CD1_2", "CD2_1", "CD2_2"})
    if (h.count(k)) has_cd = true;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double value;
      if (has_cd) {
        std::string k = "CD" + std::to_string(i + 1) + "_" + std::to_string(j + 1);
        if (h.count(k)) {
          auto n = number(h, k);
          if (!n) return std::nullopt;
          value = *n;
        } else {
          value = 0.0;
        }
      } else {
        std::string k = "PC" + std::to_string(i + 1) + "_" + std::to_string(j + 1);
        double pc;
        if (h.count(k)) {
          auto n = number(h, k);
          if (!n) return std::nullopt;
          pc = *n;
        } else {
          pc = i == j ? 1.0 : 0.0;
        }
        auto cdelt = number(h, "CDELT" + std::to_string(i + 1));
        if (!cdelt) return std::nullopt;
        value = *cdelt * pc;
      }
      cd[size_t(i)][size_t(j)] = value * units[i];
    }
  }
  double det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
  if (std::isfinite(det) && det != 0.0) return cd;
  return std::nullopt;
}

}  // namespace

std::optional<SolveHint> hint_from_fits(const FitsHeader& h, uint32_t width, uint32_t height) {
  auto cd = matrix(h);
  auto wcs_position = [&]() -> std::optional<Vec2> {
    if (!wcs_axes(h)) return std::nullopt;
    auto n1 = number(h, "CRVAL1");
    if (!n1) return std::nullopt;
    auto u1 = unit_scale(h, 1);
    if (!u1) return std::nullopt;
    auto n2 = number(h, "CRVAL2");
    if (!n2) return std::nullopt;
    auto u2 = unit_scale(h, 2);
    if (!u2) return std::nullopt;
    double ra = *n1 * *u1, dec = *n2 * *u2;
    SolveHint probe;
    probe.position = Vec2{ra, dec};
    auto sp = probe.sanitized();
    if (!sp || !sp->position) return std::nullopt;
    Vec2 position = *sp->position;
    auto ct = get(h, "CTYPE1");
    if (ct && (*ct == "RA---TAN" || *ct == "RA---TAN-SIP") && cd) {
      auto x = number(h, "CRPIX1"), y = number(h, "CRPIX2");
      if (x && y) {
        Wcs w;
        w.ra = ra;
        w.dec = dec;
        w.crpix = {*x - 1.0, *y - 1.0};
        w.cd = *cd;
        w.proj = Proj::Tan;
        Vec2 s = w.pixel_to_sky((double(width) - 1.0) / 2.0, (double(height) - 1.0) / 2.0);
        SolveHint ph;
        ph.position = s;
        auto sp2 = ph.sanitized();
        if (!sp2) return std::nullopt;
        return sp2->position;
      }
    }
    return position;
  }();
  std::optional<Vec2> position = wcs_position;
  if (!position) {
    struct Triple {
      const char *r, *d;
      bool hours;
    };
    for (const Triple& t : {Triple{"OBJCTRA", "OBJCTDEC", true}, Triple{"RA", "DEC", false}, Triple{"TELRA", "TELDEC", true}}) {
      auto r = get(h, t.r);
      if (!r) continue;
      auto rv = coordinate(*r, true, t.hours);
      if (!rv) continue;
      auto d = get(h, t.d);
      if (!d) continue;
      auto dv = coordinate(*d, false, false);
      if (!dv) continue;
      position = Vec2{*rv, *dv};
      break;
    }
  }
  auto positive = [&](const char* k) -> std::optional<double> {
    auto v = number(h, k);
    if (v && *v > 0.0) return v;
    return std::nullopt;
  };
  auto sampling = [&](const char* k) -> std::optional<double> {
    auto v = positive(k);
    if (v && *v <= 3600.0) return v;
    return std::nullopt;
  };
  std::optional<double> scale;
  if (cd) {
    double s = fmax_r(std::hypot((*cd)[0][0], (*cd)[1][0]), std::hypot((*cd)[0][1], (*cd)[1][1])) * 3600.0;
    if (std::isfinite(s) && s > 0.0 && s <= 3600.0) scale = s;
  }
  if (!scale) scale = sampling("PIXSCALE");
  if (!scale) scale = sampling("SECPIX");
  if (!scale) scale = sampling("SCALE");
  if (!scale) {
    auto pixel = positive("XPIXSZ");
    if (!pixel) pixel = positive("YPIXSZ");
    if (pixel) {
      auto fl = positive("FOCALLEN");
      if (fl) scale = 206.26480624709636 * *pixel / *fl;
    }
  }
  SolveHint out;
  out.position = position;
  out.scale_arcsec = scale;
  return out.for_frame(width, height);
}

}  // namespace skyshot
