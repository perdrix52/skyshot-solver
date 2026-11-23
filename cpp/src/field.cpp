#include "skyshot/field.hpp"

#include <cstdio>

#include "skyshot/compat.hpp"
#include "skyshot/geometry.hpp"

namespace skyshot::api {

Json FieldSummary::to_json() const {
  Json o = Json::object();
  o.set("ra_deg", ra_deg).set("dec_deg", dec_deg).set("ra_hms", ra_hms).set("dec_dms", dec_dms);
  o.set("galactic_l_deg", galactic_l_deg).set("galactic_b_deg", galactic_b_deg).set("width_deg", width_deg).set("height_deg", height_deg);
  o.set("scale_arcsec", scale_arcsec).set("up_pa_deg", up_pa_deg).set("mirrored", mirrored);
  o.set("crota2_deg", crota2_deg).set("cdelt1_deg", cdelt1_deg).set("cdelt2_deg", cdelt2_deg);
  Json c = Json::array();
  for (auto& p : corners) c.push(Json::from_arr(p));
  o.set("corners", std::move(c));
  o.set("pole", pole ? Json(*pole) : Json());
  o.set("sip_order", sip_order);
  return o;
}

FieldSummary field_summary(const Wcs& wcs, uint32_t width, uint32_t height) { return field_summary_in(wcs, width, height, std::nullopt); }

FieldSummary field_summary_in(const Wcs& wcs, uint32_t width, uint32_t height, const std::optional<std::array<double, 4>>& sky) {
  double w = double(width), h = double(height);
  Vec2 c = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  Vec2 centre = wcs.pixel_to_sky(c[0], c[1]);
  auto step = [&](double dx, double dy) {
    Vec2 s = wcs.pixel_to_sky(c[0] + dx, c[1] + dy);
    return geometry::project(s[0], s[1], centre).value_or(Vec2{0.0, 0.0});
  };
  Vec2 jx = step(1.0, 0.0), jy = step(0.0, 1.0);
  Vec2 up = step(0.0, -1.0);
  double x0 = -0.5, y0 = -0.5, x1 = w - 0.5, y1 = h - 0.5;
  std::array<Vec2, 4> corners;
  const Vec2 cp[4] = {{x0, y0}, {x1, y0}, {x1, y1}, {x0, y1}};
  for (size_t i = 0; i < 4; ++i) corners[i] = wcs.pixel_to_sky(cp[i][0], cp[i][1]);
  std::optional<std::string> pole;
  for (auto [dec, name] : {std::pair<double, const char*>{90.0, "N"}, {-90.0, "S"}}) {
    auto p = wcs.sky_to_pixel(0.0, dec);
    if (p && (*p)[0] >= x0 && (*p)[0] <= x1 && (*p)[1] >= y0 && (*p)[1] <= y1) {
      pole = name;
      break;
    }
  }
  auto cd = compat::cdelt_crota(wcs.cd);
  auto [l, b] = galactic(centre[0], centre[1]);
  FieldSummary f;
  f.ra_deg = centre[0];
  f.dec_deg = centre[1];
  f.ra_hms = format_hms(centre[0]);
  f.dec_dms = format_dms(centre[1]);
  f.galactic_l_deg = l;
  f.galactic_b_deg = b;
  if (sky) {
    const auto& e = *sky;
    double y = (e[1] + e[3] - 1.0) / 2.0;
    f.width_deg = wcs.arc_deg({e[0] - 0.5, y}, {e[2] - 0.5, y});
    double x = (e[0] + e[2] - 1.0) / 2.0;
    f.height_deg = wcs.arc_deg({x, e[1] - 0.5}, {x, e[3] - 0.5});
  } else {
    f.width_deg = wcs.arc_deg({x0, c[1]}, {x1, c[1]});
    f.height_deg = wcs.arc_deg({c[0], y0}, {c[0], y1});
  }
  f.scale_arcsec = std::sqrt(std::fabs(jx[0] * jy[1] - jy[0] * jx[1])) * 3600.0;
  f.up_pa_deg = rem_euclid(to_degrees(std::atan2(up[0], up[1])), 360.0);
  f.mirrored = jx[0] * jy[1] - jy[0] * jx[1] < 0.0;
  f.crota2_deg = cd.crota2;
  f.cdelt1_deg = cd.cdelt1;
  f.cdelt2_deg = cd.cdelt2;
  f.corners = corners;
  f.pole = pole;
  f.sip_order = wcs.sip ? wcs.sip->order : 0;
  return f;
}

std::pair<double, double> galactic(double ra, double dec) {
  static const double M[3][3] = {{-0.0548755604162154, -0.873437090234885, -0.4838350155487132},
                                 {0.4941094278755837, -0.4448296299600112, 0.7469822444972189},
                                 {-0.8676661490190047, -0.1980763734312015, 0.4559837761750669}};
  Vec3 u = geometry::unit(ra, dec);
  double g[3];
  for (int i = 0; i < 3; ++i) g[i] = M[i][0] * u[0] + M[i][1] * u[1] + M[i][2] * u[2];
  return {rem_euclid(to_degrees(std::atan2(g[1], g[0])), 360.0), to_degrees(std::asin(clampd(g[2], -1.0, 1.0)))};
}

std::string format_hms(double ra_deg) {
  const int64_t DAY = 24 * 3600 * 100;
  int64_t cs = rem_euclid_i(sat_cast<int64_t>(std::round(ra_deg / 15.0 * 3600.0 * 100.0)), DAY);
  char buf[64];
  std::snprintf(buf, sizeof buf, "%02lldh %02lldm %05.2fs", (long long)(cs / 360000), (long long)(cs / 6000 % 60), double(cs % 6000) / 100.0);
  return buf;
}

std::string format_dms(double dec_deg) {
  int64_t ds = sat_cast<int64_t>(std::round(std::fabs(dec_deg) * 36000.0));
  char buf[96];
  std::snprintf(buf, sizeof buf, "%s%02lld\xC2\xB0 %02lld\xE2\x80\xB2 %04.1f\xE2\x80\xB3", (dec_deg < 0.0 && ds > 0) ? "\xE2\x88\x92" : "+",
                (long long)(ds / 36000), (long long)(ds / 600 % 60), double(ds % 600) / 10.0);
  return buf;
}

}  // namespace skyshot::api
