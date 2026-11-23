#include "skyshot/types.hpp"

namespace skyshot {

// ---------- JSON helpers ----------
namespace {
double req_f64(const Json& o, const char* k) {
  const Json* v = o.get(k);
  if (!v) throw Error(std::string("missing field `") + k + "`");
  if (!v->is_number()) throw Error(std::string("invalid type for field `") + k + "`, expected a number");
  return v->as_f64();
}
double opt_f64(const Json& o, const char* k, double dflt) {
  const Json* v = o.get(k);
  if (!v) return dflt;
  if (!v->is_number()) throw Error(std::string("invalid type for field `") + k + "`, expected a number");
  return v->as_f64();
}
uint32_t req_u32(const Json& o, const char* k) {
  const Json* v = o.get(k);
  if (!v) throw Error(std::string("missing field `") + k + "`");
  auto u = v->as_u64();
  if (!u || *u > UINT32_MAX) throw Error(std::string("invalid value for field `") + k + "`, expected u32");
  return uint32_t(*u);
}
Vec2 vec2_from(const Json& j, const char* what) {
  if (!j.is_array() || j.items().size() != 2 || !j.items()[0].is_number() || !j.items()[1].is_number())
    throw Error(std::string("invalid ") + what);
  return {j.items()[0].as_f64(), j.items()[1].as_f64()};
}
Json vec2_json(const Vec2& v) { return Json::from_arr(v); }
}  // namespace

// ---------- Star / StarField ----------
Json Star::to_json() const {
  Json o = Json::object();
  o.set("x", x).set("y", y).set("flux", flux).set("sigma", sigma).set("snr", snr).set("fwhm", fwhm);
  o.set("elongation", elongation).set("saturated", saturated).set("peak", peak).set("pa", pa).set("flags", flags);
  return o;
}
Star Star::from_json(const Json& j) {
  if (!j.is_object()) throw Error("invalid type: expected a star object");
  Star s;
  s.x = req_f64(j, "x");
  s.y = req_f64(j, "y");
  s.flux = req_f64(j, "flux");
  s.sigma = opt_f64(j, "sigma", 0.25);
  s.snr = opt_f64(j, "snr", 20.0);
  s.fwhm = opt_f64(j, "fwhm", 3.0);
  s.elongation = opt_f64(j, "elongation", 1.0);
  if (const Json* v = j.get("saturated")) {
    if (!v->is_bool()) throw Error("invalid type for field `saturated`, expected a boolean");
    s.saturated = v->as_bool();
  }
  s.peak = opt_f64(j, "peak", 0.0);
  s.pa = opt_f64(j, "pa", 0.0);
  if (const Json* v = j.get("flags")) {
    auto u = v->as_u64();
    if (!u || *u > UINT32_MAX) throw Error("invalid value for field `flags`, expected u32");
    s.flags = uint32_t(*u);
  }
  return s;
}

Json StarField::to_json() const {
  Json o = Json::object();
  o.set("width", width).set("height", height);
  Json arr = Json::array();
  for (const auto& s : stars) arr.push(s.to_json());
  o.set("stars", std::move(arr));
  o.set("epoch", epoch ? Json(*epoch) : Json());
  return o;
}
StarField StarField::from_json(const Json& j) {
  if (!j.is_object()) throw Error("invalid type: expected a StarField object");
  StarField f;
  f.width = req_u32(j, "width");
  f.height = req_u32(j, "height");
  const Json* st = j.get("stars");
  if (!st) throw Error("missing field `stars`");
  if (!st->is_array()) throw Error("invalid type for field `stars`, expected a sequence");
  for (const auto& s : st->items()) f.stars.push_back(Star::from_json(s));
  if (const Json* e = j.get("epoch")) {
    if (!e->is_null()) {
      if (!e->is_number()) throw Error("invalid type for field `epoch`");
      f.epoch = e->as_f64();
    }
  }
  return f;
}

// ---------- Sip ----------
std::vector<std::pair<uint32_t, uint32_t>> Sip::terms(uint32_t order) {
  std::vector<std::pair<uint32_t, uint32_t>> t;
  for (uint32_t n = 2; n <= order; ++n)
    for (int64_t p = int64_t(n); p >= 0; --p) t.emplace_back(uint32_t(p), n - uint32_t(p));
  return t;
}

static void powers(double x, size_t n, double out[16]) {
  out[0] = 1.0;
  for (size_t i = 1; i < 16; ++i) out[i] = 0.0;
  for (size_t i = 1; i <= std::min<size_t>(n, 15); ++i) out[i] = out[i - 1] * x;
}

std::pair<double, double> Sip::eval(double u, double v) const {
  size_t n = order;
  double pu[16], pv[16];
  powers(u, n, pu);
  powers(v, n, pv);
  double fa = 0.0, fb = 0.0;
  size_t k = 0;
  for (size_t nn = 2; nn <= n; ++nn) {
    for (int64_t p = int64_t(nn); p >= 0; --p) {
      double m = pu[p] * pv[nn - size_t(p)];
      fa += a[k] * m;
      fb += b[k] * m;
      ++k;
    }
  }
  return {fa, fb};
}

Sip::EvalJac Sip::eval_jacobian(double u, double v) const {
  size_t n = order;
  double pu[16], pv[16];
  powers(u, n, pu);
  powers(v, n, pv);
  EvalJac r{0.0, 0.0, {{1.0, 0.0}, {0.0, 1.0}}};
  size_t k = 0;
  for (size_t nn = 2; nn <= n; ++nn) {
    for (int64_t pp = int64_t(nn); pp >= 0; --pp) {
      size_t p = size_t(pp), q = nn - p;
      double m = pu[p] * pv[q];
      r.fa += a[k] * m;
      r.fb += b[k] * m;
      double du = p > 0 ? double(p) * pu[p - 1] * pv[q] : 0.0;
      double dv = q > 0 ? double(q) * pu[p] * pv[q - 1] : 0.0;
      r.j[0][0] += a[k] * du;
      r.j[0][1] += a[k] * dv;
      r.j[1][0] += b[k] * du;
      r.j[1][1] += b[k] * dv;
      ++k;
    }
  }
  return r;
}

std::array<std::array<double, 2>, 2> Sip::jacobian(double u, double v) const {
  auto r = eval_jacobian(u, v);
  return {{{r.j[0][0], r.j[0][1]}, {r.j[1][0], r.j[1][1]}}};
}

Json Sip::to_json() const {
  Json o = Json::object();
  o.set("order", order).set("a", Json::from_vec(a)).set("b", Json::from_vec(b));
  return o;
}
Sip Sip::from_json(const Json& j) {
  Sip s;
  s.order = req_u32(j, "order");
  for (const char* k : {"a", "b"}) {
    const Json* v = j.get(k);
    if (!v || !v->is_array()) throw Error(std::string("invalid sip field `") + k + "`");
    std::vector<double>& dst = k[0] == 'a' ? s.a : s.b;
    for (const auto& x : v->items()) {
      if (!x.is_number()) throw Error("invalid sip coefficient");
      dst.push_back(x.as_f64());
    }
  }
  size_t need = Sip::terms(s.order).size();
  if (s.a.size() < need || s.b.size() < need) throw Error("sip coefficient count does not match its order");
  return s;
}

const char* proj_code(Proj p) {
  switch (p) {
    case Proj::Tan: return "TAN";
    case Proj::Arc: return "ARC";
    case Proj::Stg: return "STG";
    case Proj::Car: return "CAR";
    case Proj::Mer: return "MER";
    case Proj::Cyp: return "CYP";
  }
  return "TAN";
}

Json Refraction::to_json() const {
  Json o = Json::object();
  o.set("zenith", vec2_json(zenith)).set("scale", scale).set("rms_px", rms_px);
  o.set("rms_refracted_px", rms_refracted_px).set("min_alt_deg", min_alt_deg).set("n_pairs", uint64_t(n_pairs));
  return o;
}

// ---------- PixelGrid ----------
std::optional<std::string> PixelGrid::validate() const {
  bool ok = cols >= 2 && rows >= 2 && cols <= 129 && rows <= 129 && offsets.size() == cols * rows;
  for (double x : origin) ok = ok && std::isfinite(x);
  for (double x : step) ok = ok && std::isfinite(x) && x > 0.0;
  if (ok)
    for (const auto& o : offsets)
      for (double x : o) ok = ok && std::isfinite(x) && std::fabs(x) <= 256.0;
  if (!ok) return std::string("invalid local pixel grid");
  for (size_t y = 0; y + 1 < rows; ++y) {
    for (size_t x = 0; x + 1 < cols; ++x) {
      const Vec2& a = offsets[y * cols + x];
      const Vec2& b = offsets[y * cols + x + 1];
      const Vec2& c = offsets[(y + 1) * cols + x];
      const Vec2& d = offsets[(y + 1) * cols + x + 1];
      for (size_t k = 0; k < 2; ++k) {
        double dx = fmax_r(std::fabs(b[k] - a[k]), std::fabs(d[k] - c[k])) / step[0];
        double dy = fmax_r(std::fabs(c[k] - a[k]), std::fabs(d[k] - b[k])) / step[1];
        if (dx + dy >= 0.75) return std::string("folding local pixel grid");
      }
    }
  }
  return std::nullopt;
}

Vec2 PixelGrid::offset(Vec2 p) const {
  double u = clampd((p[0] - origin[0]) / step[0], 0., double(cols - 1));
  double v = clampd((p[1] - origin[1]) / step[1], 0., double(rows - 1));
  size_t x = std::min<size_t>(sat_cast<uint64_t>(std::floor(u)), cols - 2);
  size_t y = std::min<size_t>(sat_cast<uint64_t>(std::floor(v)), rows - 2);
  u = u - double(x);
  v = v - double(y);
  Vec2 out;
  for (size_t k = 0; k < 2; ++k) {
    out[k] = (1. - v) * ((1. - u) * offsets[y * cols + x][k] + u * offsets[y * cols + x + 1][k]) +
             v * ((1. - u) * offsets[(y + 1) * cols + x][k] + u * offsets[(y + 1) * cols + x + 1][k]);
  }
  return out;
}
Vec2 PixelGrid::forward(Vec2 p) const {
  Vec2 d = offset(p);
  return {p[0] + d[0], p[1] + d[1]};
}
std::optional<Vec2> PixelGrid::inverse(Vec2 p) const {
  if (!std::isfinite(p[0]) || !std::isfinite(p[1])) return std::nullopt;
  Vec2 q = p;
  for (int i = 0; i < 80; ++i) {
    Vec2 d = offset(q);
    Vec2 next = {p[0] - d[0], p[1] - d[1]};
    if (fmax_r(std::fabs(next[0] - q[0]), std::fabs(next[1] - q[1])) < 1e-7) return next;
    q = next;
  }
  return std::nullopt;
}

Json PixelGrid::to_json() const {
  Json o = Json::object();
  o.set("cols", uint64_t(cols)).set("rows", uint64_t(rows)).set("origin", vec2_json(origin)).set("step", vec2_json(step));
  Json off = Json::array();
  for (const auto& v : offsets) off.push(vec2_json(v));
  o.set("offsets", std::move(off));
  return o;
}
PixelGrid PixelGrid::from_json(const Json& j) {
  PixelGrid g;
  auto cols = j.get("cols") ? j.get("cols")->as_u64() : std::nullopt;
  auto rows = j.get("rows") ? j.get("rows")->as_u64() : std::nullopt;
  if (!cols || !rows || !j.get("origin") || !j.get("step") || !j.get("offsets")) throw Error("invalid local pixel grid");
  g.cols = size_t(*cols);
  g.rows = size_t(*rows);
  g.origin = vec2_from(*j.get("origin"), "pixel grid origin");
  g.step = vec2_from(*j.get("step"), "pixel grid step");
  for (const auto& o : j.get("offsets")->items()) g.offsets.push_back(vec2_from(o, "pixel grid offset"));
  if (auto e = g.validate()) throw Error(*e);
  return g;
}

// ---------- Wcs ----------
Json Wcs::to_json() const {
  Json o = Json::object();
  o.set("ra", ra).set("dec", dec).set("crpix", vec2_json(crpix));
  Json cdj = Json::array();
  cdj.push(vec2_json({cd[0][0], cd[0][1]})).push(vec2_json({cd[1][0], cd[1][1]}));
  o.set("cd", std::move(cdj));
  o.set("sip", sip ? sip->to_json() : Json());
  if (!is_tan(proj)) o.set("proj", proj_code(proj));
  if (native_pole) o.set("native_pole", vec2_json(*native_pole));
  if (local_grid) o.set("local_grid", local_grid->to_json());
  return o;
}

Wcs Wcs::from_json(const Json& j) {
  Wcs w;
  w.ra = req_f64(j, "ra");
  w.dec = req_f64(j, "dec");
  if (!j.get("crpix") || !j.get("cd")) throw Error("missing field in wcs");
  w.crpix = vec2_from(*j.get("crpix"), "crpix");
  const Json& cdj = *j.get("cd");
  if (!cdj.is_array() || cdj.items().size() != 2) throw Error("invalid cd");
  Vec2 r0 = vec2_from(cdj.items()[0], "cd"), r1 = vec2_from(cdj.items()[1], "cd");
  w.cd = {{{r0[0], r0[1]}, {r1[0], r1[1]}}};
  if (const Json* s = j.get("sip"); s && !s->is_null()) w.sip = Sip::from_json(*s);
  if (const Json* p = j.get("proj"); p && p->is_string()) {
    const std::string& c = p->as_str();
    if (c == "TAN") w.proj = Proj::Tan;
    else if (c == "ARC") w.proj = Proj::Arc;
    else if (c == "STG") w.proj = Proj::Stg;
    else if (c == "CAR") w.proj = Proj::Car;
    else if (c == "MER") w.proj = Proj::Mer;
    else if (c == "CYP") w.proj = Proj::Cyp;
    else throw Error("unknown projection " + c);
  }
  if (const Json* n = j.get("native_pole"); n && !n->is_null()) w.native_pole = vec2_from(*n, "native_pole");
  if (const Json* g = j.get("local_grid"); g && !g->is_null()) w.local_grid = std::make_shared<PixelGrid>(PixelGrid::from_json(*g));
  return w;
}

}  // namespace skyshot
