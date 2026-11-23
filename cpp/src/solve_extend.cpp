// Window extension over the whole frame, and blind search on cylinder tiles.
// Port of the remaining parts of src/solve/wide.rs and src/solve/wide/panorama.rs.
#include <algorithm>

#include "skyshot/linalg.hpp"
#include "solve_internal.hpp"

namespace skyshot::solve {

using geometry::Projection;
using geometry::unit;
using sphere::angle;
using sphere::radec;
using Rect = std::array<double, 4>;

// ---------------------------------------------------------------------------------------------
// Solver methods of wide.rs
// ---------------------------------------------------------------------------------------------
uint32_t Solver::frame_level(const Wcs& wcs) const {
  double s = to_radians(wcs.scale_arcsec()) / 3600.0;
  double area = s * s * open_area;
  uint32_t mr = idx.has_deep() ? 8u : 7u;
  while (mr > 1 && area / (PI * powi(idx.level_radius_m(mr), 2)) > 1500.0) --mr;
  return mr;
}

std::vector<Vec2> Solver::cone_radec(const Vec3& c, double radius, uint32_t level) {
  double rm = idx.level_radius_m(level);
  std::vector<Vec2> out;
  if (rm < idx.r7 * 0.999 && idx.has_deep()) {
    for (auto& st : catalog_cone(c, radius, rm)) {
      auto [ra, dec] = radec(st.v);
      out.push_back({ra, dec});
    }
    return out;
  }
  std::vector<index::BrightStar> stars;
  idx.cone_stars(c, fmin_r(radius, PI), cfg.mag_limit.value_or(std::numeric_limits<float>::max()), rm * 0.999, stars);
  for (auto& st : stars) {
    auto [ra, dec] = radec(st.v);
    out.push_back({ra, dec});
  }
  return out;
}

std::optional<FrameCat> Solver::frame_catalog(const Wcs& wcs, uint32_t level) {
  Vec2 crpix = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  Vec2 c = wcs.pixel_to_sky(crpix[0], crpix[1]);
  Vec3 center = unit(c[0], c[1]);
  double fr = frame_radius(wcs, center, 0.0);
  if (!std::isfinite(fr)) return std::nullopt;
  double radius = fmin_r(1.5 * fr + 0.15, PI);
  FrameCat fc;
  fc.stars = cone_radec(center, radius, level);
  fc.level = level;
  fc.center = center;
  fc.radius = radius;
  return fc;
}

Pairs Solver::zone_stars(const FrameCat& fc, const Wcs& wcs, Vec2 pc, double r) const {
  Pairs out;
  out.reserve(fc.stars.size() / 2);
  auto projector = geometry::prepare(wcs);
  for (const Vec2& s : fc.stars) {
    auto px = projector.sky_to_pixel_fast(s[0], s[1]);
    if (px && std::hypot((*px)[0] - pc[0], (*px)[1] - pc[1]) <= r && visible(*px)) out.push_back({s[0], s[1], (*px)[0], (*px)[1]});
  }
  return out;
}

size_t Solver::inliers_in(const FrameCat& fc, const Wcs& wcs, Rect rect, double tol) const {
  Vec2 pc = {(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0};
  double r = 0.5 * std::hypot(rect[2] - rect[0], rect[3] - rect[1]);
  Pairs inside;
  for (auto& c : zone_stars(fc, wcs, pc, r))
    if (c[2] >= rect[0] && c[2] < rect[2] && c[3] >= rect[1] && c[3] < rect[3]) inside.push_back(c);
  size_t n_img = std::min(std::max<size_t>(inside.size() * 3, 50), pts.size());
  std::vector<uint32_t> ids(n_img);
  for (size_t i = 0; i < n_img; ++i) ids[i] = uint32_t(i);
  auto g = select::Grid2::build(pts, ids, w, h, 2.0);
  std::vector<std::pair<Vec2, Vec2>> proj;
  for (auto& c : inside) proj.push_back({{c[0], c[1]}, {c[2], c[3]}});
  return match_projected(pts, g, proj, tol).n;
}

std::optional<Grown> Solver::grow_stereographic(const Wcs& seed) {
  Wcs model = seed;
  Vec2 pc = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  double radius = 0.5 * std::hypot(w, h);
  std::optional<Grown> best;
  for (double tolerance : {12.0, 6.0, 3.0, 2.5}) {
    if (elapsed_ms() >= cfg.deadline_ms) break;
    auto fc = frame_catalog(model, frame_level(model));
    if (!fc) return std::nullopt;
    Pairs cat = zone_stars(*fc, model, pc, radius);
    size_t n_img = std::min(std::max<size_t>(cat.size() * 3, 50), pts.size());
    std::vector<uint32_t> ids(n_img);
    for (size_t i = 0; i < n_img; ++i) ids[i] = uint32_t(i);
    auto grid = select::Grid2::build(pts, ids, w, h, 2.0);
    std::vector<std::pair<Vec2, Vec2>> proj;
    for (auto& c : cat) proj.push_back({{c[0], c[1]}, {c[2], c[3]}});
    auto matches = match_projected(pts, grid, proj, tolerance);
    Pairs prs = pairs_of(matches.m);
    auto wts = weights(matches.m);
    auto fit = wcsfit::fit_stereographic(prs, wts);
    if (!fit) return std::nullopt;
    Pairs inliers;
    for (size_t i = 0; i < prs.size(); ++i)
      if (fit->keep[i]) inliers.push_back(prs[i]);
    double rms = wcsfit::rms(wcsfit::residuals(fit->wcs, inliers));
    model = std::move(fit->wcs);
    best = Grown{model, std::move(inliers), rms};
  }
  return best;
}

bool Solver::extension_evidence(const Wcs& wcs, Rect seed) {
  auto outside = [&](Vec2 p) { return p[0] < seed[0] || p[0] >= seed[2] || p[1] < seed[1] || p[1] >= seed[3]; };
  Vec2 pc = {(w - 1.) / 2., (h - 1.) / 2.};
  auto fc = frame_catalog(wcs, frame_level(wcs));
  if (!fc) return false;
  Pairs cat = zone_stars(*fc, wcs, pc, 0.5 * std::hypot(w, h));
  size_t n_fit = std::min(std::max<size_t>(cat.size() * 3, 50), pts.size());
  Pairs cat2;
  for (auto& p : cat)
    if (outside({p[2], p[3]})) cat2.push_back(p);
  std::vector<uint32_t> ids;
  for (size_t i = 0; i < std::min(n_fit, pts.size()); ++i)
    if (outside(pts[i])) ids.push_back(uint32_t(i));
  double tolerance = 2.5;
  auto grid = select::Grid2::build(pts, ids, w, h, 2.0);
  std::vector<std::pair<Vec2, Vec2>> proj;
  for (auto& c : cat2) proj.push_back({{c[0], c[1]}, {c[2], c[3]}});
  size_t k = match_projected(pts, grid, proj, tolerance).n;
  double blocked = cfg.mask ? cfg.mask->blocked_area(seed[0], seed[1], seed[2], seed[3]) : 0.0;
  double area = open_area - ((seed[2] - seed[0]) * (seed[3] - seed[1]) - blocked);
  if (area <= 0. || cat2.empty() || ids.empty()) return false;
  size_t n_img = 0;
  for (auto& p : pts)
    if (outside(p)) ++n_img;
  double lambda = double(cat2.size()) * double(n_img) * PI * powi(tolerance, 2) / area;
  double ll = poisson_log_odds(k, lambda);
  double threshold = cfg.min_log_odds + trials_ln();
  return k >= 24 && ll >= threshold;
}

double Solver::zone_angle(const Wcs& wcs, Vec2 pc, double r) const {
  Vec2 c = wcs.pixel_to_sky(pc[0], pc[1]);
  Vec3 cv = unit(c[0], c[1]);
  double ang = 0.0;
  for (int k = 0; k < 16; ++k) {
    double a = double(k) * TAU / 16.0;
    double x = clampd(pc[0] + r * std::cos(a), 0.0, w - 1.0);
    double y = clampd(pc[1] + r * std::sin(a), 0.0, h - 1.0);
    Vec2 sky = wcs.pixel_to_sky(x, y);
    ang = fmax_r(ang, angle(cv, unit(sky[0], sky[1])));
  }
  return ang;
}

std::optional<Grown> Solver::grow(const Wcs& w0, Rect win, double rms0, FrameCat& fc, const Projection& start) {
  Vec2 crpix = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  Vec2 pc = {win[0] + win[2] / 2.0, win[1] + win[3] / 2.0};
  double r_end = 0.0;
  for (Vec2 q : {Vec2{0.0, 0.0}, Vec2{w, 0.0}, Vec2{0.0, h}, Vec2{w, h}}) r_end = fmax_r(r_end, std::hypot(q[0] - pc[0], q[1] - pc[1]));
  double r = 0.5 * std::hypot(win[2], win[3]);
  double r_prev = 0.0;
  Wcs model = w0;
  double rms = std::isfinite(rms0) ? clampd(rms0, 0.3, 3.0) : 1.0;
  std::optional<Grown> best;
  Pairs last_pairs;
  std::vector<double> last_w;
  Projection proj = start.fixed({w0.ra, w0.dec});
  for (;;) {
    if (elapsed_ms() >= cfg.deadline_ms) break;
    bool full = r >= r_end;
    if (frame_radius(model, fc.center, 0.0) > 0.95 * fc.radius && fc.radius < PI) {
      auto nf = frame_catalog(model, fc.level);
      if (!nf) return std::nullopt;
      fc = std::move(*nf);
    }
    Pairs cat = zone_stars(fc, model, pc, fmin_r(r, r_end));
    size_t n_img = std::min(std::max<size_t>(cat.size() * 3, 50), pts.size());
    std::vector<uint32_t> ids(n_img);
    for (size_t i = 0; i < n_img; ++i) ids[i] = uint32_t(i);
    auto g = select::Grid2::build(pts, ids, w, h, 2.0);
    double sep = std::sqrt(open_area / double(std::max<size_t>(cat.size(), 1)));
    double rcap = clampd(0.3 * sep, 1.5, 25.0);
    double tol = clampd(4.0 * rms + 0.01 * (r - r_prev), 1.5, rcap);
    std::vector<std::pair<Vec2, Vec2>> pr;
    for (auto& c : cat) pr.push_back({{c[0], c[1]}, {c[2], c[3]}});
    auto mres = match_projected(pts, g, pr, tol);
    Pairs prs = pairs_of(mres.m);
    auto wts = weights(mres.m);
    if (prs.size() < 12) break;
    uint32_t order;
    size_t np = prs.size();
    if (np <= 29) order = 1;
    else if (np <= 59) order = 2;
    else if (np <= 99) order = 3;
    else order = 4;
    uint32_t radial = np >= 120 ? 7 : 5;
    Vec2 tangent = model.pixel_to_sky(crpix[0], crpix[1]);
    bool wide = zone_angle(model, pc, fmin_r(r, r_end)) > 0.35;
    std::vector<std::optional<wcsfit::RobustFit>> fits;
    if (proj.proj == Proj::Tan) fits.push_back(wcsfit::robust_fit_wmp(prs, wts, crpix, tangent, order, radial, Projection(Proj::Tan)));
    if ((wide || proj.proj == Proj::Arc) && cfg.allow_arc && !is_cylindrical(start.proj) && start.proj != Proj::Stg)
      fits.push_back(wcsfit::robust_fit_wmp(prs, wts, crpix, tangent, order, radial, Projection(Proj::Arc)));
    if (is_cylindrical(start.proj) || start.proj == Proj::Stg) fits.push_back(wcsfit::robust_fit_wmp(prs, wts, crpix, tangent, order, 0, proj));
    struct Chosen {
      Wcs fit;
      std::vector<bool> keep;
      double e_cmp;
    };
    std::optional<Chosen> chosen;
    for (auto& f : fits) {
      if (!f) continue;
      Pairs inl;
      for (size_t i = 0; i < prs.size(); ++i)
        if (f->keep[i]) inl.push_back(prs[i]);
      double e = wcsfit::rms(wcsfit::residuals(f->wcs, inl));
      double e_cmp = f->wcs.proj == Proj::Arc ? e * 1.1 : e;
      if (std::isfinite(e) && (!chosen || e_cmp < chosen->e_cmp)) chosen = Chosen{std::move(f->wcs), std::move(f->keep), e_cmp};
    }
    if (!chosen) break;
    Pairs inl;
    for (size_t i = 0; i < prs.size(); ++i)
      if (chosen->keep[i]) inl.push_back(prs[i]);
    rms = fmax_r(wcsfit::rms(wcsfit::residuals(chosen->fit, inl)), 0.05);
    proj = Projection(chosen->fit.proj, chosen->fit.native_pole);
    model = std::move(chosen->fit);
    best = Grown{model, std::move(inl), rms};
    last_pairs = std::move(prs);
    last_w = std::move(wts);
    if (full) break;
    r_prev = r;
    r *= 1.35;
  }
  if (last_pairs.size() >= 40) {
    Vec2 tangent = model.pixel_to_sky(crpix[0], crpix[1]);
    uint32_t max_order = last_pairs.size() >= 160 ? 5 : 4;
    auto sel = wcsfit::select_model_wp(last_pairs, last_w, crpix, tangent, max_order, is_cylindrical(model.proj) ? 0 : 7,
                                       Projection(model.proj, model.native_pole));
    if (sel) {
      Pairs inl;
      for (size_t i = 0; i < last_pairs.size(); ++i)
        if (sel->keep[i]) inl.push_back(last_pairs[i]);
      double e = wcsfit::rms(wcsfit::residuals(sel->wcs, inl));
      if (std::isfinite(e) && inl.size() * 10 >= (best ? best->pairs.size() : 0) * 9) best = Grown{std::move(sel->wcs), std::move(inl), e};
    }
  }
  return best;
}

// ---------------------------------------------------------------------------------------------
// extend()
// ---------------------------------------------------------------------------------------------
namespace wide {

std::optional<Solution> extend(const Index& idx, const StarField& field, const SolveConfig& cfg, const Solution& ws,
                               std::array<double, 4> win) {
  auto t = Clock::now();
  auto ctxp = make_context(idx, field, cfg, t);
  Solver& ctx = *ctxp;
  if (!ws.wcs) return std::nullopt;
  Wcs w0 = *ws.wcs;
  w0.crpix = {w0.crpix[0] + win[0], w0.crpix[1] + win[1]};
  auto fco = ctx.frame_catalog(w0, ctx.frame_level(w0));
  if (!fco) return std::nullopt;
  FrameCat fc = std::move(*fco);
  Rect rect = {win[0], win[1], win[0] + win[2], win[1] + win[3]};
  size_t n_w0 = ctx.inliers_in(fc, w0, rect, 2.5);
  auto accept = [&](std::optional<Grown> g, const FrameCat& fcat) -> std::optional<Grown> {
    if (!g) return std::nullopt;
    if (double(ctx.inliers_in(fcat, g->wcs, rect, 2.5)) >= 0.9 * double(n_w0) && grown_holds(ctx, g->wcs, g->pairs, g->rms)) return g;
    return std::nullopt;
  };
  auto grown = accept(ctx.grow(w0, win, ws.rms_px, fc, Projection(Proj::Tan)), fc);
  if (!grown && cfg.allow_arc) grown = accept(ctx.grow(w0, win, ws.rms_px, fc, Projection(Proj::Arc)), fc);
  Rect open_bbox = cfg.mask ? cfg.mask->open_bbox(ctx.w, ctx.h) : Rect{0., 0., ctx.w, ctx.h};
  double bw = open_bbox[2] - open_bbox[0], bh = open_bbox[3] - open_bbox[1];
  bool elongated = fmax_r(bw, bh) > 1.5 * fmin_r(bw, bh);
  if ((!grown || elongated) && cfg.allow_panorama && ctx.elapsed_ms() < cfg.deadline_ms) {
    Pairs seed_pairs;
    for (const auto& m : ws.matches) seed_pairs.push_back({m.x + win[0], m.y + win[1], m.ra, m.dec});
    if (auto seed = wcsfit::fit_stereographic(seed_pairs, {})) {
      if (auto fit = ctx.grow_stereographic(seed->wcs)) {
        size_t retained = 0;
        for (auto& p : seed_pairs) {
          auto q = fit->wcs.sky_to_pixel(p[2], p[3]);
          if (q && std::hypot((*q)[0] - p[0], (*q)[1] - p[1]) <= 2.5) ++retained;
        }
        if (double(retained) >= 0.9 * double(seed_pairs.size()) && grown_holds(ctx, fit->wcs, fit->pairs, fit->rms) &&
            ctx.extension_evidence(fit->wcs, rect)) {
          size_t n = ctx.inliers_in(fc, fit->wcs, open_bbox, 2.5);
          size_t old = grown ? ctx.inliers_in(fc, grown->wcs, open_bbox, 2.5) : 0;
          if (!grown || double(n) > 1.1 * double(old)) grown = std::move(*fit);
        }
      }
    }
    Pairs seed_cat = ctx.zone_stars(fc, w0, {win[0] + win[2] / 2.0, win[1] + win[3] / 2.0}, 0.5 * std::hypot(win[2], win[3]));
    std::vector<uint32_t> ids(ctx.pts.size());
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = uint32_t(i);
    auto grid = select::Grid2::build(ctx.pts, ids, ctx.w, ctx.h, 2.0);
    std::vector<std::pair<Vec2, Vec2>> proj;
    for (auto& c : seed_cat) proj.push_back({{c[0], c[1]}, {c[2], c[3]}});
    auto matches = match_projected(ctx.pts, grid, proj, 2.5);
    Pairs prs = ctx.pairs_of(matches.m);
    Vec2 cp = {win[0] + win[2] / 2.0, win[1] + win[3] / 2.0};
    Vec2 tangent = w0.pixel_to_sky(cp[0], cp[1]);
    auto candidates = wcsfit::panorama_candidates(prs, cp, tangent);
    candidates.push_back(Projection(Proj::Stg));
    for (const Projection& candidate : candidates) {
      if (ctx.elapsed_ms() >= cfg.deadline_ms) break;
      auto sf = wcsfit::robust_fit_wmp(prs, {}, cp, tangent, 1, 0, candidate);
      Wcs seed = sf ? sf->wcs : w0;
      if (auto fit = accept(ctx.grow(seed, win, ws.rms_px, fc, candidate), fc)) {
        size_t n = ctx.inliers_in(fc, fit->wcs, open_bbox, 2.5);
        size_t support = grown ? ctx.inliers_in(fc, grown->wcs, open_bbox, 2.5) : 0;
        if (!grown || double(n) > 1.1 * double(support)) grown = std::move(fit);
      }
    }
  }
  if (!grown) return std::nullopt;
  return finish(ctx, field, cfg, ws, win, *grown);
}

// ---------------------------------------------------------------------------------------------
// Blind identification on tangent tiles of a cylinder (panorama.rs)
// ---------------------------------------------------------------------------------------------
namespace {
constexpr double TILE = 512.;
constexpr double FOCAL = 443.4050067376326;

struct Tile {
  double focal, horizon;
  Vec2 center;
  Vec3 c, e, n;
  Tile(double width, double horizon_, double fov, Vec2 center_) : horizon(horizon_), center(center_) {
    focal = width / to_radians(fov);
    double a = (horizon - center[1]) / focal;
    double s = std::sin(a), co = std::cos(a);
    c = {co, 0., s};
    e = {0., 1., 0.};
    n = {-s, 0., co};
  }
  std::optional<Vec3> ray(Vec2 p) const {
    double lon = (p[0] - center[0]) / focal;
    double lat = (horizon - p[1]) / focal;
    if (std::fabs(lat) >= PI / 2. || std::fabs(lon) >= PI) return std::nullopt;
    return Vec3{std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
  }
  std::optional<Vec2> forward(Vec2 p) const {
    auto v = ray(p);
    if (!v) return std::nullopt;
    double z = sphere::dot(*v, c);
    if (z <= 0.) return std::nullopt;
    Vec2 q = {TILE / 2. + FOCAL * sphere::dot(*v, e) / z, TILE / 2. - FOCAL * sphere::dot(*v, n) / z};
    if (q[0] >= 0. && q[0] < TILE && q[1] >= 0. && q[1] < TILE) return q;
    return std::nullopt;
  }
  Vec2 inverse(Vec2 q) const {
    double x = (q[0] - TILE / 2.) / FOCAL;
    double y = -(q[1] - TILE / 2.) / FOCAL;
    Vec3 v = sphere::normalize({c[0] + x * e[0] + y * n[0], c[1] + x * e[1] + y * n[1], c[2] + x * e[2] + y * n[2]});
    return {center[0] + focal * std::atan2(v[1], v[0]), horizon - focal * std::asin(v[2])};
  }
};
}  // namespace

std::optional<Solution> panorama_search(const Index& idx, const Index* wide_idx, const StarField& field, const SolveConfig& cfg,
                                        Clock::time_point t0, double end, SolveStats& stats, size_t& trials) {
  const Index& index = wide_idx ? *wide_idx : idx;
  if (to_degrees(index.r7) * 60. < 4.) return std::nullopt;
  double w = double(field.width), h = double(field.height);
  std::vector<double> widths;
  for (const Star& s : field.stars)
    if (std::isfinite(s.fwhm) && s.fwhm > 0.) widths.push_back(s.fwhm);
  std::stable_sort(widths.begin(), widths.end(), [](double a, double b) { return total_cmp(a, b) < 0; });
  double psf_max = widths.size() / 4 < widths.size() ? 2. * widths[widths.size() / 4] : INF;
  const double centers[4][2] = {{0.5, 0.45}, {0.75, 0.45}, {0.25, 0.45}, {0.5, 0.25}};
  const double fovs[10][2] = {{240., 2. / 3.}, {260., 2. / 3.}, {220., 2. / 3.}, {180., 2. / 3.}, {300., 2. / 3.},
                              {280., 2. / 3.}, {320., 2. / 3.}, {360., 2. / 3.}, {240., 0.5},      {240., 0.85}};
  for (auto& cc : centers) {
    double cx = cc[0], cy = cc[1];
    for (auto& fh : fovs) {
      double fov = fh[0], horizon_fraction = fh[1];
      double left = end - ms_since(t0);
      if (left < MIN_WINDOW_MS) return std::nullopt;
      Tile tile(w, h * horizon_fraction, fov, {w * cx, h * cy});
      StarField sub;
      sub.width = uint32_t(TILE);
      sub.height = uint32_t(TILE);
      sub.epoch = field.epoch;
      for (const Star& s : field.stars) {
        if (!(s.fwhm <= psf_max || !std::isfinite(s.fwhm))) continue;
        if (cfg.mask && cfg.mask->is_blocked(s.x, s.y)) continue;
        auto p = tile.forward({s.x, s.y});
        if (!p) continue;
        Star t = s;
        t.x = (*p)[0];
        t.y = (*p)[1];
        sub.stars.push_back(t);
      }
      if (sub.stars.size() < MIN_WINDOW_STARS) continue;
      SolveConfig c = cfg;
      c.deadline_ms = clampd(left * 0.25, MIN_WINDOW_MS, 180.);
      c.windows = false;
      c.keep_matches = true;
      c.extra_trials = trials;
      c.mask.reset();
      c.wide_index.reset();
      c.min_fov_deg = WIDE_INDEX_FOV.first;
      c.max_fov_deg = WIDE_INDEX_FOV.second;
      Solution ws = solve_field(index, sub, c);
      trials += ws.stats.verifications;
      accumulate(stats, ws.stats);
      if (ws.status == "ambiguous") {
        Solution sol = blank_solution("ambiguous", field.width, field.height);
        sol.margin_nats = ws.margin_nats;
        sol.n_candidates = ws.n_candidates;
        stats.total_ms = ms_since(t0);
        stats.verifications = trials - cfg.extra_trials;
        sol.stats = stats;
        return sol;
      }
      if (ws.status != "solved") continue;
      Pairs pairs;
      for (const auto& m : ws.matches) {
        Vec2 p = tile.inverse({m.x, m.y});
        pairs.push_back({p[0], p[1], m.ra, m.dec});
      }
      linalg::Mat cross(3, 3);
      Vec3 mean = {0., 0., 0.};
      for (size_t i = 0; i < pairs.size(); ++i) {
        if (i % 4 == 0) continue;
        auto v = tile.ray({pairs[i][0], pairs[i][1]});
        if (!v) return std::nullopt;
        Vec3 sky = unit(pairs[i][2], pairs[i][3]);
        for (size_t a = 0; a < 3; ++a)
          for (size_t b = 0; b < 3; ++b) cross(a, b) += (*v)[a] * sky[b];
        for (size_t k = 0; k < 3; ++k) mean[k] += sky[k];
      }
      auto svd = linalg::svd(cross);
      double rot[3][3];
      for (size_t a = 0; a < 3; ++a)
        for (size_t b = 0; b < 3; ++b) {
          double s = 0;
          for (size_t k = 0; k < 3; ++k) s += svd.u(a, k) * svd.v(b, k);
          rot[a][b] = s;
        }
      auto [pa, pd] = radec({rot[2][0], rot[2][1], rot[2][2]});
      Vec2 cp = {w * cx, h * cy};
      auto [ra, dec] = radec(sphere::normalize(mean));
      Vec2 tangent = {ra, dec};
      auto seed = wcsfit::cylinder_seed(pairs, cp, tangent, {pa, pd});
      if (!seed) continue;
      Pairs kept;
      for (auto& p : pairs) {
        auto q = seed->sky_to_pixel(p[2], p[3]);
        if (q && std::hypot((*q)[0] - p[0], (*q)[1] - p[1]) < 2.5) kept.push_back(p);
      }
      pairs = std::move(kept);
      if (pairs.size() < 32) continue;
      Rect extent = {INF, INF, -INF, -INF};
      for (auto& p : pairs) {
        extent[0] = fmin_r(extent[0], p[0]);
        extent[1] = fmin_r(extent[1], p[1]);
        extent[2] = fmax_r(extent[2], p[0]);
        extent[3] = fmax_r(extent[3], p[1]);
      }
      Rect win = {extent[0], extent[1], extent[2] - extent[0], extent[3] - extent[1]};
      SolveConfig c2 = cfg;
      c2.deadline_ms = fmax_r(end - ms_since(t0), 0.);
      c2.extra_trials = trials;
      auto ctxp = make_context(idx, field, c2, Clock::now());
      Solver& ctx = *ctxp;
      auto fc = ctx.frame_catalog(*seed, ctx.frame_level(*seed));
      if (!fc) continue;
      auto grown = ctx.grow(*seed, win, 1., *fc, Projection(seed->proj, seed->native_pole));
      if (grown) {
        size_t retained = 0;
        for (auto& p : pairs) {
          auto q = grown->wcs.sky_to_pixel(p[2], p[3]);
          if (q && std::hypot((*q)[0] - p[0], (*q)[1] - p[1]) < 2.5) ++retained;
        }
        if (retained * 10 >= pairs.size() * 9 && grown_holds(ctx, grown->wcs, grown->pairs, grown->rms) &&
            ctx.extension_evidence(grown->wcs, extent)) {
          auto sol = finish(ctx, field, c2, ws, win, *grown);
          if (!sol) return std::nullopt;
          stats.total_ms = ms_since(t0);
          stats.verifications = trials - cfg.extra_trials;
          sol->stats = stats;
          return sol;
        }
      }
    }
  }
  return std::nullopt;
}

}  // namespace wide
}  // namespace skyshot::solve
