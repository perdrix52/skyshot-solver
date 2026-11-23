// Window search for fields the whole-frame search cannot solve, and the whole-frame extension.
// Port of src/solve/wide.rs and src/solve/wide/panorama.rs.
#include <algorithm>

#include "skyshot/linalg.hpp"
#include "solve_internal.hpp"

namespace skyshot::solve {

using geometry::Projection;
using geometry::unit;
using sphere::angle;
using sphere::radec;

namespace {
constexpr std::array<std::array<double, 4>, 20> WINDOWS = {{
    {0.2, 0.2, 0.8, 0.8},     {0.3, 0.3, 0.7, 0.7},    {0.36, 0.36, 0.64, 0.64}, {0.2, 0.0, 0.8, 0.6},
    {0.3, 0.05, 0.7, 0.45},   {0.36, 0.08, 0.64, 0.36}, {0.0, 0.0, 0.6, 0.6},     {0.4, 0.0, 1.0, 0.6},
    {0.05, 0.05, 0.45, 0.45}, {0.55, 0.05, 0.95, 0.45}, {0.12, 0.12, 0.4, 0.4},    {0.6, 0.12, 0.88, 0.4},
    {0.0, 0.2, 0.6, 0.8},     {0.4, 0.2, 1.0, 0.8},    {0.2, 0.4, 0.8, 1.0},     {0.05, 0.3, 0.45, 0.7},
    {0.55, 0.3, 0.95, 0.7},   {0.12, 0.36, 0.4, 0.64}, {0.6, 0.36, 0.88, 0.64},  {0.36, 0.6, 0.64, 0.88},
}};
constexpr double WINDOW_SHARE = 0.4;
constexpr double NO_MASK_BUDGET_MS = 300.0;
constexpr double GROWN_MAX_RMS = 4.0;
using Rect = std::array<double, 4>;

size_t count_in(const StarField& field, const Rect& r, const SkyMask* mask) {
  size_t n = 0;
  for (const Star& s : field.stars)
    if (s.x >= r[0] && s.x < r[2] && s.y >= r[1] && s.y < r[3] && !(mask && mask->is_blocked(s.x, s.y))) ++n;
  return n;
}

std::vector<Rect> plan(const StarField& field, const SkyMask* mask) {
  double w = double(field.width), h = double(field.height);
  Rect b = mask ? mask->open_bbox(w, h) : Rect{0.0, 0.0, w, h};
  double bw = b[2] - b[0], bh = b[3] - b[1];
  std::vector<std::pair<Rect, double>> out;
  std::vector<Rect> rectangles;
  for (size_t i = 0; i < WINDOWS.size(); ++i) {
    const Rect& f = WINDOWS[i];
    if (i == 0 && fmax_r(bw, bh) > 2.0 * fmin_r(bw, bh)) {
      double side = 0.2 * fmin_r(bw, bh);
      std::vector<Rect> compact;
      const double cs[4][2] = {{0.5, 0.5}, {0.25, 0.5}, {0.75, 0.5}, {0.5, 0.25}};
      for (auto& c : cs) {
        double x = b[0] + c[0] * bw, y = b[1] + c[1] * bh;
        compact.push_back({x - side / 2.0, y - side / 2.0, x + side / 2.0, y + side / 2.0});
      }
      std::vector<size_t> counts;
      for (auto& r : compact) counts.push_back(count_in(field, r, mask));
      std::vector<size_t> order(compact.size());
      for (size_t k = 0; k < order.size(); ++k) order[k] = k;
      std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) { return counts[a] > counts[c]; });
      for (size_t k : order) rectangles.push_back(compact[k]);
    }
    if (i == 3 && fmax_r(bw, bh) > 1.5 * fmin_r(bw, bh)) {
      for (double size : {0.4, 0.28}) {
        double side = size * fmin_r(bw, bh);
        const double cs[4][2] = {{0.5, 0.5}, {0.5, 0.25}, {0.25, 0.5}, {0.75, 0.5}};
        for (auto& c : cs) {
          double x = b[0] + c[0] * bw, y = b[1] + c[1] * bh;
          rectangles.push_back({x - side / 2., y - side / 2., x + side / 2., y + side / 2.});
        }
      }
    }
    rectangles.push_back({b[0] + f[0] * bw, b[1] + f[1] * bh, b[0] + f[2] * bw, b[1] + f[3] * bh});
  }
  for (const Rect& r : rectangles) {
    double open = mask ? mask->open_fraction(r[0], r[1], r[2], r[3]) : 1.0;
    if (open < 0.5) continue;
    size_t n = count_in(field, r, mask);
    if (n < wide::MIN_WINDOW_STARS) continue;
    out.emplace_back(r, open >= 0.9 ? 10.0 : std::round(open * 10.0));
  }
  std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return total_cmp(b.second, a.second) < 0; });
  std::vector<Rect> res;
  for (auto& o : out) res.push_back(o.first);
  return res;
}

StarField window_field(const StarField& field, const Rect& r) {
  StarField sf;
  sf.width = sat_cast<uint32_t>(fmax_r(std::round(r[2] - r[0]), 1.0));
  sf.height = sat_cast<uint32_t>(fmax_r(std::round(r[3] - r[1]), 1.0));
  for (const Star& s : field.stars)
    if (s.x >= r[0] && s.x < r[2] && s.y >= r[1] && s.y < r[3]) {
      Star t = s;
      t.x = s.x - r[0];
      t.y = s.y - r[1];
      sf.stars.push_back(t);
    }
  sf.epoch = field.epoch;
  return sf;
}

}  // namespace

namespace wide {

std::optional<Solution> solve_windows(const Index& idx, const StarField& field, const SolveConfig& cfg, Clock::time_point t0,
                                      const SolveStats& spent) {
  return solve_windows_within(idx, field, cfg, t0, spent, false);
}

void accumulate(SolveStats& total, const SolveStats& s) {
  total.hint_used |= s.hint_used;
  total.hint_fallback |= s.hint_fallback;
  total.exhaustive_pass |= s.exhaustive_pass;
  total.deep_search |= s.deep_search;
  total.rungs_tried += s.rungs_tried;
  total.anchors += s.anchors;
  total.keys += s.keys;
  total.probes += s.probes;
  total.records += s.records;
  total.hits += s.hits;
  total.bins += s.bins;
  total.verifications += s.verifications;
  total.cheap_rejects += s.cheap_rejects;
  total.stage2 += s.stage2;
  total.branch2 += s.branch2;
  total.keygen_ms += s.keygen_ms;
  total.select_ms += s.select_ms;
  total.probe_ms += s.probe_ms;
  total.verify_ms += s.verify_ms;
}

bool mapping_holds(const Solver& ctx, const Wcs& wcs) {
  Rect b = ctx.cfg.mask ? ctx.cfg.mask->open_bbox(ctx.w, ctx.h) : Rect{0.0, 0.0, ctx.w, ctx.h};
  for (int i = 0; i <= 8; ++i)
    for (int j = 0; j <= 8; ++j) {
      double x = b[0] + double(i) * fmax_r(b[2] - b[0] - 1.0, 0.0) / 8.0;
      double y = b[1] + double(j) * fmax_r(b[3] - b[1] - 1.0, 0.0) / 8.0;
      if (!ctx.visible({x, y})) continue;
      Vec2 sky = wcs.pixel_to_sky(x, y);
      auto back = wcs.sky_to_pixel(sky[0], sky[1]);
      if (!back) return false;
      if (!(std::isfinite((*back)[0]) && std::isfinite((*back)[1])) || std::hypot((*back)[0] - x, (*back)[1] - y) > 0.1) return false;
    }
  return true;
}

bool grown_holds(const Solver& ctx, const Wcs& wcs, const Pairs& pairs, double rms) {
  if (pairs.size() < 20 || !std::isfinite(rms) || rms > GROWN_MAX_RMS) return false;
  if (!mapping_holds(ctx, wcs)) return false;
  Rect b = ctx.cfg.mask ? ctx.cfg.mask->open_bbox(ctx.w, ctx.h) : Rect{0.0, 0.0, ctx.w, ctx.h};
  double x0 = std::numeric_limits<double>::max(), x1 = std::numeric_limits<double>::lowest(), y0 = x0, y1 = x1;
  for (auto& p : pairs) {
    x0 = fmin_r(x0, p[0]);
    x1 = fmax_r(x1, p[0]);
    y0 = fmin_r(y0, p[1]);
    y1 = fmax_r(y1, p[1]);
  }
  return x1 - x0 >= 0.4 * (b[2] - b[0]) && y1 - y0 >= 0.4 * (b[3] - b[1]);
}

std::optional<Solution> finish(Solver& ctx, const StarField& field, const SolveConfig& cfg, const Solution& ws, std::array<double, 4> win,
                               const Grown& grown) {
  const Wcs& wcs = grown.wcs;
  Vec2 crpix = {(ctx.w - 1.0) / 2.0, (ctx.h - 1.0) / 2.0};
  Vec2 center = wcs.pixel_to_sky(crpix[0], crpix[1]);
  size_t n_matches = grown.pairs.empty() ? ws.n_matches : grown.pairs.size();
  auto refraction = ctx.measure_refraction(wcs, grown.pairs);
  Solution s;
  s.status = "solved";
  s.width = field.width;
  s.height = field.height;
  s.ra = center[0];
  s.dec = center[1];
  s.scale_arcsec = wcs.scale_arcsec();
  s.rotation_deg = wcs.rotation_deg();
  s.parity = wcs.determinant() > 0.0 ? 1 : -1;
  s.fov_width_deg = ctx.fov_width(wcs);
  s.n_matches = n_matches;
  s.n_catalog = ws.n_catalog;
  s.rms_px = grown.rms;
  s.log_odds = ws.log_odds;
  s.cluster_anchors = ws.cluster_anchors;
  s.verify_level = ws.verify_level;
  if (cfg.keep_matches) s.matches = ctx.match_out(grown.pairs, wcs);
  s.wcs = wcs;
  s.margin_nats = ws.margin_nats;
  s.n_candidates = ws.n_candidates;
  s.window = win;
  s.refraction = refraction;
  s.sky_extent = ctx.sky_extent();
  return s;
}

std::optional<Solution> refit_whole(Solver& ctx, const Solution& f) {
  if (!f.wcs) return std::nullopt;
  Wcs w0 = *f.wcs;
  auto fc = ctx.frame_catalog(w0, ctx.frame_level(w0));
  if (!fc) return std::nullopt;
  Rect rect = {0.0, 0.0, ctx.w, ctx.h};
  Projection start = ((w0.proj == Proj::Arc && ctx.cfg.allow_arc) || (w0.proj != Proj::Arc && !is_tan(w0.proj) && ctx.cfg.allow_panorama))
                         ? Projection(w0.proj, w0.native_pole)
                         : Projection(Proj::Tan);
  auto g = ctx.grow(w0, {0.0, 0.0, ctx.w, ctx.h}, f.rms_px, *fc, start);
  if (!g) return std::nullopt;
  size_t n_new = ctx.inliers_in(*fc, g->wcs, rect, 2.5), n_old = ctx.inliers_in(*fc, w0, rect, 2.5);
  if (double(n_new) < 0.9 * double(n_old) || !grown_holds(ctx, g->wcs, g->pairs, g->rms)) return std::nullopt;
  Vec2 crpix = {(ctx.w - 1.0) / 2.0, (ctx.h - 1.0) / 2.0};
  Vec2 center = g->wcs.pixel_to_sky(crpix[0], crpix[1]);
  Solution s = f;
  s.ra = center[0];
  s.dec = center[1];
  s.scale_arcsec = g->wcs.scale_arcsec();
  s.rotation_deg = g->wcs.rotation_deg();
  s.parity = g->wcs.determinant() > 0.0 ? 1 : -1;
  s.fov_width_deg = ctx.fov_width(g->wcs);
  s.n_matches = g->pairs.size();
  s.rms_px = g->rms;
  s.matches = ctx.cfg.keep_matches ? ctx.match_out(g->pairs, g->wcs) : std::vector<MatchOut>{};
  s.wcs = g->wcs;
  return s;
}

std::optional<Solution> solve_windows_within(const Index& idx, const StarField& field, const SolveConfig& cfg, Clock::time_point t0,
                                             const SolveStats& spent, bool uncapped) {
  size_t trials = cfg.extra_trials + spent.verifications;
  SolveStats stats = spent;
  bool landscape = cfg.mask && cfg.mask->is_landscape();
  double start = ms_since(t0);
  double end = (landscape || uncapped) ? cfg.deadline_ms : fmin_r(cfg.deadline_ms, start + NO_MASK_BUDGET_MS);
  double w = double(field.width), h = double(field.height);
  const Index* wi = cfg.wide_index.get();
  if (cfg.allow_panorama && w > 1.8 * h) {
    if (auto sol = panorama_search(idx, wi, field, cfg, t0, start + 0.55 * (end - start), stats, trials)) return sol;
  }
  struct Search {
    Rect r;
    const Index* index;
    bool coarse;
  };
  std::vector<Search> searches;
  auto rects = plan(field, cfg.mask.get());
  for (size_t i = 0; i < rects.size(); ++i) {
    if (i == 1 && wi) searches.push_back({{0.0, 0.0, w, h}, wi, true});
    searches.push_back({rects[i], &idx, false});
    if (wi) searches.push_back({rects[i], wi, true});
  }
  if (searches.size() <= 2 && wi) searches.push_back({{0.0, 0.0, w, h}, wi, true});
  for (const Search& se : searches) {
    const Rect& r = se.r;
    double left = end - ms_since(t0);
    if (left < MIN_WINDOW_MS) break;
    StarField sub = window_field(field, r);
    double min_fov = se.coarse ? WIDE_INDEX_FOV.first : cfg.min_fov_deg;
    double max_fov = se.coarse ? WIDE_INDEX_FOV.second : cfg.max_fov_deg;
    SolveConfig sub_cfg = cfg;
    sub_cfg.mode = SolveMode::Standard;
    sub_cfg.deep_index.reset();
    sub_cfg.deadline_ms = fmin_r(fmax_r(left * WINDOW_SHARE, MIN_WINDOW_MS), left);
    sub_cfg.windows = false;
    sub_cfg.keep_matches = true;
    sub_cfg.extra_trials = trials;
    if (cfg.mask) sub_cfg.mask = std::make_shared<SkyMask>(cfg.mask->shifted(r[0], r[1]));
    sub_cfg.min_fov_deg = min_fov;
    sub_cfg.max_fov_deg = max_fov;
    sub_cfg.wide_index.reset();
    Solution ws = solve_field(*se.index, sub, sub_cfg);
    trials += ws.stats.verifications;
    accumulate(stats, ws.stats);
    if (ws.status == "solved") {
      Rect win = {r[0], r[1], double(sub.width), double(sub.height)};
      SolveConfig ext_cfg = cfg;
      ext_cfg.deadline_ms = fmax_r(end - ms_since(t0), 0.0);
      ext_cfg.extra_trials = trials;
      auto ext = extend(idx, field, ext_cfg, ws, win);
      if (!ext) continue;
      Solution sol = std::move(*ext);
      stats.total_ms = ms_since(t0);
      stats.verifications = trials - cfg.extra_trials;
      sol.stats = stats;
      return sol;
    } else if (ws.status == "ambiguous") {
      Solution s = blank_solution("ambiguous", field.width, field.height);
      s.margin_nats = ws.margin_nats;
      s.n_candidates = ws.n_candidates;
      s.stats = stats;
      return s;
    }
  }
  return std::nullopt;
}

}  // namespace wide
}  // namespace skyshot::solve
