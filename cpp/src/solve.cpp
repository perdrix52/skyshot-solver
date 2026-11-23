// Blind solver: dominance rungs -> fingerprints -> index probes -> consistency filters ->
// similarity hypotheses -> Hough voting -> two-stage verification. Port of src/solve.rs.
#include "skyshot/solve.hpp"

#include <algorithm>

#include "skyshot/linalg.hpp"
#include "solve_internal.hpp"

namespace skyshot {

const char* solve_mode_name(SolveMode m) {
  switch (m) {
    case SolveMode::Auto: return "auto";
    case SolveMode::Standard: return "standard";
    default: return "deep";
  }
}

namespace solve {

using geometry::unit;
using sphere::angle;
using sphere::gnomonic;
using sphere::radec;
using sphere::tangent_basis;

// ---------- JSON ----------
Json SolveStats::to_json() const {
  Json o = Json::object();
  if (hint_used) o.set("hint_used", true);
  if (hint_fallback) o.set("hint_fallback", true);
  if (deep_search) o.set("deep_search", true);
  o.set("n_stars", uint64_t(n_stars)).set("rungs_tried", uint64_t(rungs_tried)).set("anchors", uint64_t(anchors));
  o.set("keys", uint64_t(keys)).set("probes", uint64_t(probes)).set("records", uint64_t(records)).set("hits", uint64_t(hits));
  o.set("bins", uint64_t(bins)).set("verifications", uint64_t(verifications)).set("cheap_rejects", uint64_t(cheap_rejects));
  o.set("stage2", uint64_t(stage2)).set("exhaustive_pass", exhaustive_pass).set("branch2", uint64_t(branch2));
  o.set("keygen_ms", keygen_ms).set("select_ms", select_ms).set("probe_ms", probe_ms).set("verify_ms", verify_ms).set("total_ms", total_ms);
  return o;
}

Json MatchOut::to_json() const {
  Json o = Json::object();
  o.set("x", x).set("y", y).set("ra", ra).set("dec", dec).set("residual_px", residual_px);
  return o;
}

Json Solution::to_json() const {
  Json o = Json::object();
  o.set("status", status).set("width", width).set("height", height);
  o.set("wcs", wcs ? wcs->to_json() : Json());
  o.set("ra", ra).set("dec", dec).set("scale_arcsec", scale_arcsec).set("rotation_deg", rotation_deg).set("parity", parity);
  o.set("fov_width_deg", fov_width_deg).set("n_matches", uint64_t(n_matches)).set("n_catalog", uint64_t(n_catalog));
  o.set("rms_px", rms_px).set("log_odds", log_odds).set("cluster_anchors", uint64_t(cluster_anchors)).set("verify_level", verify_level);
  o.set("stats", stats.to_json());
  Json m = Json::array();
  for (const auto& x : matches) m.push(x.to_json());
  o.set("matches", std::move(m));
  o.set("margin_nats", margin_nats).set("n_candidates", n_candidates);
  if (window) o.set("window", Json::from_arr(*window));
  if (refraction) o.set("refraction", refraction->to_json());
  if (sky_extent) o.set("sky_extent", Json::from_arr(*sky_extent));
  return o;
}

// ---------- vote grid ----------
static std::pair<uint64_t, uint32_t> vote_grid(double s, double wmax) {
  double s_arcsec = s * (180.0 / PI * 3600.0);
  int64_t lsi = sat_cast<int64_t>(std::floor(std::log(s_arcsec) * 4.0)) + 64;
  uint64_t ls = uint64_t(std::clamp<int64_t>(lsi, 0, 255));
  double fov = std::exp((double(ls) - 64.0 + 0.5) * 0.25) / 3600.0 * (PI / 180.0) * wmax;
  uint32_t nb = sat_cast<uint32_t>(clampd(std::round(std::log2(8.0 / fov)), 0.0, 15.0));
  return {ls, nb};
}

static double vote_cell_rad(double s, double wmax) { return 2.0 / double(uint64_t(1) << vote_grid(s, wmax).second); }

static bool same_hypothesis(const Solution& a, const Solution& b, double wmax) {
  if (a.parity != b.parity) return false;
  double sa = a.scale_arcsec, sb = b.scale_arcsec;
  if (!(sa > 0.0 && sb > 0.0 && std::isfinite(sa) && std::isfinite(sb))) return false;
  if (std::fabs(std::log(sa / sb)) > 0.5) return false;
  double drot = rem_euclid(a.rotation_deg - b.rotation_deg, 360.0);
  if (fmin_r(drot, 360.0 - drot) > 20.0) return false;
  double s = to_radians(fmin_r(sa, sb)) / 3600.0;
  return angle(unit(a.ra, a.dec), unit(b.ra, b.dec)) <= 2.0 * vote_cell_rad(s, wmax);
}

Solution blank_solution(const std::string& status, uint32_t width, uint32_t height) {
  Solution s;
  s.status = status;
  s.width = width;
  s.height = height;
  return s;
}

using C2 = std::pair<double, double>;

static std::optional<std::pair<C2, C2>> fit_similarity(const std::vector<C2>& z, const std::vector<C2>& w) {
  double n = double(z.size());
  if (z.size() < 2) return std::nullopt;
  double zsx = -0.0, zsy = -0.0, wsx = -0.0, wsy = -0.0;
  for (auto& p : z) zsx += p.first;
  for (auto& p : z) zsy += p.second;
  for (auto& p : w) wsx += p.first;
  for (auto& p : w) wsy += p.second;
  C2 zm{zsx / n, zsy / n}, wm{wsx / n, wsy / n};
  double num_re = 0, num_im = 0, den = 0;
  for (size_t i = 0; i < z.size() && i < w.size(); ++i) {
    double zx = z[i].first - zm.first, zy = z[i].second - zm.second;
    double wx = w[i].first - wm.first, wy = w[i].second - wm.second;
    num_re += wx * zx + wy * zy;
    num_im += wy * zx - wx * zy;
    den += zx * zx + zy * zy;
  }
  if (den <= 0.0) return std::nullopt;
  C2 a{num_re / den, num_im / den};
  C2 b{wm.first - (a.first * zm.first - a.second * zm.second), wm.second - (a.first * zm.second + a.second * zm.first)};
  return std::make_pair(a, b);
}

static Wcs wcs_from_similarity(const Vec3& center, C2 a, double p, Vec2 crpix) {
  auto [ra, dec] = radec(center);
  double k = 180.0 / PI;
  Wcs w;
  w.ra = ra;
  w.dec = dec;
  w.crpix = crpix;
  w.cd = {{{a.first * k, -a.second * p * k}, {a.second * k, a.first * p * k}}};
  w.proj = Proj::Tan;
  return w;
}

MatchResult match_projected(const std::vector<Vec2>& pts, const select::Grid2& grid, const std::vector<std::pair<Vec2, Vec2>>& projected,
                            double radius) {
  std::unordered_set<uint32_t> used;
  MatchResult r;
  for (auto& [sky, px] : projected) {
    auto nn = grid.nearest_within(pts, px, radius, UINT32_MAX);
    if (nn && used.insert(nn->second).second) r.m.emplace_back(nn->second, sky);
  }
  r.n = r.m.size();
  return r;
}

double poisson_log_odds(size_t k, double lambda) {
  double kf = double(k);
  if (kf > lambda) return kf * std::log(kf / lambda) - kf + lambda;
  return 0.0;
}

double log_binom_tail(size_t n_, size_t k_, double p) {
  double n = double(n_), k = double(k_);
  if (k <= n * p) return 0.0;
  if (k >= n) return n * std::log(p);
  double q = k / n;
  return -n * (q * std::log(q / p) + (1.0 - q) * std::log((1.0 - q) / (1.0 - p)));
}

// ---------- Solver ----------
Solver::Solver(const Index& idx_, const StarField& field, const SolveConfig& cfg_, Clock::time_point t0_)
    : idx(idx_), cfg(cfg_), t0(t0_), n_refs(cfg_.n_refs), n_rest(cfg_.n_rest) {
  const SkyMask* mask = cfg.mask.get();
  std::vector<const Star*> stars;
  for (const Star& s : field.stars) {
    if (!(std::isfinite(s.x) && std::isfinite(s.y))) continue;
    if (mask && mask->is_blocked(s.x, s.y)) continue;
    stars.push_back(&s);
  }
  std::stable_sort(stars.begin(), stars.end(), [](const Star* a, const Star* b) { return a->flux > b->flux; });
  if (stars.size() > cfg.max_stars) stars.resize(cfg.max_stars);
  w = double(field.width);
  h = double(field.height);
  for (const Star* s : stars) {
    pts.push_back({s->x, s->y});
    sig.push_back(clampd(s->sigma, 0.05, 2.0));
  }
  d = select::dominance_px(pts, w, h);
  edge = select::edge_px(pts, w, h);
  open_area = mask ? fmax_r(w * h - mask->blocked_area(0.0, 0.0, w, h), 1.0) : w * h;
  if (cfg.hint && cfg.hint->position) {
    auto pos = *cfg.hint->position;
    double radius = 3.0;
    if (auto f = cfg.hint->fov_deg(field.width, field.height)) radius = fmax_r(*f * SQRT_2, 3.0);
    radius = fmin_r(radius, 180.0);
    hint_cone = std::make_pair(sphere::unit(pos[0], pos[1]), std::cos(to_radians(radius)));
  }
  stats.n_stars = stars.size();
  hits.reserve(4096);
  for (int i = 0; i < 1024; ++i) {
    pa_sin.push_back(std::sin((double(i) + 0.5) * TAU / 1024.0));
    pa_cos.push_back(std::cos((double(i) + 0.5) * TAU / 1024.0));
  }
}

std::unique_ptr<Solver> make_context(const Index& idx, const StarField& field, const SolveConfig& cfg, Clock::time_point t0) {
  return std::make_unique<Solver>(idx, field, cfg, t0);
}

std::optional<Refraction> Solver::measure_refraction(const Wcs& wcs, const Pairs& pairs) const {
  if ((!cfg.windows && cfg.extra_trials > 0) || !(cfg.mask && cfg.mask->is_landscape()) ||
      wcs.width_deg(uint32_t(w), uint32_t(h)) < 30.0)
    return std::nullopt;
  return refract::measure(wcs, pairs, w, h);
}

std::optional<std::array<double, 4>> Solver::sky_extent() const {
  if (!cfg.mask) return std::nullopt;
  return cfg.mask->open_bbox(w, h);
}

double Solver::fov_width(const Wcs& wcs) const {
  if (auto e = sky_extent()) {
    double y = ((*e)[1] + (*e)[3] - 1.0) / 2.0;
    return wcs.arc_deg({(*e)[0] - 0.5, y}, {(*e)[2] - 0.5, y});
  }
  return wcs.width_deg(uint32_t(w), uint32_t(h));
}

bool Solver::visible(Vec2 px) const {
  return px[0] >= 0.0 && px[1] >= 0.0 && px[0] <= w - 1.0 && px[1] <= h - 1.0 && (!cfg.mask || !cfg.mask->is_blocked(px[0], px[1]));
}

double Solver::frame_radius(const Wcs& wcs, const Vec3& c, double margin) const {
  double x0 = -margin, y0 = -margin, x1 = w + margin, y1 = h + margin;
  double radius = 0.0;
  for (int k = 0; k <= 16; ++k) {
    double t = double(k) / 16.0;
    Vec2 qs[4] = {{x0 + t * (x1 - x0), y0}, {x0 + t * (x1 - x0), y1}, {x0, y0 + t * (y1 - y0)}, {x1, y0 + t * (y1 - y0)}};
    for (auto& q : qs) {
      Vec2 sky = wcs.pixel_to_sky(q[0], q[1]);
      radius = fmax_r(radius, angle(c, unit(sky[0], sky[1])));
    }
  }
  return radius;
}

double Solver::max_frame_radius(const Wcs& wcs, double tan_limit) const { return wcs.proj == Proj::Tan ? tan_limit : 2.9; }

std::optional<std::pair<Vec3, double>> Solver::hypothesis(Vec2 a_px, std::pair<double, double> dir, const Vec3& av, uint32_t pa_code,
                                                          double s, double p) const {
  double sp = pa_sin[pa_code], cp = pa_cos[pa_code];
  double ex, ey;
  if (p > 0.0) {
    ex = dir.first;
    ey = -dir.second;
  } else {
    ex = dir.first;
    ey = dir.second;
  }
  double rc = sp * ex - cp * ey;
  double rs = sp * ey + cp * ex;
  double vx = (w - 1.0) / 2.0 - a_px[0];
  double vy = p * ((h - 1.0) / 2.0 - a_px[1]);
  double xi = s * (rc * vx - rs * vy);
  double eta = s * (rs * vx + rc * vy);
  double rr2 = xi * xi + eta * eta;
  double k;
  if (rr2 < 0.04) {
    k = 1.0 + rr2 / 3.0;
  } else {
    double rr = std::sqrt(rr2);
    if (rr > 1.4) return std::nullopt;
    k = std::tan(rr) / rr;
  }
  auto [east, north] = tangent_basis(av);
  Vec3 c = sphere::normalize({av[0] + k * (xi * east[0] + eta * north[0]), av[1] + k * (xi * east[1] + eta * north[1]),
                              av[2] + k * (xi * east[2] + eta * north[2])});
  Vec3 u = {rc * east[0] + rs * north[0], rc * east[1] + rs * north[1], rc * east[2] + rs * north[2]};
  auto [ce, cn] = tangent_basis(c);
  double phi_c = std::atan2(sphere::dot(u, cn), sphere::dot(u, ce));
  return std::make_pair(c, phi_c);
}

uint64_t Solver::bin_key(const Vec3& center, double s, double phi, double p) const {
  auto [ls, nb] = vote_grid(s, fmax_r(w, h));
  uint64_t n = uint64_t(1) << nb;
  double ax = std::fabs(center[0]), ay = std::fabs(center[1]), az = std::fabs(center[2]);
  uint64_t face;
  double u, v;
  if (ax >= ay && ax >= az) {
    face = center[0] > 0.0 ? 0 : 1;
    u = center[1] / ax;
    v = center[2] / ax;
  } else if (ay >= az) {
    face = center[1] > 0.0 ? 2 : 3;
    u = center[0] / ay;
    v = center[2] / ay;
  } else {
    face = center[2] > 0.0 ? 4 : 5;
    u = center[0] / az;
    v = center[1] / az;
  }
  uint64_t i = std::min(sat_cast<uint64_t>((u + 1.0) * 0.5 * double(n)), n - 1);
  uint64_t j = std::min(sat_cast<uint64_t>((v + 1.0) * 0.5 * double(n)), n - 1);
  uint64_t cell = (face << 32) | (j << 16) | i;
  uint64_t pb = sat_cast<uint64_t>(rem_euclid(phi, TAU) / to_radians(10.0)) % 36;
  uint64_t par = p > 0.0 ? 1 : 0;
  return cell ^ (uint64_t(nb) << 35) ^ (pb << 40) ^ (ls << 46) ^ (par << 54);
}

std::optional<uint64_t> Solver::add_hit(const Hit& hit) {
  uint64_t key = bin_key(hit.center, hit.s, hit.phi, double(hit.parity));
  uint32_t hi = uint32_t(hits.size());
  Bin& bin = bins[key];
  Hit h2 = hit;
  if (bin.hits == 0) bin.parity = int8_t(hit.parity);
  h2.next = bin.hits == 0 ? NONE : bin.head;
  bin.head = hi;
  bin.hits += 1;
  bool present = false;
  for (size_t i = 0; i < bin.n_anchors; ++i)
    if (bin.anchors[i] == hit.img_anchor) present = true;
  if (!present && bin.n_anchors < bin.anchors.size()) {
    bin.anchors[bin.n_anchors] = hit.img_anchor;
    bin.n_anchors += 1;
  }
  size_t n = bin.n_anchors;
  bool trigger = !present && n >= cfg.trigger_anchors && (bin.verified_at == 0 || n >= size_t(bin.verified_at) + 2);
  hits.push_back(h2);
  if (trigger) {
    bin.verified_at = uint8_t(n);
    return key;
  }
  return std::nullopt;
}

std::optional<Solution> Solver::process_anchor(uint32_t a, const std::vector<std::pair<double, uint32_t>>& nb, double r,
                                               std::vector<fingerprint::KeyVariant>& variants, std::unordered_set<uint64_t>& seen) {
  size_t n_refs_l = std::min(n_refs, nb.size());
  double min_fov = to_radians(cfg.min_fov_deg);
  double max_fov = to_radians(cfg.max_fov_deg);
  double wmax = fmax_r(w, h);
  Vec2 a_px = pts[a];
  double d_a = d[a];
  bool exact_a = d_a < edge[a];
  elem_cache.clear();
  std::vector<index::ProbeRecord> recs;
  for (size_t ri = 0; ri < n_refs_l; ++ri) {
    uint32_t ref_id = nb[ri].second;
    std::vector<uint32_t> rest;
    for (auto& x : nb) {
      if (x.second == ref_id) continue;
      if (rest.size() >= n_rest) break;
      rest.push_back(x.second);
    }
    if (rest.size() < 3) continue;
    for (size_t i = 0; i < rest.size(); ++i)
      for (size_t j = i + 1; j < rest.size(); ++j)
        for (size_t k = j + 1; k < rest.size(); ++k) {
          std::array<uint32_t, 3> sub = {rest[i], rest[j], rest[k]};
          std::array<uint32_t, 4> ids = {ref_id, sub[0], sub[1], sub[2]};
          std::sort(ids.begin() + 1, ids.end());
          uint64_t sg = hash_u64((uint64_t(a) << 40) ^ (uint64_t(ids[0]) << 20) ^ uint64_t(ids[1])) ^
                        hash_u64((uint64_t(ids[2]) << 32) | uint64_t(ids[3]));
          if (!seen.insert(sg).second) continue;
          double sigma = 0.0;
          for (uint32_t q : {a, ref_id, sub[0], sub[1], sub[2]}) sigma = fmax_r(sigma, sig[q]);
          Vec2 r_px = pts[ref_id];
          double min_d = INF;
          for (uint32_t q : {ref_id, sub[0], sub[1], sub[2]}) min_d = fmin_r(min_d, d[q]);
          auto tk = Clock::now();
          std::array<Vec2, 3> nbr_px = {pts[sub[0]], pts[sub[1]], pts[sub[2]]};
          bool gate_only = idx.coder.has_value() && cfg.learned_k > 0;
          std::optional<double> ar;
          if (gate_only) ar = fingerprint::image_gate(a_px, r_px, nbr_px, sigma);
          else ar = fingerprint::image_keys(a_px, r_px, nbr_px, sigma, variants);
          if (!ar) {
            stats.keygen_ms += ms_since(tk);
            continue;
          }
          double ar_px = *ar;
          if (idx.coder) {
            size_t kk = cfg.learned_k > 0 ? cfg.learned_k : variants.size();
            auto rel = [&](uint32_t q) { return std::array<double, 2>{pts[q][0] - a_px[0], pts[q][1] - a_px[1]}; };
            learned::Raw raw;
            raw.xy = {rel(ref_id), rel(sub[0]), rel(sub[1]), rel(sub[2])};
            raw.d = {d_a, d[ref_id], d[sub[0]], d[sub[1]], d[sub[2]]};
            raw.sigma = sigma;
            raw.edge = edge[a];
            auto b2 = idx.coder->image_variants(raw, kk, elem_cache, variants);
            if (!b2) {
              stats.keygen_ms += ms_since(tk);
              continue;
            }
            stats.branch2 += size_t(*b2);
          }
          stats.keygen_ms += ms_since(tk);
          stats.keys += 1;
          std::pair<double, double> dir{(r_px[0] - a_px[0]) / ar_px, (r_px[1] - a_px[1]) / ar_px};
          double lo_member = std::exp(-cfg.member_tol);
          for (size_t vi = 0; vi < variants.size(); ++vi) {
            fingerprint::KeyVariant v = variants[vi];
            stats.probes += 1;
            idx.probe(v.key, recs);
            for (const index::ProbeRecord& rec : recs) {
              stats.records += 1;
              uint32_t m = rec.level();
              double rm = idx.level_radius_m(m);
              double s = rec.ar_rad(idx.r7) / ar_px;
              double fov = s * wmax;
              if (fov < min_fov || fov > max_fov) continue;
              if (std::log(s * r / rm) > cfg.rung_tol) continue;
              if (s * min_d < rm * lo_member) continue;
              uint32_t dc = rec.dcode();
              double lo = rm * std::exp2(double(dc) / 2.0);
              double hi = dc >= 7 ? INF : rm * std::exp2(double(dc + 1) / 2.0);
              double sd = s * d_a;
              double tol = std::exp(cfg.dom_tol);
              if (sd > hi * tol) continue;
              if (exact_a && sd < lo / tol) continue;
              double p = v.flip == rec.flip() ? 1.0 : -1.0;
              Vec3 av = idx.star(rec.anchor()).v;
              auto hyp = hypothesis(a_px, dir, av, rec.pa_code(), s, p);
              if (!hyp) continue;
              if (hint_cone && sphere::dot(hint_cone->first, hyp->first) < hint_cone->second) continue;
              stats.hits += 1;
              Hit hit;
              hit.img_anchor = a;
              hit.cat_anchor = rec.anchor();
              hit.center = hyp->first;
              hit.s = s;
              hit.phi = hyp->second;
              hit.parity = int32_t(p);
              hit.next = NONE;
              if (auto bk = add_hit(hit)) {
                auto tv = Clock::now();
                auto verified = verify_bin(*bk);
                stats.verify_ms += ms_since(tv);
                if (verified) {
                  if (cfg.min_margin_nats <= 0.0) return std::move(verified->first);
                  accept(std::move(verified->first), verified->second);
                  if (continuation_over()) return std::nullopt;
                }
              }
            }
          }
        }
  }
  return std::nullopt;
}

std::optional<std::pair<Solution, Hit>> Solver::verify_bin(uint64_t key) {
  auto it = bins.find(key);
  if (it == bins.end()) return std::nullopt;
  Bin bin = it->second;
  std::vector<Hit> all;
  uint32_t hh = bin.head;
  while (hh != NONE) {
    Hit hit = hits[hh];
    all.push_back(hit);
    hh = hit.next;
  }
  if (all.empty()) return std::nullopt;
  Hit first = all[0];
  if (continuing() && matches_candidate(first)) return std::nullopt;
  stats.verifications += 1;
  double fov_deg = to_degrees(first.s) * fmax_r(w, h);
  double tol_c = (0.03 + 0.002 * fov_deg) * all[0].s * fmax_r(w, h);
  double tol_s = 0.04 + 0.002 * fov_deg;
  double tol_p = to_radians(3.0 + 0.1 * fov_deg);
  auto consistent = [&](const Hit& a, const Hit& b) {
    double dphi = rem_euclid(a.phi - b.phi + PI, TAU) - PI;
    return std::fabs(std::log(a.s / b.s)) < tol_s && std::fabs(dphi) < tol_p && angle(a.center, b.center) < tol_c;
  };
  std::pair<size_t, size_t> best{0, 0};
  for (size_t i = 0; i < all.size(); ++i) {
    std::vector<uint32_t> img_seen, cat_seen;
    for (const Hit& o : all) {
      if (consistent(all[i], o) && std::find(img_seen.begin(), img_seen.end(), o.img_anchor) == img_seen.end() &&
          std::find(cat_seen.begin(), cat_seen.end(), o.cat_anchor) == cat_seen.end()) {
        img_seen.push_back(o.img_anchor);
        cat_seen.push_back(o.cat_anchor);
      }
    }
    if (img_seen.size() > best.first) best = {img_seen.size(), i};
  }
  if (best.first < 2) {
    stats.cheap_rejects += 1;
    return std::nullopt;
  }
  Hit seed = all[best.second];
  std::vector<std::pair<uint32_t, uint32_t>> prs;
  for (const Hit& o : all) {
    if (consistent(seed, o)) {
      bool any = false;
      for (auto& pr : prs)
        if (pr.first == o.img_anchor || pr.second == o.cat_anchor) any = true;
      if (!any) prs.emplace_back(o.img_anchor, o.cat_anchor);
    }
  }
  const Hit& base = seed;
  double p = double(base.parity);
  auto [ce, cn] = tangent_basis(base.center);
  Vec2 cpx = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  std::vector<C2> z, wv;
  for (auto& [ia, ca] : prs) {
    auto q = gnomonic(idx.star(ca).v, base.center, ce, cn);
    if (!q) return std::nullopt;
    Vec2 px = pts[ia];
    z.emplace_back(px[0] - cpx[0], p * (px[1] - cpx[1]));
    wv.emplace_back((*q)[0], (*q)[1]);
  }
  auto fit = fit_similarity(z, wv);
  if (!fit) return std::nullopt;
  C2 a = fit->first, b = fit->second;
  double s = std::hypot(a.first, a.second);
  if (!(std::isfinite(s) && s > 0.0)) return std::nullopt;
  double den = a.first * a.first + a.second * a.second;
  C2 z0{-(b.first * a.first + b.second * a.second) / den, -(b.second * a.first - b.first * a.second) / den};
  Vec2 crpix = {cpx[0] + z0.first, cpx[1] + p * z0.second};
  Wcs wcs = wcs_from_similarity(base.center, a, p, crpix);
  auto sol = verify_wcs(wcs, bin.n_anchors);
  if (!sol) return std::nullopt;
  return std::make_pair(std::move(*sol), base);
}

bool Solver::continuation_over() const {
  if (!accept_ms) return false;
  if (margin().second < cfg.min_margin_nats) return true;
  double cap = fmin_r(*accept_ms * (1.0 + fmax_r(cfg.continuation_factor, 0.0)), cfg.deadline_ms);
  return elapsed_ms() > cap;
}

bool Solver::matches_candidate(const Hit& hit) const {
  double wmax = fmax_r(w, h);
  for (const Candidate& c : cands) {
    const Hit& s = c.seed;
    double dphi = rem_euclid(hit.phi - s.phi + PI, TAU) - PI;
    if (hit.parity == s.parity && std::fabs(std::log(hit.s / s.s)) <= 0.5 && std::fabs(dphi) <= to_radians(20.0) &&
        angle(hit.center, s.center) <= 2.0 * vote_cell_rad(fmin_r(s.s, hit.s), wmax))
      return true;
  }
  return false;
}

void Solver::accept(Solution sol, const Hit& seed) {
  if (!accept_ms) accept_ms = elapsed_ms();
  double wmax = fmax_r(w, h);
  for (const Candidate& c : cands)
    if (same_hypothesis(c.sol, sol, wmax)) return;
  cands.push_back(Candidate{std::move(sol), seed});
}

std::pair<size_t, double> Solver::margin() const {
  size_t best = 0;
  double second = -INF;
  for (size_t i = 0; i < cands.size(); ++i) {
    if (i == 0) continue;
    double lo = cands[i].sol.log_odds;
    if (lo > cands[best].sol.log_odds) {
      second = cands[best].sol.log_odds;
      best = i;
    } else if (lo > second) {
      second = lo;
    }
  }
  switch (cands.size()) {
    case 0: return {0, NaN};
    case 1: return {0, INF};
    default: return {best, cands[best].sol.log_odds - second};
  }
}

void Solver::sweep_pending_bins() {
  if (cands.empty()) return;
  int8_t best_parity = int8_t(cands.front().sol.parity);
  uint8_t trigger = uint8_t(std::max<size_t>(cfg.trigger_anchors, 1));
  struct P {
    bool same;
    uint8_t n;
    uint64_t key;
  };
  std::vector<P> pending;
  for (auto& [k, b] : bins)
    if (b.n_anchors >= trigger && b.n_anchors > b.verified_at) pending.push_back({b.parity == best_parity, b.n_anchors, k});
  std::sort(pending.begin(), pending.end(), [](const P& a, const P& b) {
    if (a.same != b.same) return !a.same;
    if (a.n != b.n) return a.n > b.n;
    return a.key < b.key;
  });
  for (const P& p : pending) {
    if (continuation_over()) break;
    auto it = bins.find(p.key);
    if (it != bins.end()) it->second.verified_at = p.n;
    auto tv = Clock::now();
    auto verified = verify_bin(p.key);
    stats.verify_ms += ms_since(tv);
    if (verified) accept(std::move(verified->first), verified->second);
  }
}

std::optional<Solution> Solver::conclude() {
  if (cands.empty()) return std::nullopt;
  auto [best, mg] = margin();
  uint32_t n = uint32_t(cands.size());
  if (mg < cfg.min_margin_nats) {
    Solution s = blank_solution("ambiguous", uint32_t(w), uint32_t(h));
    s.log_odds = cands[best].sol.log_odds;
    s.margin_nats = mg;
    s.n_candidates = n;
    return s;
  }
  Solution s = std::move(cands[best].sol);
  s.margin_nats = mg;
  s.n_candidates = n;
  return s;
}

std::optional<Solution> Solver::verify_wcs(const Wcs& wcs0, size_t cluster) { return verify_wcs_within(wcs0, cluster, 0.0); }

Pairs Solver::pairs_of(const Matches& m) const {
  Pairs out;
  out.reserve(m.size());
  for (auto& [i, sky] : m) out.push_back({pts[i][0], pts[i][1], sky[0], sky[1]});
  return out;
}

std::vector<double> Solver::weights(const Matches& m) const {
  std::vector<double> out;
  out.reserve(m.size());
  for (auto& x : m) {
    double s = clampd(sig[x.first], 0.15, 2.0);
    out.push_back(1.0 / (s * s));
  }
  return out;
}

std::vector<MatchOut> Solver::match_out(const Pairs& pairs, const Wcs& wcs) const {
  std::vector<MatchOut> out;
  for (const auto& p : pairs) {
    auto q = wcs.sky_to_pixel(p[2], p[3]).value_or(Vec2{NaN, NaN});
    out.push_back(MatchOut{p[0], p[1], p[2], p[3], std::hypot(q[0] - p[0], q[1] - p[1])});
  }
  return out;
}

MatchResult Solver::count_matches(const select::Grid2& g, const Pairs& cat, double radius, const Wcs& wcs) const {
  auto projector = geometry::prepare(wcs);
  std::vector<std::pair<Vec2, Vec2>> proj;
  proj.reserve(cat.size());
  for (const auto& c : cat)
    if (auto px = projector.sky_to_pixel_fast(c[0], c[1])) proj.push_back({{c[0], c[1]}, *px});
  return match_projected(pts, g, proj, radius);
}

std::vector<index::StarInfo> Solver::catalog_cone(const Vec3& c, double radius, double rm) {
  bool needs_deep = rm < idx.r7 * 0.999 && idx.has_deep();
  if (cfg.mag_limit && !needs_deep) {
    std::vector<index::BrightStar> bright;
    idx.cone_stars(c, radius, *cfg.mag_limit, rm * 0.999, bright);
    std::vector<index::StarInfo> out;
    for (auto& b : bright) out.push_back(index::StarInfo{b.v, b.d_rad, b.mag});
    return out;
  }
  return idx.cached_cone_candidates(c, radius, rm * 0.999, cfg.mag_limit, needs_deep, catalog_cache);
}

std::optional<Pairs> Solver::catalog_in_frame_margin(const Wcs& wcs, double rm, double margin_px) {
  Vec3 c = unit(wcs.ra, wcs.dec);
  double radius;
  if (is_tan(wcs.proj)) {
    Vec2 corners[4] = {{-margin_px, -margin_px}, {w + margin_px, -margin_px}, {-margin_px, h + margin_px}, {w + margin_px, h + margin_px}};
    radius = 0.0;
    for (auto& q : corners) {
      Vec2 sky = wcs.pixel_to_sky(q[0], q[1]);
      radius = fmax_r(radius, angle(c, unit(sky[0], sky[1])));
    }
  } else {
    radius = frame_radius(wcs, c, margin_px);
  }
  if (!std::isfinite(radius) || radius > max_frame_radius(wcs, 1.3)) return std::nullopt;
  Pairs out;
  for (auto& st : catalog_cone(c, radius * 1.02, rm)) {
    auto [ra, dec] = radec(st.v);
    out.push_back({ra, dec, 0.0, 0.0});
  }
  return out;
}

std::optional<Pairs> Solver::catalog_in_frame(const Wcs& wcs, double rm) {
  Vec3 c = unit(wcs.ra, wcs.dec);
  double radius;
  if (is_tan(wcs.proj)) {
    Vec2 corners[4] = {{0.0, 0.0}, {w, 0.0}, {0.0, h}, {w, h}};
    radius = 0.0;
    for (auto& q : corners) {
      Vec2 sky = wcs.pixel_to_sky(q[0], q[1]);
      radius = fmax_r(radius, angle(c, unit(sky[0], sky[1])));
    }
  } else {
    radius = frame_radius(wcs, c, 0.0);
  }
  if (!std::isfinite(radius) || radius > max_frame_radius(wcs, 1.2)) return std::nullopt;
  auto stars = catalog_cone(c, radius * 1.02, rm);
  auto projector = geometry::prepare(wcs);
  Pairs out;
  for (auto& st : stars) {
    auto [ra, dec] = radec(st.v);
    auto px = projector.sky_to_pixel(ra, dec);
    if (px && visible(*px)) out.push_back({ra, dec, (*px)[0], (*px)[1]});
  }
  return out;
}

size_t Solver::inliers_at(const Wcs& wcs, double tol) {
  double s = to_radians(wcs.scale_arcsec()) / 3600.0;
  double area = s * s * open_area;
  uint32_t mr = idx.has_deep() ? 8u : 7u;
  while (mr > 1 && area / (PI * powi(idx.level_radius_m(mr), 2)) > 1500.0) --mr;
  auto cat = catalog_in_frame(wcs, idx.level_radius_m(mr));
  if (!cat) return 0;
  size_t n_img = std::min(std::max<size_t>(cat->size() * 3, 50), pts.size());
  if (!inlier_grid || inlier_grid->first != n_img) {
    std::vector<uint32_t> ids(n_img);
    for (size_t i = 0; i < n_img; ++i) ids[i] = uint32_t(i);
    inlier_grid = std::make_pair(n_img, select::Grid2::build(pts, ids, w, h, 2.0));
  }
  return count_matches(inlier_grid->second, *cat, tol, wcs).n;
}

std::optional<Solution> Solver::verify_wcs_within(const Wcs& wcs0, size_t cluster, double min_r1) {
  catalog_cache.clear();
  inlier_grid.reset();
  double s = to_radians(wcs0.scale_arcsec()) / 3600.0;
  double area = s * s * open_area;
  uint32_t mv = idx.has_deep() ? 8u : 7u;
  while (mv > 1 && area / (PI * powi(idx.level_radius_m(mv), 2)) > 250.0) --mv;
  double rm = idx.level_radius_m(mv);
  double thr_px = 0.8 * rm / s;
  std::vector<uint32_t> img;
  for (uint32_t i = 0; i < pts.size(); ++i)
    if (d[i] >= thr_px) img.push_back(i);
  auto catr = catalog_in_frame(wcs0, rm);
  if (!catr) return std::nullopt;
  Pairs cat = std::move(*catr);
  if (cat.size() < 5 || img.size() < 5) return std::nullopt;
  double r1 = fmax_r(fmax_r(0.004 * fmax_r(w, h), 4.0), min_r1);
  auto g = select::Grid2::build(pts, img, w, h, 2.0);
  size_t k1 = count_matches(g, cat, r1, wcs0).n;
  double p1 = fmin_r(double(img.size()) * PI * r1 * r1 / open_area, 0.999);
  double lt = log_binom_tail(cat.size(), k1, p1);
  bool curved_patch = cfg.allow_panorama && !cfg.windows && to_degrees(s) * fmax_r(w, h) > 8.0;
  double screen_threshold = -14.0 - (curved_patch ? 0.0 : trials_ln());
  if (lt > screen_threshold || k1 < 5) return std::nullopt;
  stats.stage2 += 1;
  Vec2 crpix = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  struct Identified {
    Wcs wcs;
    double radius;
    Matches matched;
    double ll;
    bool curved;
  };
  std::optional<Identified> identified;
  int orders = curved_patch ? 2 : 1;
  for (int order = 1; order <= orders; ++order) {
    if (order == 2) {
      if (elapsed_ms() >= cfg.deadline_ms) break;
      stats.verifications += 1;
    }
    Wcs wcs = wcs0;
    double radius = r1;
    bool fitted = true;
    for (int it = 0; it < 3; ++it) {
      auto m = count_matches(g, cat, radius, wcs).m;
      if (m.size() < (order == 2 ? 24u : 6u)) {
        fitted = false;
        break;
      }
      Pairs prs = pairs_of(m);
      Vec2 c = wcs.pixel_to_sky(crpix[0], crpix[1]);
      std::optional<wcsfit::RobustFit> candidate;
      if (order == 1) {
        candidate = wcsfit::robust_fit(prs, crpix, c, 1);
      } else {
        auto sel = wcsfit::select_order(prs, crpix, c, 2);
        if (sel && sel->wcs.sip && sel->wcs.sip->order == 2) candidate = wcsfit::RobustFit{std::move(sel->wcs), std::move(sel->keep)};
      }
      if (!candidate) {
        fitted = false;
        break;
      }
      auto r = wcsfit::residuals(candidate->wcs, prs);
      std::vector<double> inl;
      for (size_t i = 0; i < r.size(); ++i)
        if (candidate->keep[i]) inl.push_back(r[i]);
      double rms = wcsfit::rms(inl);
      wcs = std::move(candidate->wcs);
      radius = clampd(4.0 * rms, 3.0, radius);
    }
    if (!fitted) continue;
    auto mr = count_matches(g, cat, radius, wcs);
    size_t k = mr.n;
    double lambda = double(img.size()) * double(cat.size()) * PI * powi(radius, 2) / open_area;
    double ll = poisson_log_odds(k, lambda);
    double xmin = std::numeric_limits<double>::max(), xmax = std::numeric_limits<double>::lowest(), ymin = xmin, ymax = xmax;
    for (auto& [i, sky] : mr.m) {
      (void)sky;
      Vec2 q = pts[i];
      xmin = fmin_r(xmin, q[0]);
      xmax = fmax_r(xmax, q[0]);
      ymin = fmin_r(ymin, q[1]);
      ymax = fmax_r(ymax, q[1]);
    }
    bool spread_ok = (xmax - xmin) >= 0.2 * w && (ymax - ymin) >= 0.2 * h;
    double threshold = cfg.min_log_odds + trials_ln();
    if (ll >= threshold && k >= (order == 2 ? 24u : 8u) && spread_ok) {
      identified = Identified{wcs, radius, std::move(mr.m), ll, order == 2};
      break;
    }
  }
  if (!identified) return std::nullopt;
  Wcs wcs = identified->wcs;
  double radius = identified->radius;
  double ll = identified->ll;
  bool curved_recovery = identified->curved;
  Pairs id_pairs = pairs_of(identified->matched);
  double id_rms = wcsfit::rms(wcsfit::residuals(wcs, id_pairs));
  Refined rf = refine(wcs, radius);
  Pairs pairs;
  double fit_rms;
  if (std::isfinite(rf.rms) && rf.pairs.size() >= 8) {
    size_t n_ref = inliers_at(rf.wcs, 2.5);
    size_t n_id = inliers_at(wcs, 2.5);
    if (double(n_ref) >= 0.9 * double(n_id)) {
      wcs = std::move(rf.wcs);
      pairs = std::move(rf.pairs);
      fit_rms = rf.rms;
    } else {
      pairs = std::move(id_pairs);
      fit_rms = id_rms;
    }
  } else {
    pairs = std::move(id_pairs);
    fit_rms = id_rms;
  }
  if (!std::isfinite(fit_rms) || fit_rms > 4.0) return std::nullopt;
  if (curved_recovery && !wide::mapping_holds(*this, wcs)) return std::nullopt;
  auto refraction = measure_refraction(wcs, pairs);
  Vec2 center = wcs.pixel_to_sky(crpix[0], crpix[1]);
  double det = wcs.determinant();
  Solution sol;
  sol.status = "solved";
  sol.width = uint32_t(w);
  sol.height = uint32_t(h);
  sol.ra = center[0];
  sol.dec = center[1];
  sol.scale_arcsec = wcs.scale_arcsec();
  sol.rotation_deg = wcs.rotation_deg();
  sol.parity = det > 0.0 ? 1 : -1;
  sol.fov_width_deg = fov_width(wcs);
  sol.n_matches = pairs.size();
  sol.n_catalog = cat.size();
  sol.rms_px = fit_rms;
  sol.log_odds = ll;
  sol.cluster_anchors = cluster;
  sol.verify_level = mv;
  if (cfg.keep_matches) sol.matches = match_out(pairs, wcs);
  sol.margin_nats = NaN;
  sol.n_candidates = 1;
  sol.refraction = refraction;
  sol.sky_extent = sky_extent();
  sol.wcs = std::move(wcs);
  return sol;
}

std::optional<Solver::Refit> Solver::refit_linear(const Wcs& prior, const Pairs& pairs) const {
  Vec2 crpix = prior.crpix;
  Pairs und;
  for (const auto& p : pairs) {
    double u = p[0] - crpix[0], v = p[1] - crpix[1];
    double du = 0.0, dv = 0.0;
    if (prior.sip) std::tie(du, dv) = prior.sip->eval(u, v);
    und.push_back({crpix[0] + u + du, crpix[1] + v + dv, p[2], p[3]});
  }
  Vec2 c = prior.pixel_to_sky(crpix[0], crpix[1]);
  auto fit = wcsfit::robust_fit_wmp(und, {}, crpix, c, 1, 0, geometry::Projection(prior.proj, prior.native_pole));
  if (!fit) return std::nullopt;
  Wcs wcs = std::move(fit->wcs);
  wcs.sip = prior.sip;
  Pairs inl;
  for (size_t i = 0; i < pairs.size(); ++i)
    if (fit->keep[i]) inl.push_back(pairs[i]);
  double rms = wcsfit::rms(wcsfit::residuals(wcs, inl));
  if (!std::isfinite(rms)) return std::nullopt;
  return Refit{std::move(wcs), rms, std::move(inl)};
}

std::optional<Vec2> Solver::offset_vote(const Wcs& prior, const Pairs& cat, const select::Grid2& g, double radius) const {
  const double BIN = 3.0;
  auto key = [](int64_t bx, int64_t by) { return (uint64_t(bx + (int64_t(1) << 20)) << 32) | uint64_t(by + (int64_t(1) << 20)); };
  struct V {
    uint32_t n;
    double x, y;
  };
  std::unordered_map<uint64_t, V> votes;
  std::vector<std::pair<double, uint32_t>> nn;
  uint32_t total = 0;
  for (const auto& c : cat) {
    auto p = prior.sky_to_pixel_fast(c[0], c[1]);
    if (!p) continue;
    g.knn(pts, *p, 8, UINT32_MAX, nn);
    for (auto& [d2, i] : nn) {
      if (d2 > radius * radius) break;
      Vec2 q = pts[i];
      double dx = q[0] - (*p)[0], dy = q[1] - (*p)[1];
      V& e = votes.try_emplace(key(sat_cast<int64_t>(std::floor(dx / BIN)), sat_cast<int64_t>(std::floor(dy / BIN))), V{0, 0.0, 0.0}).first->second;
      e.n += 1;
      e.x += dx;
      e.y += dy;
      total += 1;
    }
  }
  // deterministic order: ascending key
  std::vector<uint64_t> keys;
  keys.reserve(votes.size());
  for (auto& kv : votes) keys.push_back(kv.first);
  std::sort(keys.begin(), keys.end());
  V best{0, 0.0, 0.0};
  for (uint64_t k : keys) {
    int64_t bx = int64_t(k >> 32) - (int64_t(1) << 20), by = int64_t(k & 0xffffffffull) - (int64_t(1) << 20);
    V sum{0, 0.0, 0.0};
    for (int64_t oy = -1; oy <= 1; ++oy)
      for (int64_t ox = -1; ox <= 1; ++ox) {
        auto f = votes.find(key(bx + ox, by + oy));
        if (f != votes.end()) sum = V{sum.n + f->second.n, sum.x + f->second.x, sum.y + f->second.y};
      }
    if (sum.n > best.n) best = sum;
  }
  double background = double(total) * 9.0 * BIN * BIN / (PI * radius * radius);
  if (best.n >= 8 && double(best.n) >= 5.0 * fmax_r(background, 1.0)) return Vec2{best.x / double(best.n), best.y / double(best.n)};
  return std::nullopt;
}

std::optional<Solution> Solver::verify_tracked(const Wcs& prior_in, double search_px) {
  const Wcs* prior = &prior_in;
  double s = to_radians(prior->scale_arcsec()) / 3600.0;
  double area = s * s * open_area;
  uint32_t mv = idx.has_deep() ? 8u : 7u;
  while (mv > 1 && area / (PI * powi(idx.level_radius_m(mv), 2)) > 250.0) --mv;
  double rm = idx.level_radius_m(mv);
  double thr_px = 0.8 * rm / s;
  std::vector<uint32_t> img;
  for (uint32_t i = 0; i < pts.size(); ++i)
    if (d[i] >= thr_px) img.push_back(i);
  auto catr = catalog_in_frame(*prior, rm);
  if (!catr) return std::nullopt;
  Pairs cat = std::move(*catr);
  if (cat.size() < 5 || img.size() < 5) return std::nullopt;
  auto g = select::Grid2::build(pts, img, w, h, 2.0);
  double base = fmax_r(0.004 * fmax_r(w, h), 4.0);
  double r1 = fmax_r(base, search_px);
  std::optional<Wcs> shifted;
  if (search_px > 1.5 * base) {
    if (auto o = offset_vote(*prior, cat, g, search_px)) {
      Vec2 c = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
      Vec2 sky = prior->pixel_to_sky(c[0] - (*o)[0], c[1] - (*o)[1]);
      Wcs sw = *prior;
      sw.ra = sky[0];
      sw.dec = sky[1];
      shifted = std::move(sw);
      r1 = 1.5 * base;
    }
  }
  if (shifted) prior = &*shifted;
  auto mr = count_matches(g, cat, r1, *prior);
  size_t k1 = mr.n;
  Matches m = std::move(mr.m);
  double p1 = fmin_r(double(img.size()) * PI * r1 * r1 / open_area, 0.999);
  double lt = log_binom_tail(cat.size(), k1, p1);
  if (lt > -14.0 || k1 < 5) return std::nullopt;
  Wcs wcs = *prior;
  double radius = r1;
  for (int it = 0; it < 3; ++it) {
    if (m.size() < 6) return std::nullopt;
    auto rf = refit_linear(*prior, pairs_of(m));
    if (!rf) return std::nullopt;
    wcs = std::move(rf->wcs);
    radius = clampd(4.0 * rf->rms, 2.0, radius);
    m = count_matches(g, cat, radius, wcs).m;
  }
  size_t k = m.size();
  double lambda = double(img.size()) * double(cat.size()) * PI * powi(radius, 2) / open_area;
  double kf = double(k);
  double ll = kf > lambda ? kf * std::log(kf / lambda) - kf + lambda : 0.0;
  double xmin = std::numeric_limits<double>::max(), xmax = std::numeric_limits<double>::lowest(), ymin = xmin, ymax = xmax;
  for (auto& [i, sky] : m) {
    (void)sky;
    Vec2 q = pts[i];
    xmin = fmin_r(xmin, q[0]);
    xmax = fmax_r(xmax, q[0]);
    ymin = fmin_r(ymin, q[1]);
    ymax = fmax_r(ymax, q[1]);
  }
  bool spread_ok = (xmax - xmin) >= 0.2 * w && (ymax - ymin) >= 0.2 * h;
  if (ll < cfg.min_log_odds || k < 8 || !spread_ok) return std::nullopt;
  Pairs prs = pairs_of(m);
  double fit_rms = wcsfit::rms(wcsfit::residuals(wcs, prs));
  uint32_t mr_level = idx.has_deep() ? 8u : 7u;
  while (mr_level > 1 && area / (PI * powi(idx.level_radius_m(mr_level), 2)) > 1000.0) --mr_level;
  if (mr_level > mv) {
    if (auto deep = catalog_in_frame(wcs, idx.level_radius_m(mr_level))) {
      size_t n_img = std::min(std::max<size_t>(deep->size() * 3, 50), pts.size());
      std::vector<uint32_t> ids(n_img);
      for (size_t i = 0; i < n_img; ++i) ids[i] = uint32_t(i);
      auto gd = select::Grid2::build(pts, ids, w, h, 2.0);
      auto md = count_matches(gd, *deep, clampd(3.0 * fit_rms, 1.5, radius), wcs).m;
      if (md.size() >= 20) {
        auto rf = refit_linear(wcs, pairs_of(md));
        if (rf && rf->inl.size() >= prs.size()) {
          wcs = std::move(rf->wcs);
          fit_rms = rf->rms;
          prs = std::move(rf->inl);
        }
      }
    }
  }
  Vec2 crpix = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  Vec2 center = wcs.pixel_to_sky(crpix[0], crpix[1]);
  double det = wcs.determinant();
  Solution sol;
  sol.status = "solved";
  sol.width = uint32_t(w);
  sol.height = uint32_t(h);
  sol.ra = center[0];
  sol.dec = center[1];
  sol.scale_arcsec = wcs.scale_arcsec();
  sol.rotation_deg = wcs.rotation_deg();
  sol.parity = det > 0.0 ? 1 : -1;
  sol.fov_width_deg = fov_width(wcs);
  sol.n_matches = prs.size();
  sol.n_catalog = cat.size();
  sol.rms_px = fit_rms;
  sol.log_odds = ll;
  sol.cluster_anchors = 0;
  sol.verify_level = mv;
  sol.margin_nats = NaN;
  sol.n_candidates = 1;
  sol.sky_extent = sky_extent();
  if (cfg.keep_matches) sol.matches = match_out(prs, wcs);
  sol.wcs = std::move(wcs);
  return sol;
}

Refined Solver::refine(Wcs wcs0, double start_radius) {
  Vec2 crpix = {(w - 1.0) / 2.0, (h - 1.0) / 2.0};
  double s = to_radians(wcs0.scale_arcsec()) / 3600.0;
  double area = s * s * open_area;
  uint32_t mr = idx.has_deep() ? 8u : 7u;
  while (mr > 1 && area / (PI * powi(idx.level_radius_m(mr), 2)) > 1500.0) --mr;
  double margin = 0.15 * fmax_r(w, h);
  auto catr = catalog_in_frame_margin(wcs0, idx.level_radius_m(mr), margin);
  if (!catr) return Refined{wcs0, {}, INF, 0};
  Pairs cat = std::move(*catr);
  size_t n_img = std::min(std::max<size_t>(cat.size() * 3, 50), pts.size());
  std::vector<uint32_t> ids(n_img);
  for (size_t i = 0; i < n_img; ++i) ids[i] = uint32_t(i);
  auto g = select::Grid2::build(pts, ids, w, h, 2.0);
  double fov = to_degrees(s) * fmax_r(w, h);
  uint32_t max_order = fov > 20.0 ? 5 : (fov > 3.0 ? 4 : 3);
  double half_diag = 0.5 * std::hypot(w, h);
  double sep = std::sqrt(open_area / double(std::max<size_t>(cat.size(), 1)));
  double rcap = clampd(0.3 * sep, 1.5, 25.0);
  std::vector<double> zones = fov > 8.0 ? std::vector<double>{0.45, 0.75, 1.0, 1.2} : std::vector<double>{1.2, 1.2};
  Wcs wcs = std::move(wcs0);
  double rms = clampd(start_radius, 2.0, 8.0) / 4.0;
  Refined best{wcs, {}, INF, 0};
  std::pair<uint32_t, uint32_t> model{1, 0};
  std::vector<double> best_weights;
  uint32_t radial_max = fov > 8.0 ? 7 : 0;
  size_t total = zones.size() + 2;
  for (size_t step = 0; step < total; ++step) {
    double zf = zones[std::min(step, zones.size() - 1)];
    double zr = zf * half_diag;
    auto projector = geometry::prepare(wcs);
    std::vector<std::pair<Vec2, Vec2>> sub;
    for (const auto& c : cat) {
      auto px = projector.sky_to_pixel_fast(c[0], c[1]);
      if (!px) continue;
      double r = std::hypot((*px)[0] - crpix[0], (*px)[1] - crpix[1]);
      if (r <= zr && (*px)[0] > -margin && (*px)[1] > -margin && (*px)[0] < w + margin && (*px)[1] < h + margin &&
          (!cfg.mask || !cfg.mask->is_blocked((*px)[0], (*px)[1])))
        sub.push_back({{c[0], c[1]}, *px});
    }
    double extra = step < zones.size() ? 0.004 * zr : 0.0;
    double radius = clampd(4.0 * rms + extra, 1.5, rcap);
    auto mres = match_projected(pts, g, sub, radius);
    Pairs prs = pairs_of(mres.m);
    std::vector<double> wts = weights(mres.m);
    if (prs.size() < 8) {
      if (step == 0) return best;
      continue;
    }
    Vec2 c = wcs.pixel_to_sky(crpix[0], crpix[1]);
    uint32_t cap;
    size_t np = prs.size();
    if (np <= 29) cap = 1;
    else if (np <= 59) cap = 2;
    else if (np <= 99) cap = 3;
    else if (np <= 199) cap = 4;
    else cap = 5;
    cap = std::min(cap, max_order);
    uint32_t zone_cap = zf < 0.6 ? 0 : (zf < 0.9 ? 5 : 7);
    auto radial_for = [&](uint32_t o) -> uint32_t {
      for (uint32_t r : {7u, 5u})
        if (r <= std::min(radial_max, zone_cap) && prs.size() >= 6 * wcsfit::model_columns(o, r)) return r;
      return 0;
    };
    std::optional<std::tuple<Wcs, std::vector<bool>, std::pair<uint32_t, uint32_t>>> fitted;
    if (step == zones.size()) {
      auto f = wcsfit::select_model_w(prs, wts, crpix, c, cap, radial_for(1));
      if (f) fitted = std::make_tuple(std::move(f->wcs), std::move(f->keep), f->model);
    } else if (step > zones.size()) {
      uint32_t o = std::min(model.first, cap);
      uint32_t r = model.second > 0 ? std::min(radial_for(o), model.second) : 0;
      auto f = wcsfit::robust_fit_wm(prs, wts, crpix, c, o, r);
      if (f) fitted = std::make_tuple(std::move(f->wcs), std::move(f->keep), std::make_pair(o, r));
    } else if (fov > 8.0) {
      uint32_t o = std::min<uint32_t>(cap, 3);
      uint32_t r = radial_for(o);
      auto f = wcsfit::robust_fit_wm(prs, wts, crpix, c, o, r);
      if (f) fitted = std::make_tuple(std::move(f->wcs), std::move(f->keep), std::make_pair(o, r));
    } else {
      auto f = wcsfit::robust_fit_w(prs, wts, crpix, c, 1);
      if (f) fitted = std::make_tuple(std::move(f->wcs), std::move(f->keep), std::make_pair(1u, 0u));
    }
    if (!fitted) continue;
    auto& [fit, keep, m] = *fitted;
    Pairs inl;
    for (size_t i = 0; i < prs.size(); ++i)
      if (keep[i]) inl.push_back(prs[i]);
    double r = wcsfit::rms(wcsfit::residuals(fit, inl));
    if (!std::isfinite(r)) continue;
    model = m;
    uint32_t order = fit.sip ? fit.sip->order : 1;
    wcs = fit;
    rms = r;
    best_weights.clear();
    for (size_t i = 0; i < keep.size(); ++i)
      if (keep[i]) best_weights.push_back(i < wts.size() ? wts[i] : 1.0);
    best = Refined{wcs, std::move(inl), r, order};
  }
  if (fov > 8.0 && best.rms < 2.0) {
    auto lens = wcsfit::select_lens(best.pairs, best_weights, best.wcs, model, w, h);
    if (lens) {
      double e = wcsfit::rms(wcsfit::residuals(*lens, best.pairs));
      if (std::isfinite(e) && e <= fmax_r(1.1 * best.rms, 0.3)) best = Refined{std::move(*lens), best.pairs, e, 9};
    }
  }
  return best;
}

// ---------- entry points ----------
static Solution unsolved(const StarField& field, const SolveStats& stats) {
  Solution s = blank_solution("unsolved", field.width, field.height);
  s.stats = stats;
  return s;
}

Solution track_field(const Index& idx, const StarField& field, const Wcs& prior, double search_px, bool refine,
                     const SolveConfig& cfg) {
  auto t0 = Clock::now();
  auto ctx = make_context(idx, field, cfg, t0);
  ctx->stats.select_ms = ctx->elapsed_ms();
  ctx->stats.verifications = 1;
  auto fast = ctx->verify_tracked(prior, search_px);
  std::optional<Solution> sol;
  if (fast && refine && (!is_tan(prior.proj) || fast->fov_width_deg > 60.0)) {
    auto r = wide::refit_whole(*ctx, *fast);
    sol = r ? std::move(r) : std::move(fast);
  } else if (fast && refine) {
    Wcs tight = *fast->wcs;
    auto r = ctx->verify_wcs_within(tight, 0, 0.0);
    sol = r ? std::move(r) : std::move(fast);
  } else {
    sol = std::move(fast);
  }
  ctx->stats.verify_ms = ctx->elapsed_ms() - ctx->stats.select_ms;
  ctx->stats.total_ms = ctx->elapsed_ms();
  if (sol) {
    sol->stats = ctx->stats;
    return std::move(*sol);
  }
  return unsolved(field, ctx->stats);
}

Solution solve_windows_only(const Index& idx, const StarField& field, const SolveConfig& cfg, size_t spent_verifications) {
  auto t0 = Clock::now();
  SolveConfig c = cfg;
  c.hint.reset();
  c.extra_trials = cfg.extra_trials + spent_verifications;
  auto r = wide::solve_windows_within(idx, field, c, t0, SolveStats{}, true);
  if (r) return std::move(*r);
  return blank_solution("unsolved", field.width, field.height);
}

static Solution solve_field_single(const Index& idx, const StarField& field, const SolveConfig& cfg);
static Solution solve_field_blind(const Index& idx, const StarField& field, const SolveConfig& cfg);

static SolveConfig hinted_config(const Index& idx, const StarField& field, const SolveConfig& cfg, const SolveHint& hint) {
  auto fov = hint.fov_deg(field.width, field.height);
  bool deep = cfg.mode == SolveMode::Auto && cfg.deep_index && fov && *fov < DEEP_FOV_DEG;
  bool deep_bounds = deep || cfg.mode == SolveMode::Deep || idx.manifest.profile == "deep";
  double lower = deep_bounds ? 0.03 : cfg.min_fov_deg;
  double upper = deep_bounds ? 0.5 : cfg.max_fov_deg;
  SolveConfig c = cfg;
  c.hint = hint;
  c.mode = deep ? SolveMode::Deep : cfg.mode;
  c.min_fov_deg = fov ? fmax_r(*fov / 2.0, lower) : lower;
  c.max_fov_deg = fov ? fmin_r(*fov * 2.0, upper) : upper;
  c.deadline_ms = fmin_r(cfg.deadline_ms * 0.25, 200.0);
  c.windows = false;
  return c;
}

Solution solve_field(const Index& idx, const StarField& field, const SolveConfig& cfg) {
  if (!cfg.hint) return solve_field_blind(idx, field, cfg);
  auto hint = cfg.hint->for_frame(field.width, field.height);
  if (!hint) {
    SolveConfig c = cfg;
    c.hint.reset();
    return solve_field_blind(idx, field, c);
  }
  if (cfg.deadline_ms < 20.0) {
    SolveConfig c = cfg;
    c.hint.reset();
    return solve_field_blind(idx, field, c);
  }
  auto t0 = Clock::now();
  SolveConfig c = hinted_config(idx, field, cfg, *hint);
  Solution first = solve_field_blind(idx, field, c);
  first.stats.hint_used = true;
  if (first.status != "unsolved") return first;
  SolveConfig c2 = cfg;
  c2.hint.reset();
  c2.deadline_ms = fmax_r(cfg.deadline_ms - ms_since(t0), 0.0);
  c2.extra_trials = cfg.extra_trials + first.stats.verifications;
  Solution sol = solve_field_blind(idx, field, c2);
  bool deep_search = sol.stats.deep_search;
  wide::accumulate(sol.stats, first.stats);
  sol.stats.deep_search = deep_search;
  sol.stats.hint_fallback = true;
  sol.stats.total_ms = ms_since(t0);
  return sol;
}

static Solution solve_field_blind(const Index& idx, const StarField& field, const SolveConfig& cfg) {
  bool direct = idx.manifest.profile == "deep";
  const Index* deep = direct ? &idx : cfg.deep_index.get();
  auto run_deep = [&](const Index& index, double deadline_ms, size_t trials) {
    SolveConfig c = cfg;
    c.deadline_ms = deadline_ms;
    c.min_fov_deg = cfg.hint ? fmax_r(cfg.min_fov_deg, 0.03) : 0.03;
    c.max_fov_deg = cfg.hint ? fmin_r(cfg.max_fov_deg, 0.5) : 0.5;
    c.mask.reset();
    c.windows = false;
    c.wide_index.reset();
    c.deep_index.reset();
    c.mode = SolveMode::Standard;
    c.extra_trials = cfg.extra_trials + trials;
    Solution sol = solve_field_single(index, field, c);
    sol.stats.deep_search = true;
    return sol;
  };
  if (direct || cfg.mode == SolveMode::Deep) {
    if (!deep) return blank_solution("unsolved", field.width, field.height);
    return run_deep(*deep, cfg.deadline_ms, 0);
  }
  if (!deep || cfg.mode != SolveMode::Auto) return solve_field_single(idx, field, cfg);
  auto t0 = Clock::now();
  SolveConfig sc = cfg;
  sc.deadline_ms = cfg.deadline_ms * 0.65;
  sc.mode = SolveMode::Standard;
  sc.deep_index.reset();
  Solution standard = solve_field_single(idx, field, sc);
  if (standard.status != "unsolved") return standard;
  double remaining = cfg.deadline_ms - ms_since(t0);
  if (remaining <= 5.0) return standard;
  Solution sol = run_deep(*deep, remaining, standard.stats.verifications);
  wide::accumulate(sol.stats, standard.stats);
  sol.stats.total_ms = ms_since(t0);
  return sol;
}

static Solution solve_field_single(const Index& idx, const StarField& field, const SolveConfig& cfg) {
  auto t0 = Clock::now();
  double w = double(field.width), h = double(field.height);
  auto ctxp = make_context(idx, field, cfg, t0);
  Solver& ctx = *ctxp;
  ctx.stats.select_ms = ctx.elapsed_ms();
  auto rungs = select::rung_order(ctx.d, cfg.target_pool, cfg.min_pool, cfg.max_pool);
  std::vector<fingerprint::KeyVariant> variants;
  variants.reserve(128);
  std::unordered_set<uint64_t> seen;
  std::vector<std::pair<double, uint32_t>> nbuf;
  struct Pass {
    bool exhaustive;
    size_t n_refs, n_rest;
  };
  std::vector<Pass> passes = {{false, 2, 4}, {false, cfg.n_refs, cfg.n_rest}};
  if (cfg.exhaustive) passes.push_back({true, cfg.n_refs, cfg.n_rest});
  std::optional<Solution> result;
  double deadline = (cfg.windows && cfg.mask && cfg.mask->is_landscape()) ? cfg.deadline_ms * wide::LANDSCAPE_WHOLE_SHARE : cfg.deadline_ms;
  bool done = false;
  for (const Pass& pass : passes) {
    if (done) break;
    ctx.n_refs = pass.n_refs;
    ctx.n_rest = pass.n_rest;
    if (pass.exhaustive) ctx.stats.exhaustive_pass = true;
    for (int k : rungs) {
      if (ctx.elapsed_ms() > deadline) {
        done = true;
        break;
      }
      ctx.stats.rungs_tried += 1;
      double r = select::rung_radius(k);
      std::vector<uint32_t> pool;
      for (uint32_t i = 0; i < ctx.pts.size(); ++i)
        if (ctx.d[i] >= r) pool.push_back(i);
      if (pool.size() < 7) continue;
      auto grid = select::Grid2::build(ctx.pts, pool, w, h, 2.0);
      std::vector<uint32_t> anchors;
      for (uint32_t i : pool) {
        double di = ctx.d[i];
        if (pass.exhaustive ? di < 2.0 * r : di >= 2.0 * r) anchors.push_back(i);
      }
      for (uint32_t a : anchors) {
        grid.knn(ctx.pts, ctx.pts[a], 6, a, nbuf);
        if (nbuf.size() < 4) continue;
        if (ctx.edge[a] < std::sqrt(nbuf[3].first)) continue;
        ctx.stats.anchors += 1;
        std::vector<std::pair<double, uint32_t>> nb = nbuf;
        if (auto sol = ctx.process_anchor(a, nb, r, variants, seen)) {
          result = std::move(sol);
          done = true;
          break;
        }
        if (ctx.continuation_over()) {
          done = true;
          break;
        }
        if (ctx.elapsed_ms() > deadline) {
          done = true;
          break;
        }
      }
      if (done) break;
      if (ctx.continuing()) {
        done = true;
        break;
      }
    }
  }
  if (ctx.continuing()) ctx.sweep_pending_bins();
  if (!result) result = ctx.conclude();
  ctx.stats.bins = ctx.bins.size();
  ctx.stats.total_ms = ctx.elapsed_ms();
  ctx.stats.probe_ms = ctx.stats.total_ms - ctx.stats.select_ms - ctx.stats.verify_ms;
  Solution s = result ? std::move(*result) : blank_solution("unsolved", field.width, field.height);
  s.stats = ctx.stats;
  if (s.status == "unsolved" && cfg.windows) {
    if (auto ws = wide::solve_windows(idx, field, cfg, t0, s.stats)) return std::move(*ws);
  }
  return s;
}

}  // namespace solve
}  // namespace skyshot
