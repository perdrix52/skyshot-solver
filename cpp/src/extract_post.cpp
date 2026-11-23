// Duplicate merging, PSF-lobe merging, vetting (vet.rs) and line removal (lines.rs).
#include <algorithm>
#include <cstdlib>
#include <deque>

#include "extract_internal.hpp"
#include "skyshot/parallel.hpp"

namespace skyshot::extract_detail {

// =============================================================================================
// merge_duplicates
// =============================================================================================
struct Rel {
  uint32_t other;
  bool dup;
  float dist;
};

void merge_duplicates(const Ctx& ctx, std::vector<Det>& dets) {
  size_t n = dets.size();
  if (n < 2) return;
  double ff = fmax_r(double(ctx.fwhm_frame), 2.0);
  double r_dup = fmax_r(1.6 * ff, 3.0);
  double r_blend = fmax_r(2.0 * ff, 4.0);
  double cell = r_blend;
  size_t gw = sat_cast<size_t>(double(ctx.w) / cell) + 1, gh = sat_cast<size_t>(double(ctx.h) / cell) + 1;
  auto cell_of = [&](double x, double y) {
    size_t cx = std::min(sat_cast<size_t>(fmax_r(x / cell, 0.0)), gw - 1);
    size_t cy = std::min(sat_cast<size_t>(fmax_r(y / cell, 0.0)), gh - 1);
    return std::make_pair(cx, cy);
  };
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    bool ca = dets[a].core > 0.0f, cb = dets[b].core > 0.0f;
    if (ca != cb) return ca;  // cores first
    int c = total_cmp(dets[b].h, dets[a].h);
    if (c != 0) return c < 0;
    return a < b;
  });
  std::vector<uint32_t> rank(n, 0);
  for (size_t r = 0; r < n; ++r) rank[order[r]] = uint32_t(r);
  std::vector<uint32_t> head(gw * gh, UINT32_MAX), next(n, UINT32_MAX);
  for (size_t i = 0; i < n; ++i) {
    auto [gx, gy] = cell_of(dets[i].x, dets[i].y);
    next[i] = head[gy * gw + gx];
    head[gy * gw + gx] = uint32_t(i);
  }
  std::vector<std::vector<uint32_t>> discs, big;
  auto bright = [&](const Det& d) { return d.h >= 0.1f * (ctx.sat_level - d.bg); };
  for (size_t i = 0; i < n; ++i) {
    const Det& d = dets[i];
    std::vector<std::vector<uint32_t>>* map;
    double r;
    if (d.core > 0.0f) {
      map = &discs;
      r = double(d.core) + 2.0;
    } else if (d.fwhm > 1.6 * ff && bright(d)) {
      map = &big;
      r = std::min(1.2 * d.fwhm, 40.0);
    } else {
      continue;
    }
    if (map->empty()) map->resize(gw * gh);
    auto c0 = cell_of(d.x - r, d.y - r);
    auto c1 = cell_of(d.x + r, d.y + r);
    for (size_t yy = c0.second; yy <= c1.second; ++yy)
      for (size_t xx = c0.first; xx <= c1.first; ++xx) (*map)[yy * gw + xx].push_back(uint32_t(i));
  }
  std::vector<std::vector<Rel>> rels(n);
  parallel_for(n, [&](size_t i) {
    const Det& d = dets[i];
    auto& acc = rels[i];
    auto [gx, gy] = cell_of(d.x, d.y);
    if (!discs.empty()) {
      for (uint32_t j : discs[gy * gw + gx]) {
        const Det& s = dets[j];
        if (rank[j] < rank[i] && (dist(s.x - d.x, s.y - d.y) < double(s.core) + 2.0 || (s.px == d.px && s.py == d.py)))
          acc.push_back({j, true, 0.0f});
      }
    }
    if (!big.empty()) {
      for (uint32_t j : big[gy * gw + gx]) {
        const Det& o = dets[j];
        double dd = dist(o.x - d.x, o.y - d.y);
        if (rank[j] < rank[i] && dd >= r_dup && dd < std::min(1.2 * o.fwhm, 40.0)) {
          std::pair<double, double> pa{double(o.px), double(o.py)}, pb{double(d.px), double(d.py)};
          if (!ctx.separate(pa, pb, d.bg, d.sig_n)) acc.push_back({j, true, float(dd)});
        }
      }
    }
    for (size_t yy = gy >= 1 ? gy - 1 : 0; yy <= std::min(gy + 1, gh - 1); ++yy) {
      for (size_t xx = gx >= 1 ? gx - 1 : 0; xx <= std::min(gx + 1, gw - 1); ++xx) {
        uint32_t j = head[yy * gw + xx];
        while (j != UINT32_MAX) {
          const Det& o = dets[j];
          if (rank[j] < rank[i]) {
            double dd = dist(o.x - d.x, o.y - d.y);
            if ((o.px == d.px && o.py == d.py) || dd < 1.0) {
              acc.push_back({j, true, float(dd)});
            } else if (dd < r_dup) {
              std::pair<double, double> pa{double(o.px), double(o.py)}, pb{double(d.px), double(d.py)};
              bool separate = ctx.separate(pa, pb, d.bg, d.sig_n);
              acc.push_back({j, !separate, float(dd)});
            } else if (dd < r_blend) {
              acc.push_back({j, false, float(dd)});
            }
          }
          j = next[j];
        }
      }
    }
  });
  std::vector<bool> keep(n, false);
  struct Bl {
    size_t i;
    double sep;
    float ratio;
  };
  std::vector<Bl> blended;
  for (size_t i : order) {
    bool dup_kept = false;
    for (const Rel& r : rels[i])
      if (r.dup && keep[r.other]) {
        dup_kept = true;
        break;
      }
    if (dup_kept) continue;
    keep[i] = true;
    for (const Rel& r : rels[i]) {
      if (!keep[r.other]) continue;
      float hi = fmax_r(dets[i].h, F32_MIN_POSITIVE), hj = fmax_r(dets[r.other].h, F32_MIN_POSITIVE);
      blended.push_back({i, double(r.dist), hj / hi});
      blended.push_back({size_t(r.other), double(r.dist), hi / hj});
    }
  }
  std::stable_sort(blended.begin(), blended.end(), [](const Bl& a, const Bl& b) {
    if (a.i != b.i) return a.i < b.i;
    return total_cmp(a.sep, b.sep) < 0;
  });
  {
    std::vector<Bl> dd;
    for (const Bl& b : blended)
      if (dd.empty() || dd.back().i != b.i) dd.push_back(b);
    blended.swap(dd);
  }
  std::vector<std::optional<Det>> redo(blended.size());
  parallel_for(blended.size(), [&](size_t k) {
    const Bl& b = blended[k];
    if (!(keep[b.i] && dets[b.i].core == 0.0f && b.sep < 1.3 * ff && b.ratio >= 0.25f)) return;
    const Det& d = dets[b.i];
    float cap = float(fmax_r(b.sep / 3.0, 0.6));
    auto m = ctx.measure(d.px, d.py, true, cap, false);
    if (m) {
      m->flags |= d.flags | star_flags::BLENDED;
      redo[k] = std::move(m);
    }
  });
  for (size_t k = 0; k < blended.size(); ++k)
    if (redo[k]) dets[blended[k].i] = std::move(*redo[k]);
  for (const Bl& b : blended) dets[b.i].flags |= star_flags::BLENDED;
  std::vector<Det> kept;
  kept.reserve(n);
  for (size_t i = 0; i < n; ++i)
    if (keep[i]) kept.push_back(std::move(dets[i]));
  dets.swap(kept);
}

// =============================================================================================
// merge_psf_lobes
// =============================================================================================
void merge_psf_lobes(const Ctx& ctx, std::vector<Det>& dets) {
  size_t n = dets.size();
  double ff = double(ctx.fwhm_frame);
  if (n < 20 || ff <= 0.0) return;
  double reach = 2.5 * ff;
  double cell = fmax_r(reach, 4.0);
  size_t gw = sat_cast<size_t>(double(ctx.w) / cell) + 1, gh = sat_cast<size_t>(double(ctx.h) / cell) + 1;
  auto cell_of = [&](double x, double y) {
    return std::make_pair(std::min(sat_cast<size_t>(fmax_r(x / cell, 0.0)), gw - 1),
                          std::min(sat_cast<size_t>(fmax_r(y / cell, 0.0)), gh - 1));
  };
  std::vector<uint32_t> head(gw * gh, UINT32_MAX), next(n, UINT32_MAX);
  for (size_t i = 0; i < n; ++i) {
    auto [cx, cy] = cell_of(dets[i].x, dets[i].y);
    next[i] = head[cy * gw + cx];
    head[cy * gw + cx] = uint32_t(i);
  }
  struct Pair {
    uint32_t p, s;
    std::array<double, 2> v;
  };
  std::vector<std::optional<Pair>> pr(n);
  parallel_for(n, [&](size_t i) {
    const Det& d = dets[i];
    if ((d.flags & star_flags::SATURATED) || d.h < 8.0f * d.sig_n) return;
    auto [cx, cy] = cell_of(d.x, d.y);
    bool has = false;
    double best_d = 0;
    uint32_t best_j = 0;
    for (size_t yy = cy >= 1 ? cy - 1 : 0; yy <= std::min(cy + 1, gh - 1); ++yy)
      for (size_t xx = cx >= 1 ? cx - 1 : 0; xx <= std::min(cx + 1, gw - 1); ++xx) {
        uint32_t j = head[yy * gw + xx];
        while (j != UINT32_MAX) {
          const Det& o = dets[j];
          double dd = dist(o.x - d.x, o.y - d.y);
          if (j != i && o.h > d.h && dd < reach && (!has || dd < best_d)) {
            has = true;
            best_d = dd;
            best_j = j;
          }
          j = next[j];
        }
      }
    if (!has) return;
    const Det& o = dets[best_j];
    if (d.h >= 0.1f * o.h) pr[i] = Pair{best_j, uint32_t(i), {d.x - o.x, d.y - o.y}};
  });
  std::vector<Pair> pairs;
  for (auto& p : pr)
    if (p) pairs.push_back(*p);
  if (pairs.size() < 8) return;
  std::stable_sort(pairs.begin(), pairs.end(),
                   [&](const Pair& a, const Pair& b) { return total_cmp(dets[b.p].flux, dets[a.p].flux) < 0; });
  if (pairs.size() > 3000) pairs.resize(3000);
  double radius = 0.35 * double(std::max(ctx.w, ctx.h));
  size_t bw = sat_cast<size_t>(double(ctx.w) / radius) + 1, bh = sat_cast<size_t>(double(ctx.h) / radius) + 1;
  auto bucket_of = [&](double x, double y) {
    return std::make_pair(std::min(sat_cast<size_t>(fmax_r(x / radius, 0.0)), bw - 1),
                          std::min(sat_cast<size_t>(fmax_r(y / radius, 0.0)), bh - 1));
  };
  std::vector<std::vector<uint32_t>> buckets(bw * bh);
  for (size_t k = 0; k < pairs.size(); ++k) {
    auto [bx, by] = bucket_of(dets[pairs[k].p].x, dets[pairs[k].p].y);
    buckets[by * bw + bx].push_back(uint32_t(k));
  }
  std::vector<uint8_t> lobe(pairs.size(), 0);
  parallel_for(pairs.size(), [&](size_t k) {
    const Pair& pk = pairs[k];
    double x = dets[pk.p].x, y = dets[pk.p].y;
    auto [bx, by] = bucket_of(x, y);
    size_t near = 0, agree = 0;
    double tol = fmax_r(0.35 * std::hypot(pk.v[0], pk.v[1]), 2.0);
    for (size_t yy = by >= 1 ? by - 1 : 0; yy <= std::min(by + 1, bh - 1); ++yy)
      for (size_t xx = bx >= 1 ? bx - 1 : 0; xx <= std::min(bx + 1, bw - 1); ++xx)
        for (uint32_t m : buckets[yy * bw + xx]) {
          const Pair& q = pairs[m];
          const Det& o = dets[q.p];
          if (m == k || dist(o.x - x, o.y - y) > radius) continue;
          near += 1;
          agree += size_t(std::hypot(q.v[0] - pk.v[0], q.v[1] - pk.v[1]) < tol);
        }
    lobe[k] = agree >= 3 && agree * 10 >= near * 3;
  });
  std::vector<bool> drop(n, false);
  for (size_t k = 0; k < pairs.size(); ++k) {
    size_t p = pairs[k].p, s = pairs[k].s;
    if (!lobe[k] || drop[p] || drop[s]) continue;
    double fp = fmax_r(dets[p].flux, 0.0), fs = fmax_r(dets[s].flux, 0.0);
    double total = fmax_r(fp + fs, std::numeric_limits<double>::min());
    double sx = dets[s].x, sy = dets[s].y, sf = dets[s].fwhm;
    Det& d = dets[p];
    double sep = dist(sx - d.x, sy - d.y);
    d.x = (fp * d.x + fs * sx) / total;
    d.y = (fp * d.y + fs * sy) / total;
    d.flux = fp + fs;
    d.snr *= std::sqrt(total / fmax_r(fp, std::numeric_limits<double>::min()));
    d.fwhm = fmax_r(d.fwhm, sep + 0.5 * (d.fwhm + sf));
    d.flags &= ~star_flags::BLENDED;
    drop[s] = true;
  }
  std::vector<Det> kept;
  for (size_t i = 0; i < n; ++i)
    if (!drop[i]) kept.push_back(std::move(dets[i]));
  dets.swap(kept);
}

// =============================================================================================
// vet.rs
// =============================================================================================
namespace {
double quantile_f64(std::vector<double>& v, double q) {
  if (v.empty()) return 0.0;
  size_t i = std::min(sat_cast<size_t>(double(v.size()) * q), v.size() - 1);
  std::nth_element(v.begin(), v.begin() + long(i), v.end(), [](double a, double b) { return total_cmp(a, b) < 0; });
  return v[i];
}
double median_f64(std::vector<double>& v) { return quantile_f64(v, 0.5); }

struct PsfSample {
  double x, y, fwhm;
  bool has_shape;
  double e, pa, len;
};

std::optional<PsfSample> psf_sample(const Ctx& ctx, const Cand& cand, float min_snr) {
  auto pk = ctx.native_peak(cand);
  if (!pk) return std::nullopt;
  auto [px, py] = *pk;
  auto [plane, sig] = ctx.plane_bg(px, py);
  float bg = plane.c, pv = ctx.px(px, py);
  if (std::isnan(pv - bg) || pv - bg < min_snr * sig) return std::nullopt;
  float thr = fmax_r(1.5f * sig, 0.02f * (pv - bg));
  double f = double(thr / (pv - bg));
  Moments small = ctx.moments(double(px), double(py), 3, plane, thr);
  if (small.shape(f).elong > 4.0) small = ctx.moments(small.x, small.y, 5, plane, thr);
  if (small.sum <= 0.0 || small.clipped) return std::nullopt;
  Moments big = ctx.moments(small.x, small.y, 8, plane, thr);
  PsfSample s;
  s.x = small.x;
  s.y = small.y;
  s.has_shape = false;
  s.e = s.pa = s.len = 0;
  if (big.sum > 0.0 && !big.clipped) {
    auto sh = big.shape(f);
    auto [major, minor] = big.axes();
    s.has_shape = true;
    s.e = sh.elong;
    s.pa = sh.pa;
    s.len = std::sqrt(12.0 * fmax_r(major - minor, 0.0));
  }
  s.fwhm = small.shape(f).fwhm;
  return s;
}

std::vector<bool> streak_members(const std::vector<PsfSample>& samples) {
  struct L {
    size_t i;
    double x, y, pa;
  };
  std::vector<L> lng;
  for (size_t i = 0; i < samples.size(); ++i)
    if (samples[i].has_shape && samples[i].e > 1.8) lng.push_back({i, samples[i].x, samples[i].y, samples[i].pa});
  std::vector<bool> on(samples.size(), false);
  for (const L& a : lng) {
    double ang = to_radians(a.pa);
    double sn = std::sin(ang), cs = std::cos(ang);
    double width = fmax_r(0.5 * samples[a.i].fwhm, 2.0);
    size_t collinear = 0;
    for (const L& b : lng) {
      double dpa = rem_euclid(b.pa - a.pa, 180.0);
      if (b.i != a.i && fmin_r(dpa, 180.0 - dpa) < 10.0 && std::fabs((b.x - a.x) * sn - (b.y - a.y) * cs) < width) ++collinear;
    }
    on[a.i] = collinear >= 3;
  }
  return on;
}
}  // namespace

FramePsf frame_psf(const Ctx& ctx, const std::vector<Cand>& cands) {
  if (cands.size() < 5) return {};
  auto key = [&](size_t i) { return cands[i].v - cands[i].bg; };
  size_t tiles = ctx.grid->nx * ctx.grid->ny;
  std::vector<bool> taken(cands.size(), false);
  std::vector<size_t> order;
  for (int round_i = 0; round_i < 6; ++round_i) {
    std::vector<size_t> best(tiles, SIZE_MAX);
    for (size_t i = 0; i < cands.size(); ++i) {
      if (taken[i]) continue;
      size_t t = ctx.grid->tile_index(cands[i].x, cands[i].y);
      if (best[t] == SIZE_MAX || key(i) > key(best[t])) best[t] = i;
    }
    std::vector<size_t> round;
    for (size_t i : best)
      if (i != SIZE_MAX) round.push_back(i);
    if (round.empty()) break;
    std::stable_sort(round.begin(), round.end(), [&](size_t a, size_t b) { return total_cmp(key(b), key(a)) < 0; });
    for (size_t i : round) {
      taken[i] = true;
      if (order.size() < 420 && ctx.is_compact_star(cands[i])) order.push_back(i);
    }
    if (order.size() >= 420) break;
  }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return total_cmp(key(b), key(a)) < 0; });
  size_t skip = order.size() / 20;
  if (order.size() > skip + 300) order.resize(skip + 300);
  std::vector<PsfSample> samples;
  const float snrs[2] = {10.0f, fmin_r(ctx.k, 10.0f)};
  for (float min_snr : snrs) {
    if (samples.size() >= 5) break;
    samples.clear();
    for (size_t t = skip; t < order.size(); ++t)
      if (auto s = psf_sample(ctx, cands[order[t]], min_snr)) samples.push_back(*s);
  }
  auto streak = streak_members(samples);
  std::vector<const PsfSample*> kept;
  for (size_t i = 0; i < samples.size(); ++i)
    if (!streak[i]) kept.push_back(&samples[i]);
  if (kept.size() < 5) return {};
  std::vector<double> fwhm, elong;
  for (auto* s : kept) fwhm.push_back(s->fwhm);
  for (auto* s : kept)
    if (s->has_shape) elong.push_back(s->e);
  size_t n = elong.size();
  double c2 = 0, s2 = 0;
  size_t n_e = 0;
  std::vector<double> length;
  for (auto* s : kept) {
    if (!s->has_shape) continue;
    if (s->e > 1.3) {
      double a = to_radians(2.0 * s->pa);
      c2 += std::cos(a);
      s2 += std::sin(a);
      n_e += 1;
      length.push_back(s->len);
    }
  }
  double fwhm_med = quantile_f64(fwhm, 0.35);
  double elong_med = median_f64(elong);
  double coherence = n_e > 0 ? std::hypot(c2, s2) / double(n_e) : 0.0;
  FramePsf out;
  out.fwhm = float(fwhm_med);
  if (n >= 10 && elong_med > 1.6 && coherence > 0.8 && n_e * 10 >= n * 6)
    out.trail = Trail{float(rem_euclid(0.5 * std::atan2(s2, c2), PI)), float(median_f64(length))};
  return out;
}

void vet(const Ctx& ctx, std::vector<Det>& dets) {
  auto clean = [](const Det& d) {
    return (d.flags & (star_flags::SATURATED | star_flags::NEAR_EDGE | star_flags::RIDGE)) == 0 && d.h >= 10.0f * d.sig_n;
  };
  std::vector<double> ratio, fwhm, elong;
  for (const Det& d : dets)
    if (clean(d)) {
      ratio.push_back(double(d.ratio4));
      fwhm.push_back(d.fwhm);
      elong.push_back(d.elong);
    }
  bool enough = ratio.size() >= 5;
  double ratio_med = enough ? median_f64(ratio) : 0.5;
  double fwhm_med = enough ? median_f64(fwhm) : double(fmax_r(ctx.fwhm_frame, 2.0f));
  double elong_med = enough ? median_f64(elong) : 1.2;
  double sharp_limit = fmin_r(0.45 * ratio_med, 0.15);
  double broad_limit = fmax_r(2.5 * fwhm_med, fwhm_med + 4.0);
  double fwhm_limit = fmin_r(0.5 * fwhm_med, 1.2);
  auto width = [](const Det& d) { return std::sqrt(fmax_r(d.flux / (1.1331 * double(d.h)), 0.0)); };
  std::vector<double> widths;
  for (const Det& d : dets)
    if (clean(d)) widths.push_back(width(d));
  double width_med = enough ? median_f64(widths) : 0.0;
  float sat_tol = fmax_r(0.5f * ctx.lsb, 1e-6f * std::fabs(ctx.sat_level));
  for (Det& d : dets) {
    bool saturated = (d.flags & star_flags::SATURATED) != 0;
    if (d.h >= 10.0f * d.sig_n) {
      bool sharp = double(d.ratio4) < sharp_limit;
      bool tiny = d.fwhm < fwhm_limit;
      bool peaky = d.fwhm < 1.5 && double(d.h) > 0.7 * d.flux && fwhm_med > 2.0;
      bool clipped = saturated || d.bg + d.h >= ctx.sat_level - sat_tol;
      bool packed = d.snr >= 20.0 && width(d) < (clipped ? 0.6 : 0.4) * width_med;
      if (packed || (!saturated && (sharp || tiny || peaky))) d.flags |= star_flags::HOT_PIXEL;
    }
    bool bright = d.h >= 0.2f * (ctx.sat_level - d.bg);
    if (!saturated && !bright && d.core == 0.0f && enough && d.fwhm > broad_limit) d.flags |= star_flags::EXTENDED;
    if (ctx.trail) {
      double diff = rem_euclid(d.pa - to_degrees(double(ctx.trail->pa)), 180.0);
      if (d.elong > 1.2 && fmin_r(diff, 180.0 - diff) < 20.0) d.flags |= star_flags::TRAILED;
    } else if (!saturated) {
      double floor = d.snr >= 30.0 ? 2.5 : 4.0;
      if (d.elong > fmax_r(2.0 * elong_med, floor)) d.flags |= star_flags::ELONGATED;
    }
  }
}

// =============================================================================================
// lines.rs
// =============================================================================================
namespace {
using P2f = std::array<float, 2>;

struct Points {
  std::vector<P2f> xy;
  size_t n_det;
  const std::vector<Det>* dets;
  float cell;
  size_t gw, gh;
  std::vector<uint32_t> head, next;

  Points(const std::vector<Det>& d, const std::vector<P2f>& hints, size_t w, size_t h, float cell_) : n_det(d.size()), dets(&d), cell(cell_) {
    for (const Det& x : d) xy.push_back({float(x.x), float(x.y)});
    xy.insert(xy.end(), hints.begin(), hints.end());
    gw = sat_cast<size_t>(float(w) / cell) + 1;
    gh = sat_cast<size_t>(float(h) / cell) + 1;
    head.assign(gw * gh, UINT32_MAX);
    next.assign(xy.size(), UINT32_MAX);
    for (size_t i = 0; i < xy.size(); ++i) {
      size_t c = cell_of(xy[i]);
      next[i] = head[c];
      head[c] = uint32_t(i);
    }
  }
  size_t cell_of(const P2f& p) const {
    size_t cx = std::min(sat_cast<size_t>(fmax_r(p[0] / cell, 0.0f)), gw - 1);
    size_t cy = std::min(sat_cast<size_t>(fmax_r(p[1] / cell, 0.0f)), gh - 1);
    return cy * gw + cx;
  }
  template <class F>
  void visit(P2f lo, P2f hi, F&& f) const {
    size_t c0x = std::min(sat_cast<size_t>(fmax_r(lo[0] / cell, 0.0f)), gw - 1);
    size_t c0y = std::min(sat_cast<size_t>(fmax_r(lo[1] / cell, 0.0f)), gh - 1);
    size_t c1x = std::min(sat_cast<size_t>(fmax_r(hi[0] / cell, 0.0f)), gw - 1);
    size_t c1y = std::min(sat_cast<size_t>(fmax_r(hi[1] / cell, 0.0f)), gh - 1);
    for (size_t cy = c0y; cy <= c1y; ++cy)
      for (size_t cx = c0x; cx <= c1x; ++cx) {
        uint32_t j = head[cy * gw + cx];
        while (j != UINT32_MAX) {
          const P2f& p = xy[j];
          if (p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1]) f(size_t(j));
          j = next[j];
        }
      }
  }
  std::vector<std::pair<float, size_t>> nearest(size_t i, float max_r, size_t k_max) const {
    P2f p = xy[i];
    int64_t cx = std::min<int64_t>(sat_cast<int64_t>(fmax_r(p[0] / cell, 0.0f)), int64_t(gw) - 1);
    int64_t cy = std::min<int64_t>(sat_cast<int64_t>(fmax_r(p[1] / cell, 0.0f)), int64_t(gh) - 1);
    std::vector<std::pair<float, size_t>> found;
    int64_t rings = sat_cast<int64_t>(std::ceil(max_r / cell)) + 1;
    auto cmp = [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) { return total_cmp(a.first, b.first) < 0; };
    for (int64_t k = 0; k <= rings; ++k) {
      if (found.size() >= k_max) {
        std::stable_sort(found.begin(), found.end(), cmp);
        if (found[k_max - 1].first < float(k - 1) * cell) break;
      }
      for (int64_t gy = cy - k; gy <= cy + k; ++gy)
        for (int64_t gx = cx - k; gx <= cx + k; ++gx) {
          if (std::max(std::llabs(gy - cy), std::llabs(gx - cx)) != k || gx < 0 || gy < 0 || gx >= int64_t(gw) || gy >= int64_t(gh)) continue;
          uint32_t j = head[size_t(gy) * gw + size_t(gx)];
          while (j != UINT32_MAX) {
            if (j != i && j < n_det) {
              const P2f& q = xy[j];
              float a = q[0] - p[0], b = q[1] - p[1];
              float d = std::sqrt(a * a + b * b);
              if (d >= 1.0f && d <= max_r) found.emplace_back(d, size_t(j));
            }
            j = next[j];
          }
        }
    }
    std::stable_sort(found.begin(), found.end(), cmp);
    if (found.size() > k_max) found.resize(k_max);
    return found;
  }
  bool aligned(size_t k, float angle) const {
    if (k >= n_det) return false;
    const Det& d = (*dets)[k];
    double diff = rem_euclid(d.pa - to_degrees(double(angle)), 180.0);
    return d.elong >= 1.4 && fmin_r(diff, 180.0 - diff) < 15.0;
  }
  bool contains(const std::deque<size_t>& c, size_t k) const { return std::ranges::contains(c, k); }

  std::vector<size_t> grow(size_t a, size_t b, float ff, bool dotted) const {
    std::deque<size_t> chain{a, b};
    for (int dir_i = 0; dir_i < 2; ++dir_i) {
      bool forward = dir_i == 0;
      for (;;) {
        P2f p0 = forward ? xy[chain.front()] : xy[chain.back()];
        P2f p1 = forward ? xy[chain.back()] : xy[chain.front()];
        float ddx = p1[0] - p0[0], ddy = p1[1] - p0[1];
        float len = std::sqrt(ddx * ddx + ddy * ddy);
        if (len <= 0.0f || chain.size() >= 400) break;
        P2f dir = {(p1[0] - p0[0]) / len, (p1[1] - p0[1]) / len};
        float spacing = len / float(chain.size() - 1);
        if (dotted) {
          P2f pred = {p1[0] + dir[0] * spacing, p1[1] + dir[1] * spacing};
          float tol = 1.5f + 0.1f * spacing;
          P2f lo = {pred[0] - tol, pred[1] - tol}, hi = {pred[0] + tol, pred[1] + tol};
          bool has = false;
          float best_d = 0;
          size_t best_k = 0;
          visit(lo, hi, [&](size_t k) {
            if (k >= n_det || contains(chain, k)) return;
            const P2f& p = xy[k];
            float ax = p[0] - pred[0], ay = p[1] - pred[1];
            float d = std::sqrt(ax * ax + ay * ay);
            if (d < tol && (!has || d < best_d)) {
              has = true;
              best_d = d;
              best_k = k;
            }
          });
          if (!has) break;
          if (forward) chain.push_back(best_k);
          else chain.push_front(best_k);
          continue;
        }
        float g_max = fmin_r(fmax_r(1.8f * spacing, 4.0f * ff), 150.0f);
        float tol = 1.5f + 0.005f * len;
        P2f end = {p1[0] + dir[0] * g_max, p1[1] + dir[1] * g_max};
        P2f lo = {fmin_r(p1[0], end[0]) - tol, fmin_r(p1[1], end[1]) - tol};
        P2f hi = {fmax_r(p1[0], end[0]) + tol, fmax_r(p1[1], end[1]) + tol};
        bool has = false;
        float best_t = 0;
        size_t best_k = 0;
        visit(lo, hi, [&](size_t k) {
          if (contains(chain, k)) return;
          const P2f& p = xy[k];
          float dx = p[0] - p1[0], dy = p[1] - p1[1];
          float t = dx * dir[0] + dy * dir[1];
          if (t <= 0.5f || t > g_max) return;
          float ex = p[0] - p0[0], ey = p[1] - p0[1];
          float lateral = std::fabs(ex * dir[1] - ey * dir[0]);
          if (lateral < tol && (!has || t < best_t)) {
            has = true;
            best_t = t;
            best_k = k;
          }
        });
        if (!has) break;
        if (forward) chain.push_back(best_k);
        else chain.push_front(best_k);
      }
    }
    return std::vector<size_t>(chain.begin(), chain.end());
  }

  struct Geometry {
    std::vector<std::pair<float, size_t>> t;
    float rms;
    P2f p0, dir;
  };
  Geometry geometry(const std::vector<size_t>& chain) const {
    P2f p0 = xy[chain.front()], p1 = xy[chain.back()];
    float ddx = p1[0] - p0[0], ddy = p1[1] - p0[1];
    float len = fmax_r(std::sqrt(ddx * ddx + ddy * ddy), 1e-6f);
    P2f dir = {(p1[0] - p0[0]) / len, (p1[1] - p0[1]) / len};
    Geometry g;
    for (size_t k : chain) {
      const P2f& p = xy[k];
      g.t.emplace_back((p[0] - p0[0]) * dir[0] + (p[1] - p0[1]) * dir[1], k);
    }
    std::stable_sort(g.t.begin(), g.t.end(), [](const auto& a, const auto& b) { return total_cmp(a.first, b.first) < 0; });
    float sum = -0.0f;
    for (size_t k : chain) {
      const P2f& p = xy[k];
      float v = (p[0] - p0[0]) * dir[1] - (p[1] - p0[1]) * dir[0];
      sum += v * v;
    }
    g.rms = std::sqrt(sum / float(chain.size()));
    g.p0 = p0;
    g.dir = dir;
    return g;
  }
};

std::pair<float, float> lit_fraction(const Ctx& ctx, const std::vector<std::pair<float, size_t>>& t, P2f p0, P2f dir, float ff, float off) {
  float b = float(ctx.b);
  auto lit_at = [&](float x, float y) -> std::optional<bool> {
    float xb = (x + 0.5f) / b - 0.5f, yb = (y + 0.5f) / b - 0.5f;
    float v = ctx.plane->bilinear(xb, yb);
    if (!std::isfinite(v)) return std::nullopt;
    auto g = ctx.grid->at(xb, yb);
    return v > g.bg + 2.0f * g.sig_b;
  };
  P2f normal = {-dir[1], dir[0]};
  size_t lit = 0, side = 0, total = 0;
  for (size_t i = 0; i + 1 < t.size(); ++i) {
    float ta = t[i].first + 1.5f * ff, tb = t[i + 1].first - 1.5f * ff;
    float s = ta;
    while (s < tb) {
      float x = p0[0] + dir[0] * s, y = p0[1] + dir[1] * s;
      if (auto on = lit_at(x, y)) {
        total += 1;
        lit += size_t(*on);
        bool a = lit_at(x + normal[0] * off, y + normal[1] * off).value_or(false);
        bool c = lit_at(x - normal[0] * off, y - normal[1] * off).value_or(false);
        side += size_t(a && c);
      }
      s += 1.0f;
    }
  }
  if (total < 3) return {1.0f, 0.0f};
  return {float(lit) / float(total), float(side) / float(total)};
}

bool continuous(const Ctx& ctx, const Points& pts, const std::vector<size_t>& chain, float ff) {
  auto g = pts.geometry(chain);
  if (g.rms > 1.2f) return false;
  float angle = std::atan2(g.dir[1], g.dir[0]);
  size_t accepted = 0, aligned = 0;
  for (size_t k : chain)
    if (k < pts.n_det) {
      ++accepted;
      if (pts.aligned(k, angle)) ++aligned;
    }
  if (aligned < 2 || aligned * 10 < accepted * 6) return false;
  float span = g.t.back().first - g.t.front().first;
  if (span < 6.0f * ff) return false;
  auto [on, beside] = lit_fraction(ctx, g.t, g.p0, g.dir, ff, 2.0f * ff + 2.0f);
  return on >= 0.75f && beside <= 0.35f;
}

bool dotted(const Points& pts, const std::vector<size_t>& chain) {
  auto g = pts.geometry(chain);
  if (g.rms > 0.7f) return false;
  std::vector<float> gaps;
  for (size_t i = 0; i + 1 < g.t.size(); ++i) gaps.push_back(g.t[i + 1].first - g.t[i].first);
  float sum = -0.0f;
  for (float x : gaps) sum += x;
  float mean = sum / float(gaps.size());
  float vs = -0.0f;
  for (float x : gaps) vs += (x - mean) * (x - mean);
  float var = vs / float(gaps.size());
  if (mean < 3.0f || std::sqrt(var) > 0.15f * mean) return false;
  double lo = INF, hi = 0.0;
  for (size_t k : chain) {
    double f = (*pts.dets)[k].flux;
    lo = fmin_r(lo, f);
    hi = fmax_r(hi, f);
  }
  return hi <= 5.0 * lo;
}
}  // namespace

void flag_lines(const Ctx& ctx, std::vector<Det>& dets, const std::vector<std::array<float, 2>>& hints) {
  if (dets.size() < 4) return;
  float ff = fmax_r(ctx.fwhm_frame, 2.0f);
  Points pts(dets, hints, ctx.w, ctx.h, 16.0f);
  size_t n = pts.n_det;
  float r_pair = clampf(8.0f * ff, 12.0f, 96.0f);
  std::vector<std::vector<std::vector<size_t>>> chains(n);
  parallel_for(n, [&](size_t i) {
    auto& out = chains[i];
    P2f p = pts.xy[i];
    const Det& d = dets[i];
    std::vector<size_t> seeds;
    if (d.elong >= 1.8 && d.snr >= 8.0) {
      P2f lo = {p[0] - r_pair, p[1] - r_pair}, hi = {p[0] + r_pair, p[1] + r_pair};
      pts.visit(lo, hi, [&](size_t j) {
        P2f q = pts.xy[j];
        float ax = q[0] - p[0], ay = q[1] - p[1];
        float dd = std::sqrt(ax * ax + ay * ay);
        if (j != i && dd >= 1.0f && dd <= r_pair && pts.aligned(i, std::atan2(q[1] - p[1], q[0] - p[0])) &&
            (j >= n || pts.aligned(j, std::atan2(q[1] - p[1], q[0] - p[0]))))
          seeds.push_back(j);
      });
    }
    auto nearest = pts.nearest(i, 160.0f, 4);
    for (size_t j : seeds) {
      auto chain = pts.grow(i, j, ff, false);
      if (chain.size() >= 4 && continuous(ctx, pts, chain, ff)) {
        auto g = pts.geometry(chain);
        float angle = std::atan2(g.dir[1], g.dir[0]);
        std::vector<size_t> sel;
        for (size_t k : chain)
          if (pts.aligned(k, angle)) sel.push_back(k);
        out.push_back(std::move(sel));
      }
    }
    for (auto& nj : nearest) {
      size_t j = nj.second;
      double fi = dets[i].flux, fj = dets[j].flux;
      if (j > i && fi < 3.0 * fj && fj < 3.0 * fi) {
        auto chain = pts.grow(i, j, ff, true);
        if (chain.size() >= 6 && dotted(pts, chain)) out.push_back(std::move(chain));
      }
    }
  });
  for (auto& per : chains)
    for (auto& chain : per)
      for (size_t k : chain)
        if (k < n) dets[k].flags |= star_flags::ON_LINE;
}

}  // namespace skyshot::extract_detail
