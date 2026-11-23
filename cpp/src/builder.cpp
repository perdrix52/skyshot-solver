#include "skyshot/builder.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>

#include "skyshot/fingerprint.hpp"
#include "skyshot/geometry.hpp"
#include "skyshot/learned.hpp"
#include "skyshot/parallel.hpp"
#include "skyshot/select.hpp"
#include "skyshot/sha256.hpp"

namespace skyshot::builder {

namespace fs = std::filesystem;
using sphere::CellIndex;
using sphere::CubeGrid;
using Clock = std::chrono::steady_clock;

const std::vector<PlanEntry>& record_plan(uint32_t m, bool dominant) {
  static const std::vector<PlanEntry> FULL4 = {{0, {1, 2, 3}}, {0, {1, 2, 4}}, {0, {1, 3, 4}}, {0, {2, 3, 4}}};
  static const std::vector<PlanEntry> ONE = {{0, {1, 2, 3}}};
  static const std::vector<PlanEntry> TWO = {{0, {1, 2, 3}}, {1, {2, 3, 4}}};
  static const std::vector<PlanEntry> NONE;
  if (m <= 6) return dominant ? FULL4 : ONE;
  if (m == 7 && dominant) return TWO;
  if (m == 8 && dominant) return ONE;
  return NONE;
}

static void log(Clock::time_point t0, const std::string& msg) {
  double s = std::chrono::duration<double>(Clock::now() - t0).count();
  std::fprintf(stderr, "[%7.1fs] %s\n", s, msg.c_str());
}

using V3f = std::array<float, 3>;
static V3f vec32(const Vec3& v) { return {float(v[0]), float(v[1]), float(v[2])}; }
static double chord2(const V3f& a, const V3f& b) {
  double dx = double(a[0]) - double(b[0]), dy = double(a[1]) - double(b[1]), dz = double(a[2]) - double(b[2]);
  return dx * dx + dy * dy + dz * dz;
}
static double chord_to_angle(double c) { return 2.0 * std::asin(fmin_r(c / 2.0, 1.0)); }

static uint32_t coarsen(uint32_t cell0, uint32_t n0, uint32_t k) {
  uint32_t nn0 = n0 * n0;
  uint32_t f = cell0 / nn0, i = cell0 % n0, j = (cell0 % nn0) / n0;
  uint32_t n = n0 >> k;
  return f * n * n + (j >> k) * n + (i >> k);
}

namespace {
struct Catalog {
  std::vector<d80::RawStar> raw;
  std::vector<V3f> vec;
  std::vector<uint32_t> order, rank;
};

std::pair<Catalog, std::string> load_catalog(const fs::path& dir, Clock::time_point t0) {
  auto db = d80::D80::open(dir);
  auto paths = db.tile_paths();
  Sha256 hasher;
  for (auto& p : paths) {
    std::error_code ec;
    uint64_t len = fs::file_size(p, ec);
    if (ec) throw Error("metadata " + p.string());
    std::string name = p.filename().string();
    hasher.update(name);
    uint8_t le[8];
    for (int i = 0; i < 8; ++i) le[i] = uint8_t(len >> (8 * i));
    hasher.update(le, 8);
  }
  std::string fingerprint = hasher.hex();
  std::vector<std::vector<d80::RawStar>> tiles(paths.size());
  std::vector<std::string> errors(paths.size());
  parallel_for(paths.size(), [&](size_t i) {
    try {
      auto data = read_file_bytes(paths[i].string());
      tiles[i].reserve(data.size() / 5);
      d80::decode_raw(data.data(), data.size(), tiles[i]);
    } catch (const std::exception& e) {
      errors[i] = e.what();
    }
  });
  for (size_t i = 0; i < errors.size(); ++i)
    if (!errors[i].empty()) throw Error(errors[i]);
  size_t n = 0;
  for (auto& t : tiles) n += t.size();
  Catalog cat;
  cat.raw.reserve(n);
  for (auto& t : tiles) cat.raw.insert(cat.raw.end(), t.begin(), t.end());
  tiles.clear();
  log(t0, "decoded " + std::to_string(cat.raw.size()) + " stars from " + std::to_string(paths.size()) + " tiles");
  if (cat.raw.size() >= UINT32_MAX) throw Error("too many stars");
  int lo = 0, hi = 0;
  if (!cat.raw.empty()) {
    lo = hi = cat.raw[0].mag10;
    for (auto& s : cat.raw) {
      lo = std::min<int>(lo, s.mag10);
      hi = std::max<int>(hi, s.mag10);
    }
  }
  std::vector<size_t> counts(size_t(hi - lo + 2), 0);
  for (auto& s : cat.raw) counts[size_t(s.mag10 - lo) + 1] += 1;
  for (size_t i = 1; i < counts.size(); ++i) counts[i] += counts[i - 1];
  cat.order.assign(cat.raw.size(), 0);
  for (size_t i = 0; i < cat.raw.size(); ++i) {
    size_t b = size_t(cat.raw[i].mag10 - lo);
    cat.order[counts[b]] = uint32_t(i);
    counts[b] += 1;
  }
  cat.rank.assign(cat.raw.size(), 0);
  for (size_t r = 0; r < cat.order.size(); ++r) cat.rank[cat.order[r]] = uint32_t(r);
  cat.vec.resize(cat.raw.size());
  parallel_chunks(cat.raw.size(), 1 << 16, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) cat.vec[i] = vec32(sphere::unit(cat.raw[i].ra_deg(), cat.raw[i].dec_deg()));
  });
  log(t0, "brightness order and unit vectors ready");
  return {std::move(cat), fingerprint};
}

std::vector<float> dominance(const Catalog& cat, double r_min, Clock::time_point t0) {
  const uint32_t n0 = 4096;
  CubeGrid g0(n0);
  std::vector<uint32_t> cells0(cat.vec.size());
  parallel_chunks(cat.vec.size(), 1 << 16, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) cells0[i] = g0.cell({double(cat.vec[i][0]), double(cat.vec[i][1]), double(cat.vec[i][2])});
  });
  auto idx0 = CellIndex::build(g0, cells0, cat.order.begin(), cat.order.end());
  log(t0, "level-0 grid built");
  double c2min = powi(sphere::chord(r_min), 2);
  std::vector<uint8_t> survivor(cat.raw.size(), 0);
  parallel_chunks(cat.raw.size(), 1 << 14, [&](size_t a, size_t b) {
    std::vector<uint32_t> cells;
    for (size_t id = a; id < b; ++id) {
      const V3f& v = cat.vec[id];
      uint32_t rk = cat.rank[id];
      g0.cap_cells({double(v[0]), double(v[1]), double(v[2])}, r_min, cells);
      bool ok = true;
      for (uint32_t c : cells) {
        for (const uint32_t* p = idx0.cell_begin(c); p != idx0.cell_end(c); ++p) {
          uint32_t j = *p;
          if (cat.rank[j] >= rk) break;
          if (chord2(v, cat.vec[j]) < c2min) {
            ok = false;
            break;
          }
        }
        if (!ok) break;
      }
      survivor[id] = ok;
    }
  });
  size_t n_surv = 0;
  for (uint8_t s : survivor) n_surv += s;
  {
    char buf[96];
    std::snprintf(buf, sizeof buf, "phase A: %zu stars with D >= %.3f arcmin", n_surv, to_degrees(r_min) * 60.0);
    log(t0, buf);
  }
  struct Level {
    uint32_t k;
    CubeGrid g;
    CellIndex idx;
  };
  const uint32_t ks[6] = {2, 4, 6, 8, 10, 12};
  std::vector<Level> pyramid(6);
  parallel_for(6, [&](size_t t) {
    uint32_t k = ks[t];
    CubeGrid g(n0 >> k);
    std::vector<uint32_t> cells(cells0.size());
    for (size_t i = 0; i < cells.size(); ++i) cells[i] = coarsen(cells0[i], n0, k);
    pyramid[t] = Level{k, g, CellIndex::build(g, cells, cat.order.begin(), cat.order.end())};
  });
  cells0.clear();
  cells0.shrink_to_fit();
  log(t0, "pyramid grids built");
  std::vector<float> d(cat.raw.size(), std::numeric_limits<float>::quiet_NaN());
  parallel_chunks(cat.raw.size(), 1 << 12, [&](size_t a, size_t b) {
    std::vector<uint32_t> cells;
    for (size_t id = a; id < b; ++id) {
      if (!survivor[id]) continue;
      const V3f& v = cat.vec[id];
      Vec3 vd = {double(v[0]), double(v[1]), double(v[2])};
      uint32_t rk = cat.rank[id];
      double r = 2.0 * r_min;
      for (;;) {
        if (r >= PI) {
          d[id] = 3.14159274f;  // std::f32::consts::PI
          break;
        }
        const Level* lv = &pyramid[0];
        for (size_t t = pyramid.size(); t-- > 0;)
          if (pyramid[t].g.min_cell_width() <= r / 2.0) {
            lv = &pyramid[t];
            break;
          }
        if (r >= 1.0) {
          cells.clear();
          for (uint32_t c = 0; c < lv->g.cells(); ++c) cells.push_back(c);
        } else {
          lv->g.cap_cells(vd, r, cells);
        }
        double best = std::numeric_limits<double>::max();
        for (uint32_t c : cells) {
          const uint32_t* first = lv->idx.cell_begin(c);
          const uint32_t* last = lv->idx.cell_end(c);
          const uint32_t* end = std::partition_point(first, last, [&](uint32_t j) { return cat.rank[j] < rk; });
          for (const uint32_t* p = first; p != end; ++p) best = fmin_r(best, chord2(v, cat.vec[*p]));
        }
        double ang = best == std::numeric_limits<double>::max() ? std::numeric_limits<double>::max() : chord_to_angle(std::sqrt(best));
        if (ang <= r) {
          d[id] = float(ang);
          break;
        }
        r *= 2.0;
      }
    }
  });
  log(t0, "phase B: exact dominance distances");
  return d;
}

// k nearest members (angle-sorted) around v, excluding self_pos.
bool knn(const CubeGrid& grid, const CellIndex& idx, const std::vector<Vec3>& exact, const Vec3& v, uint32_t self_pos, size_t k, double r0,
         std::vector<uint32_t>& cells, std::vector<std::pair<double, uint32_t>>& cand) {
  double r = r0;
  for (;;) {
    cand.clear();
    grid.cap_cells(v, r, cells);
    for (uint32_t c : cells)
      for (const uint32_t* q = idx.cell_begin(c); q != idx.cell_end(c); ++q) {
        uint32_t p = *q;
        if (p == self_pos) continue;
        double a = sphere::angle(v, exact[p]);
        if (a <= r) cand.emplace_back(a, p);
      }
    if (cand.size() >= k) {
      std::sort(cand.begin(), cand.end(), [](const auto& a, const auto& b) {
        if (a.first < b.first) return true;
        if (a.first > b.first) return false;
        return a.second < b.second;
      });
      cand.resize(k);
      return true;
    }
    r *= 2.0;
    if (r > 0.6) return false;
  }
}

void write_bytes(const fs::path& p, const void* data, size_t n) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  if (!f) throw Error("create " + p.string());
  f.write(static_cast<const char*>(data), std::streamsize(n));
  if (!f) throw Error("write " + p.string());
}

void write_stars(const fs::path& out, const char* stars_name, const char* cells_name, const CubeGrid& sg,
                 const std::vector<std::pair<uint32_t, uint32_t>>& l7, const std::vector<Vec3>& exact, const std::vector<double>& dist,
                 const std::vector<float>& mags, double r8) {
  double step = FRAC_PI_2 / double(sg.n);
  std::vector<uint32_t> offsets(sg.cells() + 1, 0);
  for (auto& [cell, k] : l7) offsets[size_t(cell) + 1] += 1;
  for (size_t i = 0; i < sg.cells(); ++i) offsets[i + 1] += offsets[i];
  std::vector<uint8_t> buf;
  buf.reserve(l7.size() * index::STAR_BYTES);
  for (auto& [cell, k] : l7) {
    const Vec3& v = exact[k];
    auto fc = sphere::face_coords(v);
    auto sp = sg.split(cell);
    double fa = clampd((fc.alpha + FRAC_PI_4) / step - double(sp[1]), 0.0, 0.99999999);
    double fb = clampd((fc.beta + FRAC_PI_4) / step - double(sp[2]), 0.0, 0.99999999);
    uint16_t qa = sat_cast<uint16_t>(std::floor(fa * 65536.0));
    uint16_t qb = sat_cast<uint16_t>(std::floor(fb * 65536.0));
    uint8_t dc = sat_cast<uint8_t>(clampd(std::log2(dist[k] / r8) * 16.0, 0.0, 255.0));
    uint8_t mc = sat_cast<uint8_t>(clampf(std::round((mags[k] + 2.0f) * 10.0f), 0.0f, 255.0f));
    uint8_t rec[index::STAR_BYTES] = {uint8_t(qa), uint8_t(qa >> 8), uint8_t(qb), uint8_t(qb >> 8), dc, mc};
    buf.insert(buf.end(), rec, rec + index::STAR_BYTES);
  }
  write_bytes(out / stars_name, buf.data(), buf.size());
  std::vector<uint8_t> ob(offsets.size() * 4);
  for (size_t i = 0; i < offsets.size(); ++i)
    for (int b = 0; b < 4; ++b) ob[i * 4 + size_t(b)] = uint8_t(offsets[i] >> (8 * b));
  write_bytes(out / cells_name, ob.data(), ob.size());
}

void write_codes(const fs::path& out, const std::vector<std::pair<uint32_t, uint64_t>>& recs, size_t buckets) {
  if (recs.size() > UINT32_MAX) throw Error("too many index records");
  std::vector<uint32_t> offsets(buckets + 1, 0);
  for (auto& [b, r] : recs) offsets[size_t(b) + 1] += 1;
  for (size_t i = 0; i < buckets; ++i) offsets[i + 1] += offsets[i];
  std::vector<uint8_t> cb(recs.size() * 8);
  for (size_t i = 0; i < recs.size(); ++i)
    for (int b = 0; b < 8; ++b) cb[i * 8 + size_t(b)] = uint8_t(recs[i].second >> (8 * b));
  write_bytes(out / "codes.bin", cb.data(), cb.size());
  std::vector<uint8_t> kb(offsets.size() * 4);
  for (size_t i = 0; i < offsets.size(); ++i)
    for (int b = 0; b < 4; ++b) kb[i * 4 + size_t(b)] = uint8_t(offsets[i] >> (8 * b));
  write_bytes(out / "keys.bin", kb.data(), kb.size());
}

std::string fmt_g(double v) { return format_f64(v); }

index::Manifest write_index_impl(const BuildConfig& cfg, Clock::time_point t0, const std::vector<Vec3>& exact, const std::vector<double>& dist,
                            const std::vector<float>& mags, uint64_t n_cat, const std::string& fingerprint,
                            std::optional<std::pair<Vec3, double>> anchor_cone) {
  using index::Record;
  bool deep_layout = cfg.profile == "deep";
  double r7 = to_radians(cfg.r7_arcmin) / 60.0;
  double r8 = index::level_radius(r7, 8);
  size_t surv_len = exact.size();
  std::optional<learned::Coder> coder;
  if (cfg.coder) {
    coder = learned::Coder::load(*cfg.coder);
    learned::Coder::copy_files(*cfg.coder, cfg.out);
    log(t0, "learned key coder " + cfg.coder->string() + " (" + coder->sha256 + ")");
  }
  CubeGrid sg(index::STAR_CELL_N);
  std::vector<std::pair<uint32_t, uint32_t>> l7;
  for (uint32_t k = 0; k < surv_len; ++k)
    if (dist[k] >= r7) l7.emplace_back(sg.cell(exact[k]), k);
  std::sort(l7.begin(), l7.end());
  if (l7.size() >= (size_t(1) << (deep_layout ? 28 : 24))) throw Error("level 7 too large for this record format: " + std::to_string(l7.size()));
  std::vector<uint32_t> star_id(surv_len, UINT32_MAX);
  for (size_t sid = 0; sid < l7.size(); ++sid) star_id[l7[sid].second] = uint32_t(sid);
  write_stars(cfg.out, "stars.bin", "cells.bin", sg, l7, exact, dist, mags, r8);
  log(t0, "stars.bin: " + std::to_string(l7.size()) + " level-7 stars");
  bool with_deep = cfg.max_pool_level >= 7;
  for (const char* f : {"deep.bin", "deep_cells.bin"}) {
    std::error_code ec;
    fs::remove(cfg.out / f, ec);
  }
  if (with_deep) {
    std::vector<std::pair<uint32_t, uint32_t>> deep;
    for (uint32_t k = 0; k < surv_len; ++k)
      if (dist[k] < r7) deep.emplace_back(sg.cell(exact[k]), k);
    std::sort(deep.begin(), deep.end());
    write_stars(cfg.out, "deep.bin", "deep_cells.bin", sg, deep, exact, dist, mags, r8);
    log(t0, "deep.bin: " + std::to_string(deep.size()) + " level-8-only stars");
  }
  std::vector<std::pair<uint32_t, uint64_t>> all;
  std::vector<index::LevelInfo> levels;
  for (uint32_t m = deep_layout ? 7 : 1; m <= std::min<uint32_t>(cfg.max_pool_level, 8); ++m) {
    double rm = index::level_radius(r7, m);
    std::vector<uint32_t> members;
    for (uint32_t k = 0; k < surv_len; ++k)
      if (dist[k] >= rm) members.push_back(k);
    CubeGrid grid = CubeGrid::with_min_width(rm);
    std::vector<uint32_t> mcells(members.size());
    for (size_t i = 0; i < members.size(); ++i) mcells[i] = grid.cell(exact[members[i]]);
    std::vector<uint32_t> seq(members.size());
    for (size_t i = 0; i < seq.size(); ++i) seq[i] = uint32_t(i);
    CellIndex midx = CellIndex::build(grid, mcells, seq.begin(), seq.end());
    std::vector<Vec3> mexact(members.size());
    for (size_t i = 0; i < members.size(); ++i) mexact[i] = exact[members[i]];
    std::vector<uint32_t> anchors;
    for (uint32_t p = 0; p < members.size(); ++p) {
      if (record_plan(m, dist[members[p]] >= 2.0 * rm).empty()) continue;
      if (anchor_cone && !(sphere::angle(exact[members[p]], anchor_cone->first) <= anchor_cone->second)) continue;
      anchors.push_back(p);
    }
    std::vector<float> caps = m >= cfg.caps_min_level ? cfg.caps : std::vector<float>{};
    std::vector<float> mmag(members.size());
    for (size_t i = 0; i < members.size(); ++i) mmag[i] = mags[members[i]];
    struct Capped {
      float c;
      CellIndex ci;
      std::vector<uint32_t> sub;
      std::vector<Vec3> sexact;
      std::vector<uint32_t> pos;
    };
    std::vector<Capped> capped(caps.size());
    parallel_for(caps.size(), [&](size_t t) {
      float c = caps[t];
      Capped cp;
      cp.c = c;
      for (uint32_t p = 0; p < members.size(); ++p)
        if (mmag[p] <= c) cp.sub.push_back(p);
      std::vector<uint32_t> cells(cp.sub.size());
      for (size_t i = 0; i < cp.sub.size(); ++i) cells[i] = grid.cell(mexact[cp.sub[i]]);
      std::vector<uint32_t> sq(cp.sub.size());
      for (size_t i = 0; i < sq.size(); ++i) sq[i] = uint32_t(i);
      cp.ci = CellIndex::build(grid, cells, sq.begin(), sq.end());
      for (uint32_t p : cp.sub) cp.sexact.push_back(mexact[p]);
      cp.pos.assign(members.size(), UINT32_MAX);
      for (size_t i = 0; i < cp.sub.size(); ++i) cp.pos[cp.sub[i]] = uint32_t(i);
      capped[t] = std::move(cp);
    });
    std::atomic<uint64_t> n_capped{0};
    std::vector<std::vector<std::pair<uint32_t, uint64_t>>> per(anchors.size());
    parallel_chunks(anchors.size(), 256, [&](size_t ra, size_t rb) {
      std::vector<uint32_t> cells;
      std::vector<std::pair<double, uint32_t>> cand;
      for (size_t ai = ra; ai < rb; ++ai) {
        uint32_t p = anchors[ai];
        size_t k = members[p];
        bool dominant = dist[k] >= 2.0 * rm;
        const auto& plan = deep_layout ? record_plan(8, dominant) : record_plan(m, dominant);
        const Vec3& a = exact[k];
        auto& out = per[ai];
        if (star_id[k] == UINT32_MAX) continue;
        auto [east, north] = sphere::tangent_basis(a);
        uint32_t dcode = std::min<uint32_t>(sat_cast<uint32_t>(fmax_r(std::floor(2.0 * std::log2(dist[k] / rm)), 0.0)), 7);
        auto emit = [&](const std::vector<Vec3>& nbv, const std::vector<uint32_t>& nbk) -> bool {
          std::array<std::array<double, 2>, 5> xy{};
          for (size_t t = 0; t < nbv.size(); ++t) {
            auto g = sphere::gnomonic(nbv[t], a, east, north);
            if (!g) return false;
            xy[t] = *g;
          }
          for (const PlanEntry& pe : plan) {
            auto ck = fingerprint::canonical_key({0.0, 0.0}, xy[pe.ref], {xy[pe.sub[0]], xy[pe.sub[1]], xy[pe.sub[2]]});
            if (!ck) continue;
            uint32_t key = ck->key;
            bool flip = ck->flip;
            double len = ck->len;
            if (coder) {
              learned::Raw raw;
              raw.xy = {xy[pe.ref], xy[pe.sub[0]], xy[pe.sub[1]], xy[pe.sub[2]]};
              raw.d = {dist[k], dist[nbk[pe.ref]], dist[nbk[pe.sub[0]]], dist[nbk[pe.sub[1]]], dist[nbk[pe.sub[2]]]};
              raw.sigma = 0.0;
              raw.edge = INF;
              auto kf = coder->catalogue_key(raw);
              if (!kf) continue;
              key = kf->first;
              flip = kf->second;
            }
            double ar = std::atan(len);
            double lr = std::log2(ar / rm);
            if (!(lr >= 0.0 && lr < 4.0)) continue;
            uint32_t ar_code = std::min<uint32_t>(sat_cast<uint32_t>(std::floor(lr * index::AR_SCALE)), 1023);
            double pa = rem_euclid(std::atan2(xy[pe.ref][0], xy[pe.ref][1]), TAU);
            uint32_t pa_code = sat_cast<uint32_t>(std::floor(pa / TAU * 1024.0)) % 1024;
            auto [bucket, fp] = index::bucket_fp_layout(key, deep_layout);
            Record rec = deep_layout ? Record::make_deep(fp, star_id[k], ar_code, pa_code, flip, m, dominant, dcode)
                                     : Record::make(fp, star_id[k], ar_code, pa_code, flip, m, dominant, dcode);
            out.emplace_back(uint32_t(bucket), rec.raw);
          }
          return true;
        };
        if (!knn(grid, midx, mexact, a, p, 5, 3.0 * rm, cells, cand)) continue;
        std::vector<uint32_t> prev;
        std::vector<Vec3> nbv;
        for (auto& x : cand) {
          prev.push_back(members[x.second]);
          nbv.push_back(mexact[x.second]);
        }
        emit(nbv, prev);
        float maxmag = std::numeric_limits<float>::lowest();
        for (auto& x : cand) maxmag = fmax_r(maxmag, mmag[x.second]);
        float amag = mmag[p];
        for (size_t ci = 0; ci < capped.size(); ++ci) {
          const Capped& cp = capped[ci];
          if (cp.c >= maxmag || amag > cp.c) continue;
          uint32_t self_pos = cp.pos[p];
          if (!knn(grid, cp.ci, cp.sexact, a, self_pos, 5, 3.0 * rm, cells, cand)) break;
          std::vector<uint32_t> ids;
          for (auto& x : cand) ids.push_back(members[cp.sub[x.second]]);
          if (ids == prev) continue;
          std::vector<Vec3> nbv2;
          for (auto& x : cand) nbv2.push_back(cp.sexact[x.second]);
          size_t before = out.size();
          emit(nbv2, ids);
          n_capped.fetch_add(out.size() - before, std::memory_order_relaxed);
          maxmag = std::numeric_limits<float>::lowest();
          for (auto& x : cand) maxmag = fmax_r(maxmag, mmag[cp.sub[x.second]]);
          prev = ids;
        }
      }
    });
    std::vector<std::pair<uint32_t, uint64_t>> recs;
    for (auto& v : per) recs.insert(recs.end(), v.begin(), v.end());
    per.clear();
    log(t0, "level " + std::to_string(m) + ": " + std::to_string(n_capped.load()) + " depth-truncated records");
    {
      char buf[160];
      std::snprintf(buf, sizeof buf, "level %u: R=%.3f' members=%zu anchors=%zu records=%zu", m, to_degrees(rm) * 60.0, members.size(), anchors.size(), recs.size());
      log(t0, buf);
    }
    index::LevelInfo li;
    li.m = m;
    li.radius_arcmin = to_degrees(rm) * 60.0;
    li.stars = members.size();
    li.anchors = anchors.size();
    li.records = recs.size();
    levels.push_back(li);
    all.insert(all.end(), recs.begin(), recs.end());
  }
  {
    uint64_t n0 = 0;
    double r0 = index::level_radius(r7, 0);
    for (double x : dist)
      if (x >= r0) ++n0;
    index::LevelInfo li;
    li.m = 0;
    li.radius_arcmin = to_degrees(r0) * 60.0;
    li.stars = n0;
    levels.insert(levels.begin(), li);
  }
  uint32_t fp_bits = deep_layout ? 10 : 12;
  uint64_t fp_mask = (uint64_t(1) << fp_bits) - 1;
  auto sort_key = [&](const std::pair<uint32_t, uint64_t>& x) { return (uint64_t(x.first) << fp_bits) | (x.second & fp_mask); };
  std::stable_sort(all.begin(), all.end(), [&](const auto& a, const auto& b) { return sort_key(a) < sort_key(b); });
  std::vector<std::pair<uint32_t, uint64_t>> kept;
  kept.reserve(all.size());
  uint64_t stop = 0;
  size_t i = 0;
  while (i < all.size()) {
    uint64_t key = sort_key(all[i]);
    size_t j = i;
    while (j < all.size() && sort_key(all[j]) == key) ++j;
    if (j - i <= cfg.max_bucket) kept.insert(kept.end(), all.begin() + long(i), all.begin() + long(j));
    else stop += j - i;
    i = j;
  }
  all.clear();
  all.shrink_to_fit();
  uint64_t n_records = kept.size();
  write_codes(cfg.out, kept, deep_layout ? (size_t(1) << index::DEEP_BUCKET_BITS) : index::N_BUCKETS);
  log(t0, "codes.bin: " + std::to_string(n_records) + " records (" + std::to_string(stop) + " dropped as stop words)");
  index::Manifest manifest;
  for (const char* f : {"codes.bin", "keys.bin", "stars.bin", "cells.bin", "deep.bin", "deep_cells.bin"}) {
    std::error_code ec;
    uint64_t len = fs::file_size(cfg.out / f, ec);
    if (!ec) manifest.files.emplace_back(f, len);
  }
  manifest.format = "skyanchor2-index";
  manifest.version = deep_layout ? index::FORMAT_VERSION_DEEP : (coder ? index::FORMAT_VERSION_LEARNED : index::FORMAT_VERSION);
  manifest.profile = cfg.profile;
  manifest.r7_arcmin = cfg.r7_arcmin;
  uint32_t bucket_bits = deep_layout ? index::DEEP_BUCKET_BITS : index::BUCKET_BITS;
  if (!coder) {
    manifest.key_spec = "v1 rho_step=" + fmt_g(fingerprint::RHO_STEP) + " rho_bins=" + std::to_string(fingerprint::RHO_BINS) +
                        " theta_bins=" + std::to_string(fingerprint::THETA_BINS) + " bucket_bits=" + std::to_string(bucket_bits) +
                        " fp_bits=" + std::to_string(fp_bits);
  } else {
    std::string bits = "[";
    auto b = coder->bits();
    for (size_t t = 0; t < b.size(); ++t) bits += (t ? ", " : "") + std::to_string(b[t]);
    bits += "]";
    manifest.key_spec = std::string(learned::KEY_TYPE) + " bits=" + bits + " bucket_bits=" + std::to_string(bucket_bits) + " fp_bits=" + std::to_string(fp_bits);
  }
  manifest.levels = levels;
  manifest.n_records = n_records;
  manifest.n_stars = l7.size();
  manifest.n_catalog = n_cat;
  manifest.star_cell_n = index::STAR_CELL_N;
  manifest.catalog = cfg.d80.string();
  manifest.catalog_fingerprint = fingerprint;
  manifest.build_seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  manifest.stop_words = stop;
  manifest.key_type = coder ? learned::KEY_TYPE : "v1";
  if (coder) manifest.coder_sha256 = coder->sha256;
  std::string js = manifest.to_json().dump(true);
  write_bytes(cfg.out / "manifest.json", js.data(), js.size());
  log(t0, "manifest written");
  return manifest;
}

}  // namespace

index::Manifest write_index(const BuildConfig& cfg, std::chrono::steady_clock::time_point t0, const std::vector<Vec3>& exact,
                            const std::vector<double>& dist, const std::vector<float>& mags, uint64_t n_cat, const std::string& fingerprint,
                            std::optional<std::pair<Vec3, double>> anchor_cone) {
  return write_index_impl(cfg, t0, exact, dist, mags, n_cat, fingerprint, anchor_cone);
}

index::Manifest build(const BuildConfig& cfg) {
  auto t0 = Clock::now();
  std::error_code ec;
  if (cfg.profile == "deep" && fs::exists(cfg.out / "manifest.json", ec))
    throw Error("DEEP index already exists at " + cfg.out.string() + "; build a replacement in a new directory");
  fs::create_directories(cfg.out, ec);
  if (ec) throw Error("create " + cfg.out.string());
  auto [cat, fingerprint] = load_catalog(cfg.d80, t0);
  double r7 = to_radians(cfg.r7_arcmin) / 60.0;
  double r8 = index::level_radius(r7, 8);
  auto d = dominance(cat, r8, t0);
  std::vector<uint32_t> surv;
  for (uint32_t i : cat.order)
    if (std::isfinite(d[i])) surv.push_back(i);
  uint64_t n_cat = cat.raw.size();
  log(t0, "L8 = " + std::to_string(surv.size()) + " stars");
  std::vector<Vec3> exact(surv.size());
  parallel_chunks(surv.size(), 1 << 14, [&](size_t a, size_t b) {
    for (size_t t = a; t < b; ++t) exact[t] = sphere::unit(cat.raw[surv[t]].ra_deg(), cat.raw[surv[t]].dec_deg());
  });
  std::vector<double> dist(surv.size());
  std::vector<float> mags(surv.size());
  for (size_t k = 0; k < surv.size(); ++k) {
    dist[k] = double(d[surv[k]]);
    mags[k] = cat.raw[surv[k]].mag();
  }
  d.clear();
  d.shrink_to_fit();
  if (cfg.dump_levels) {
    std::vector<uint8_t> buf;
    auto put = [&](const void* p, size_t n) { buf.insert(buf.end(), static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + n); };
    for (size_t k = 0; k < surv.size(); ++k) {
      const auto& s = cat.raw[surv[k]];
      double ra = s.ra_deg(), dec = s.dec_deg();
      float dd = float(to_degrees(dist[k])), mg = s.mag();
      put(&ra, 8);
      put(&dec, 8);
      put(&dd, 4);
      put(&mg, 4);
    }
    write_bytes(cfg.out / "levels8.bin", buf.data(), buf.size());
    std::string hdr = "{\"count\":" + std::to_string(surv.size()) +
                      ",\"fields\":\"ra f64 deg, dec f64 deg, D f32 deg, mag f32 (BP); brightness order\"}";
    write_bytes(cfg.out / "levels8.json", hdr.data(), hdr.size());
    log(t0, "levels8.bin written");
  }
  cat = Catalog{};
  return write_index_impl(cfg, t0, exact, dist, mags, n_cat, fingerprint, std::nullopt);
}

index::Manifest build_region(const BuildConfig& cfg, double ra, double dec, double radius_deg) {
  std::error_code ec;
  if (cfg.profile == "deep" && fs::exists(cfg.out / "manifest.json", ec))
    throw Error("DEEP index already exists at " + cfg.out.string() + "; use a new output directory");
  auto t0 = Clock::now();
  fs::create_directories(cfg.out, ec);
  double r7 = to_radians(cfg.r7_arcmin) / 60.0;
  double margin = 2.5 * to_degrees(index::level_radius(r7, 1));
  double total = fmin_r(radius_deg + margin, 30.0);
  auto db = d80::D80::open(cfg.d80);
  auto near = db.nearby(ra, dec, total);
  std::vector<CatalogStar> stars;
  for (auto& p : near) stars.push_back(p.second);
  std::stable_sort(stars.begin(), stars.end(), [](const CatalogStar& a, const CatalogStar& b) {
    if (a.mag < b.mag) return true;
    if (a.mag > b.mag) return false;
    return a.source_id < b.source_id;
  });
  uint64_t n_cat = stars.size();
  {
    char buf[96];
    std::snprintf(buf, sizeof buf, "region: %zu stars within %.2f deg", stars.size(), total);
    log(t0, buf);
  }
  Vec3 c = sphere::unit(ra, dec);
  auto [east, north] = sphere::tangent_basis(c);
  std::vector<Vec3> vecs;
  for (auto& s : stars) vecs.push_back(sphere::unit(s.ra, s.dec));
  double k = 180.0 / PI * 3600.0;
  double ext = std::tan(to_radians(total)) * k;
  std::vector<Vec2> pts;
  for (auto& v : vecs) {
    auto g = sphere::gnomonic(v, c, east, north).value_or(Vec2{NaN, NaN});
    pts.push_back({g[0] * k + ext, g[1] * k + ext});
  }
  auto d_arcsec = select::dominance_px(pts, 2.0 * ext, 2.0 * ext);
  double r8 = index::level_radius(r7, 8);
  std::vector<size_t> keep;
  for (size_t i = 0; i < stars.size(); ++i)
    if (std::isfinite(pts[i][0]) && (d_arcsec[i] / k) >= r8) keep.push_back(i);
  std::vector<Vec3> exact;
  std::vector<double> dist;
  std::vector<float> mags;
  for (size_t i : keep) {
    exact.push_back(vecs[i]);
    dist.push_back(fmin_r(d_arcsec[i] / k, PI));
    mags.push_back(stars[i].mag);
  }
  log(t0, "region: " + std::to_string(exact.size()) + " stars with D >= R_8");
  char fp[512];
  std::snprintf(fp, sizeof fp, "region ra=%.5f dec=%.5f r=%.3f d80=%s", ra, dec, radius_deg, cfg.d80.string().c_str());
  return write_index_impl(cfg, t0, exact, dist, mags, n_cat, fp, std::make_pair(c, to_radians(radius_deg)));
}

}  // namespace skyshot::builder
