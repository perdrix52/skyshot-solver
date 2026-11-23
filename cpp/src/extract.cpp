// Star extraction driver. Port of the top level of src/extract.rs.
#include "skyshot/extract.hpp"

#include <algorithm>
#include <set>

#include "extract_internal.hpp"
#include "skyshot/parallel.hpp"

namespace skyshot {

using namespace extract_detail;

size_t bin_factor(size_t w, size_t h, const ExtractOptions& opts) {
  if (opts.bin > 0) return std::min<size_t>(opts.bin, 8);
  if (opts.compact) return 1;
  size_t px = w * h;
  if (px <= 6500000) return 1;
  if (px <= 26500000) return 2;
  return 4;
}

float detect_lattice(const std::vector<float>& px) {
  size_t step = std::max<size_t>(px.size() / 4096, 1);
  std::vector<float> sample;
  for (size_t i = 0; i < px.size(); i += step)
    if (std::isfinite(px[i])) sample.push_back(px[i]);
  if (sample.size() < 64) return 0.0f;
  const float qs[3] = {1.0f, 1.0f / 255.0f, 1.0f / 65535.0f};
  for (float q : qs) {
    float lo = INF, hi = -INF;
    bool on_lattice = true;
    for (float v : sample) {
      float t = v / q;
      float r = std::round(t);
      lo = fmin_r(lo, r);
      hi = fmax_r(hi, r);
      if (!(std::fabs(t - r) < 2e-3f + 1e-6f * std::fabs(t))) {
        on_lattice = false;
        break;
      }
    }
    if (on_lattice && hi - lo >= 2.0f) return q;
  }
  return 0.0f;
}

namespace {

float block_mean(const float* px, size_t w, size_t x0, size_t y0, size_t b) {
  float s = 0.0f;
  int n = 0;
  for (size_t y = y0; y < y0 + b; ++y)
    for (size_t x = x0; x < x0 + b; ++x) {
      float v = px[y * w + x];
      if (std::isfinite(v)) {
        s += v;
        n += 1;
      }
    }
  return n == 0 ? std::numeric_limits<float>::quiet_NaN() : s / float(n);
}

void make_plane(const image_io::GrayImage& image, size_t b, Plane& plane) {
  size_t w = image.width, h = image.height;
  if (b <= 1) {
    plane.w = w;
    plane.h = h;
    plane.data = image.pixels.data();
    return;
  }
  size_t bw = w / b, bh = h / b;
  plane.owned.assign(bw * bh, 0.0f);
  const float* px = image.pixels.data();
  size_t rows_per_task = std::max<size_t>(16384 / std::max<size_t>(bw, 1), 1);
  size_t tasks = (bh + rows_per_task - 1) / rows_per_task;
  float* out = plane.owned.data();
  parallel_for(tasks, [&](size_t task) {
    for (size_t r = 0; r < rows_per_task; ++r) {
      size_t by = task * rows_per_task + r;
      if (by >= bh) break;
      size_t base = by * b * w;
      float* orow = out + by * bw;
      if (b == 2) {
        const float* r0 = px + base;
        const float* r1 = px + base + w;
        for (size_t bx = 0; bx < bw; ++bx) {
          float s = r0[2 * bx] + r0[2 * bx + 1] + r1[2 * bx] + r1[2 * bx + 1];
          orow[bx] = std::isfinite(s) ? s * 0.25f : block_mean(px, w, bx * 2, by * 2, 2);
        }
      } else {
        float inv = 1.0f / float(b * b);
        for (size_t bx = 0; bx < bw; ++bx) {
          float s = 0.0f;
          for (size_t dy = 0; dy < b; ++dy) {
            const float* row = px + base + dy * w + bx * b;
            float rs = -0.0f;
            for (size_t k = 0; k < b; ++k) rs += row[k];
            s += rs;
          }
          orow[bx] = std::isfinite(s) ? s * inv : block_mean(px, w, bx * b, by * b, b);
        }
      }
    }
  });
  plane.w = bw;
  plane.h = bh;
  plane.data = plane.owned.data();
}

std::vector<Cand> find_peaks(const Plane& plane, const Grid& grid, float k) {
  size_t w = plane.w, h = plane.h;
  size_t rows_per_task = std::clamp<size_t>(32768 / std::max<size_t>(w, 1), 1, 64);
  size_t tasks = (h - 2 + rows_per_task - 1) / rows_per_task;
  const float* data = plane.data;
  std::vector<std::vector<Cand>> per(tasks);
  parallel_for(tasks, [&](size_t task) {
    auto& out = per[task];
    std::vector<float> bg_row(grid.nx), sg_row(grid.nx), bound(std::max<size_t>(grid.nx, 2) - 1);
    size_t y_start = 1 + task * rows_per_task;
    size_t y_end = std::min(y_start + rows_per_task, h - 1);
    for (size_t y = y_start; y < y_end; ++y) {
      grid.row_cubic(float(y), bg_row.data());
      grid.row_sigma(float(y), sg_row.data());
      size_t nx = grid.nx;
      auto node = [&](int64_t j) -> float {
        if (nx == 1) return bg_row[0];
        if (j < 0) return 2.0f * bg_row[0] - bg_row[1];
        if (size_t(j) >= nx) return 2.0f * bg_row[nx - 1] - bg_row[nx - 2];
        return bg_row[size_t(j)];
      };
      for (size_t i = 0; i < bound.size(); ++i) {
        float lo = INF, hi = -INF;
        for (int64_t j = int64_t(i) - 1; j <= int64_t(i) + 2; ++j) {
          float v = node(j);
          lo = fmin_r(lo, v);
          hi = fmax_r(hi, v);
        }
        float sg = fmin_r(sg_row[i], sg_row[std::min(i + 1, nx - 1)]);
        bound[i] = lo - 0.5f * (hi - lo) + k * sg;
      }
      const float* up = data + (y - 1) * w;
      const float* row = data + y * w;
      const float* down = data + (y + 1) * w;
      size_t x0 = 1;
      while (x0 < w - 1) {
        size_t x1 = std::min(x0 + 16, w - 1);
        size_t ia = grid.xlin[x0].i0, ib = grid.xlin[x1 - 1].i0;
        float thr_min = fmin_r(bound[std::min(ia, bound.size() - 1)], bound[std::min(ib, bound.size() - 1)]);
        bool any = false;
        for (size_t x = x0; x < x1; ++x) any = any | (row[x] > thr_min);
        size_t bstart = x0;
        x0 = x1;
        if (!any) continue;
        for (size_t x = bstart; x < x1; ++x) {
          float v = row[x];
          if (std::isnan(v) || v <= thr_min) continue;
          float bg = grid.col_cubic(x, bg_row.data());
          float sg = grid.col_linear(x, sg_row.data());
          if (v <= bg + k * sg) continue;
          if (up[x - 1] >= v || up[x] >= v || up[x + 1] >= v || row[x - 1] >= v || row[x + 1] > v || down[x - 1] > v ||
              down[x] > v || down[x + 1] > v)
            continue;
          float s = bg + 2.0f * sg;
          const float sup[9] = {up[x - 1], up[x], up[x + 1], row[x - 1], v, row[x + 1], down[x - 1], down[x], down[x + 1]};
          int support = 0;
          for (float q : sup) support += int(q > s);
          if (support >= 3) out.push_back(Cand{uint32_t(x), uint32_t(y), v, bg, sg});
        }
      }
    }
  });
  std::vector<Cand> all;
  for (auto& p : per) all.insert(all.end(), p.begin(), p.end());
  return all;
}

void cap_per_tile(std::vector<Cand>& cands, const Grid& grid, size_t cap) {
  std::vector<uint32_t> count(grid.nx * grid.ny, 0);
  for (const Cand& c : cands) count[grid.tile_index(c.x, c.y)] += 1;
  bool all_ok = true;
  for (uint32_t n : count)
    if (size_t(n) > cap) all_ok = false;
  if (all_ok) return;
  std::vector<float> cutoff(count.size(), -INF);
  for (size_t t = 0; t < count.size(); ++t) {
    if (size_t(count[t]) > cap) {
      std::vector<float> vals;
      for (const Cand& c : cands)
        if (grid.tile_index(c.x, c.y) == t) vals.push_back(c.v - c.bg);
      size_t i = vals.size() - cap;
      std::nth_element(vals.begin(), vals.begin() + long(i), vals.end(), [](float a, float b) { return total_cmp(a, b) < 0; });
      cutoff[t] = vals[i];
    }
  }
  std::vector<Cand> kept;
  for (const Cand& c : cands)
    if (c.v - c.bg >= cutoff[grid.tile_index(c.x, c.y)]) kept.push_back(c);
  cands.swap(kept);
}

}  // namespace

namespace extract_detail {
void bin_plane(const image_io::GrayImage& image, size_t b, Plane& plane) { make_plane(image, b, plane); }
}  // namespace extract_detail

StarField extract(const image_io::GrayImage& image, const ExtractOptions& opts) {
  size_t w = image.width, h = image.height;
  StarField field;
  field.width = image.width;
  field.height = image.height;
  field.epoch = image.epoch;
  if (w < 8 || h < 8 || image.pixels.size() < w * h) return field;
  size_t b = bin_factor(w, h, opts);
  Plane plane;
  make_plane(image, b, plane);
  if (plane.w < 4 || plane.h < 4) return field;
  float lsb = opts.lsb > 0.0f ? opts.lsb : detect_lattice(image.pixels);
  auto grid = Grid::build(plane, image.pixels, w, b, lsb);
  if (!grid) return field;
  float k = fmax_r(opts.threshold_sigma, 0.5f);
  std::vector<Cand> cands = find_peaks(plane, *grid, k);
  cap_per_tile(cands, *grid, std::max<size_t>(opts.max_per_tile, 1));

  Ctx ctx;
  ctx.img = image.pixels.data();
  ctx.w = w;
  ctx.h = h;
  ctx.b = b;
  ctx.plane = &plane;
  ctx.grid = &*grid;
  ctx.lsb = lsb;
  ctx.k = k;
  float sat = -INF;
  for (const Cand& c : cands) sat = fmax_r(sat, c.v);
  ctx.sat_level = sat;
  ctx.local_background = opts.local_background;
  ctx.fwhm_frame = 0.0f;
  {
    std::vector<const Cand*> brightest;
    for (const Cand& c : cands) brightest.push_back(&c);
    size_t top = std::min<size_t>(brightest.size(), 64);
    if (top > 0 && top < brightest.size())
      std::nth_element(brightest.begin(), brightest.begin() + long(top), brightest.end(),
                       [](const Cand* a, const Cand* c) { return total_cmp(c->v, a->v) < 0; });
    float s = ctx.sat_level;
    for (size_t i = 0; i < top; ++i)
      if (auto pk = ctx.native_peak(*brightest[i])) s = fmax_r(s, ctx.px(pk->first, pk->second));
    ctx.sat_level = s;
  }
  ctx.fwhm_frame = 3.0f;
  FramePsf psf = frame_psf(ctx, cands);
  ctx.fwhm_frame = psf.fwhm;
  ctx.trail = psf.trail;

  std::vector<uint8_t> ridge(cands.size());
  parallel_for(cands.size(), [&](size_t i) { ridge[i] = ctx.is_ridge(cands[i]); });
  bool deblend = b >= 2 && !opts.compact;
  std::vector<std::optional<Det>> maybe(cands.size());
  parallel_for(cands.size(), [&](size_t i) {
    if (!(opts.keep_rejected || !ridge[i])) return;
    auto pk = ctx.native_peak(cands[i]);
    if (!pk) return;
    auto d = ctx.measure(pk->first, pk->second, opts.compact, INF, deblend);
    if (!d) return;
    if (ridge[i]) d->flags |= star_flags::RIDGE;
    maybe[i] = std::move(d);
  });
  std::vector<Det> dets;
  for (auto& m : maybe)
    if (m) dets.push_back(std::move(*m));
  std::vector<std::array<float, 2>> line_hints;
  for (size_t i = 0; i < cands.size(); ++i)
    if (ridge[i]) line_hints.push_back(ctx.to_native(float(cands[i].x), float(cands[i].y)));

  if (deblend) {
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const Det& d : dets) seen.insert({d.px, d.py});
    std::vector<std::pair<uint32_t, uint32_t>> extra;
    for (const Det& d : dets)
      for (const SecPt& s : d.secondary)
        if (s.has && seen.insert({s.x, s.y}).second) extra.emplace_back(s.x, s.y);
    std::vector<std::optional<Det>> more(extra.size());
    parallel_for(extra.size(), [&](size_t i) {
      auto m = ctx.measure(extra[i].first, extra[i].second, true, INF, false);
      if (m) {
        m->flags |= star_flags::BLENDED;
        more[i] = std::move(m);
      }
    });
    for (auto& m : more)
      if (m) dets.push_back(std::move(*m));
  }
  merge_duplicates(ctx, dets);
  merge_psf_lobes(ctx, dets);
  vet(ctx, dets);
  if (opts.remove_lines) flag_lines(ctx, dets, line_hints);
  if (!opts.keep_rejected) {
    std::vector<Det> kept;
    for (auto& d : dets)
      if ((d.flags & star_flags::REJECT) == 0) kept.push_back(std::move(d));
    dets.swap(kept);
  }
  float trail_len = ctx.trail ? ctx.trail->length : 0.0f;
  std::vector<Star> stars;
  stars.reserve(dets.size());
  for (const Det& d : dets) stars.push_back(d.to_star(trail_len));
  std::sort(stars.begin(), stars.end(), [](const Star& a, const Star& c) {
    int r = total_cmp(c.flux, a.flux);
    if (r != 0) return r < 0;
    r = total_cmp(a.y, c.y);
    if (r != 0) return r < 0;
    return total_cmp(a.x, c.x) < 0;
  });
  field.stars = std::move(stars);
  return field;
}

}  // namespace skyshot
