#include "skyshot/pipeline.hpp"

#include "skyshot/builder.hpp"
#include "skyshot/highpass.hpp"
#include "skyshot/registration.hpp"

namespace skyshot::pipeline {

using solve::Clock;
using solve::ms_since;
using solve::SolveConfig;

std::optional<double> attempt_share(const std::string& name) {
  for (auto& a : ATTEMPTS)
    if (name == a.name) return a.share;
  return std::nullopt;
}

std::optional<std::string> next_attempt(const std::string& name) {
  for (size_t i = 0; i < 3; ++i)
    if (name == ATTEMPTS[i].name) return i + 1 < 3 ? std::optional<std::string>(ATTEMPTS[i + 1].name) : std::nullopt;
  return std::nullopt;
}

std::optional<StarField> attempt_field(const image_io::GrayImage& gray, const image_io::SourceInfo& info, const std::string& name) {
  if (name == "default") return extract(gray, ExtractOptions::for_source(info));
  if (name == "local_background_compact") {
    ExtractOptions o = ExtractOptions::for_source(info);
    o.local_background = true;
    o.compact = true;
    return extract(gray, o);
  }
  if (name == "highpass_dog") return highpass::detect_highpass(gray, 2.5f, 5.0f);
  return std::nullopt;
}

Json PipelineTimings::to_json() const {
  Json o = Json::object();
  o.set("decode_ms", decode_ms).set("extract_ms", extract_ms);
  if (mask_ms) o.set("mask_ms", *mask_ms);
  o.set("solve_ms", solve_ms).set("total_ms", total_ms).set("attempts", attempts);
  o.set("fallback", fallback ? Json(*fallback) : Json());
  if (queue_ms) o.set("queue_ms", *queue_ms);
  if (network_ms) o.set("network_ms", *network_ms);
  if (tracked) o.set("tracked", true);
  return o;
}

static bool windows_for(const SolveConfig& cfg, const std::string& name, const SkyMask* mask) {
  return cfg.windows && (name == ATTEMPTS[0].name || (mask && mask->is_landscape()));
}

std::pair<solve::Solution, size_t> solve_attempts(const index::Index& idx, const std::vector<std::pair<std::string, const StarField*>>& attempts,
                                                  const SolveConfig& cfg) {
  if (attempts.empty()) throw Error("solve_attempts needs at least one star list");
  auto t0 = Clock::now();
  std::optional<std::pair<solve::Solution, size_t>> last;
  for (size_t i = 0; i < attempts.size(); ++i) {
    double remaining = cfg.deadline_ms - ms_since(t0);
    if (remaining <= 5.0 && last) break;
    double share = i + 1 == attempts.size() ? 1.0 : attempt_share(attempts[i].first).value_or(1.0);
    double budget = share < 1.0 ? fmin_r(cfg.deadline_ms * share, remaining) : remaining;
    SolveConfig c = cfg;
    c.deadline_ms = budget;
    c.windows = windows_for(cfg, attempts[i].first, cfg.mask.get());
    solve::Solution sol = solve::solve_field(idx, *attempts[i].second, c);
    bool solved = sol.status == "solved";
    last = std::make_pair(std::move(sol), i);
    if (solved) break;
  }
  return std::move(*last);
}

PipelineResult solve_gray(const index::Index& idx, const image_io::GrayImage& gray, const image_io::SourceInfo& info,
                          const SolveConfig& cfg_in) {
  SolveConfig cfg = cfg_in;
  cfg.hint = SolveHint::merge(cfg_in.hint, info.hint);
  auto t0 = Clock::now();
  PipelineTimings tm;
  std::optional<std::pair<solve::Solution, StarField>> last;
  std::shared_ptr<const SkyMask> mask = cfg.mask;
  std::optional<StarField> default_field;
  size_t spent = 0;
  for (const Attempt& at : ATTEMPTS) {
    std::string name = at.name;
    double share = at.share;
    double elapsed = ms_since(t0);
    double remaining = cfg.deadline_ms - elapsed;
    if (remaining <= 5.0 && last) break;
    auto te = Clock::now();
    auto fieldo = attempt_field(gray, info, name);
    if (!fieldo) continue;
    StarField field = std::move(*fieldo);
    if (!mask && cfg.auto_mask && !last) {
      auto tm0 = Clock::now();
      mask = skymask::estimate(gray, field);
      tm.mask_ms = ms_since(tm0);
    }
    tm.extract_ms += ms_since(te);
    remaining = fmax_r(cfg.deadline_ms - ms_since(t0), 0.0);
    double budget = share < 1.0 ? fmin_r(cfg.deadline_ms * share, remaining) : remaining;
    SolveConfig c = cfg;
    c.deadline_ms = budget;
    c.windows = cfg.windows && mask && mask->is_landscape();
    c.mask = mask;
    auto ts = Clock::now();
    solve::Solution sol = solve::solve_field(idx, field, c);
    tm.solve_ms += ms_since(ts);
    tm.attempts += 1;
    spent += sol.stats.verifications;
    bool solved = sol.status == "solved";
    if (name == ATTEMPTS[0].name) default_field = field;
    bool ambiguous = sol.status == "ambiguous";
    last = std::make_pair(std::move(sol), std::move(field));
    if (solved || ambiguous) {
      if (solved && name != ATTEMPTS[0].name) tm.fallback = name;
      break;
    }
  }
  double remaining = cfg.deadline_ms - ms_since(t0);
  if (cfg.windows && cfg.mode != SolveMode::Deep && idx.manifest.profile != "deep" && remaining > 60.0 && last &&
      last->first.status == "unsolved" && !(mask && mask->is_landscape()) && default_field) {
    auto ts = Clock::now();
    SolveConfig c = cfg;
    c.deadline_ms = (cfg.allow_panorama && double(default_field->width) > 1.8 * double(default_field->height)) ? remaining
                                                                                                                 : fmin_r(remaining, FINAL_WINDOWS_MS);
    c.mask = mask;
    solve::Solution sol = solve::solve_windows_only(idx, *default_field, c, spent);
    tm.solve_ms += ms_since(ts);
    if (sol.status != "unsolved") {
      tm.fallback = "windows";
      last = std::make_pair(std::move(sol), *default_field);
    }
  }
  tm.total_ms = ms_since(t0);
  if (!last) throw Error("at least one attempt");
  solve::Solution solution = std::move(last->first);
  StarField field = std::move(last->second);
  if (solution.stats.deep_search) mask.reset();
  auto refine_start = Clock::now();
  registration::refine_solution(idx, solution, field, cfg, mask.get(), cfg.deadline_ms - tm.total_ms);
  tm.solve_ms += ms_since(refine_start);
  tm.total_ms = ms_since(t0);
  PipelineResult r;
  r.solution = std::move(solution);
  r.timings = tm;
  r.n_stars = field.stars.size();
  r.mask = mask;
  r.stars = std::move(field);
  return r;
}

PipelineResult solve_gray_tracked(const index::Index& idx, const image_io::GrayImage& gray, const image_io::SourceInfo& info,
                                  const SolveConfig& cfg_in, const Prior* prior) {
  if (!prior) return solve_gray(idx, gray, info, cfg_in);
  auto t0 = Clock::now();
  SolveConfig cfg = cfg_in;
  if (!cfg.mask) cfg.mask = prior->mask;
  auto te = Clock::now();
  StarField field = extract(gray, ExtractOptions::for_source(info));
  double extract_ms = ms_since(te);
  auto ts = Clock::now();
  solve::Solution sol = solve::track_field(idx, field, prior->wcs, prior->search_px, prior->refine, cfg);
  double solve_ms = ms_since(ts);
  auto finish = [&](solve::Solution s, double sms, bool tracked) {
    PipelineResult r;
    r.solution = std::move(s);
    r.timings.extract_ms = extract_ms;
    r.timings.solve_ms = sms;
    r.timings.total_ms = ms_since(t0);
    r.timings.attempts = 1;
    r.timings.tracked = tracked;
    r.n_stars = field.stars.size();
    r.mask = cfg.mask;
    r.stars = field;
    return r;
  };
  if (sol.status == "solved") {
    auto rs = Clock::now();
    registration::refine_solution(idx, sol, field, cfg, cfg.mask.get(), cfg.deadline_ms - ms_since(t0));
    solve_ms += ms_since(rs);
    return finish(std::move(sol), solve_ms, true);
  }
  if (prior->fallback_ms <= 0.0) return finish(std::move(sol), solve_ms, false);
  SolveConfig rest = cfg;
  rest.deadline_ms = fmax_r(fmin_r(cfg.deadline_ms - ms_since(t0), prior->fallback_ms), 50.0);
  rest.mask.reset();
  PipelineResult r = solve_gray(idx, gray, info, rest);
  r.timings.extract_ms += extract_ms;
  r.timings.solve_ms += solve_ms;
  r.timings.total_ms = ms_since(t0);
  return r;
}

PipelineResult solve_path(const index::Index& idx, const std::filesystem::path& path, const SolveConfig& cfg) {
  auto t0 = Clock::now();
  auto loaded = image_io::load_with_info(path);
  double decode_ms = ms_since(t0);
  PipelineResult r = solve_gray(idx, loaded.image, loaded.info, cfg);
  r.timings.decode_ms = decode_ms;
  r.timings.total_ms += decode_ms;
  return r;
}

PipelineResult solve_path_hinted(const index::Index& idx, const std::filesystem::path& path, const SolveConfig& cfg_in,
                                 const std::optional<Hint>& hint, const std::filesystem::path& d80, const std::filesystem::path& cache) {
  auto t0 = Clock::now();
  auto loaded = image_io::load_with_info(path);
  const image_io::GrayImage& gray = loaded.image;
  const image_io::SourceInfo& info = loaded.info;
  double decode_ms = ms_since(t0);
  SolveConfig cfg = cfg_in;
  {
    std::optional<SolveHint> from_hint;
    if (hint) {
      SolveHint h;
      h.position = Vec2{hint->ra, hint->dec};
      if (hint->fov_deg) h.scale_arcsec = *hint->fov_deg * 3600.0 / double(gray.height);
      from_hint = h;
    }
    cfg.hint = SolveHint::merge(from_hint, cfg_in.hint);
  }
  auto remaining = [&](double floor) { return fmax_r(cfg.deadline_ms - ms_since(t0), floor); };
  auto blind = [&](double deadline_ms) {
    SolveConfig c = cfg;
    c.deadline_ms = deadline_ms;
    PipelineResult r = solve_gray(idx, gray, info, c);
    r.timings.decode_ms = decode_ms;
    r.timings.total_ms = ms_since(t0);
    return r;
  };
  if (!hint) return blind(cfg.deadline_ms);
  auto deep = [&](double deadline_ms) {
    double radius = clampd(hint->radius_deg, 0.5, 10.0);
    char key[96];
    std::snprintf(key, sizeof key, "ra%.2f_dec%.2f_r%.1f", hint->ra, hint->dec, radius);
    std::filesystem::path dir = cache / key;
    std::error_code ec;
    if (!std::filesystem::exists(dir / "manifest.json", ec)) {
      builder::BuildConfig cb;
      cb.d80 = d80;
      cb.out = dir;
      cb.r7_arcmin = 0.485;
      cb.dump_levels = false;
      builder::build_region(cb, hint->ra, hint->dec, radius);
    }
    auto local = index::Index::open(dir);
    SolveConfig lc = cfg;
    lc.deadline_ms = deadline_ms;
    lc.min_fov_deg = 0.03;
    PipelineResult rl = solve_gray(*local, gray, info, lc);
    rl.timings.decode_ms = decode_ms;
    rl.timings.total_ms = ms_since(t0);
    rl.timings.fallback = "hinted_deep" + (rl.timings.fallback ? "+" + *rl.timings.fallback : std::string());
    return rl;
  };
  if (hint->fov_deg && *hint->fov_deg > 0.0 && *hint->fov_deg < DEEP_FIRST_FOV_DEG) {
    PipelineResult rl = deep(fmin_r(cfg.deadline_ms, 1500.0));
    if (rl.solution.status == "solved") return rl;
    PipelineResult r = blind(remaining(300.0));
    return r.solution.status == "solved" ? std::move(r) : std::move(rl);
  }
  PipelineResult r = blind(fmin_r(cfg.deadline_ms, 800.0));
  if (r.solution.status == "solved") return r;
  return deep(remaining(500.0));
}

}  // namespace skyshot::pipeline
