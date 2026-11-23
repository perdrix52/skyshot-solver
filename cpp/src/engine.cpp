#include "skyshot/engine.hpp"

namespace skyshot {

Json SolveOutput::to_json() const {
  Json o = Json::object();
  o.set("schema_version", schema_version).set("solution", solution.to_json()).set("timings", timings.to_json()).set("n_stars", uint64_t(n_stars));
  return o;
}

Engine Engine::open(const std::filesystem::path& path) {
  Engine e;
  e.index_ = index::Index::open(path);
  if (auto dd = index::companion_deep_dir(path)) {
    auto idx = index::Index::open(*dd);
    if (idx->manifest.version != index::FORMAT_VERSION_DEEP) throw Error(dd->string() + " is not a DEEP companion index");
    e.deep_ = idx;
  }
  if (auto wd = index::companion_wide_dir(path)) e.wide_ = index::Index::open(*wd);
  return e;
}

solve::SolveConfig Engine::config(const solve::SolveConfig& cfg_in) const {
  if (!(std::isfinite(cfg_in.deadline_ms) && cfg_in.deadline_ms > 0.0)) throw Error("deadline_ms must be finite and positive");
  if (!(std::isfinite(cfg_in.min_fov_deg) && std::isfinite(cfg_in.max_fov_deg) && cfg_in.min_fov_deg > 0.0 && cfg_in.max_fov_deg >= cfg_in.min_fov_deg))
    throw Error("invalid field-of-view bounds");
  solve::SolveConfig cfg = cfg_in;
  if (!cfg.deep_index) cfg.deep_index = deep_;
  if (!cfg.wide_index) cfg.wide_index = wide_;
  if (cfg.mode == SolveMode::Deep && !cfg.deep_index && index_->manifest.profile != "deep")
    throw Error("DEEP mode requires a compatible index-deep companion or a DEEP main index");
  return cfg;
}

SolveOutput Engine::solve_path(const std::filesystem::path& path, const solve::SolveConfig& cfg) const {
  auto c = config(cfg);
  auto result = pipeline::solve_path(*index_, path, c);
  SolveOutput o;
  o.solution = std::move(result.solution);
  o.timings = result.timings;
  o.n_stars = result.n_stars;
  return o;
}

SolveOutput Engine::solve_sources(const StarField& field, const solve::SolveConfig& cfg_in) const {
  if (!(field.width >= 16 && field.width <= 50000 && field.height >= 16 && field.height <= 50000))
    throw Error("source-list dimensions must be between 16 and 50000 pixels");
  for (const Star& s : field.stars)
    if (!(std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.flux) && std::isfinite(s.sigma) && s.sigma > 0.0))
      throw Error("source positions, flux and uncertainties must be finite; sigma must be positive");
  auto cfg = config(cfg_in);
  auto t0 = solve::Clock::now();
  auto solution = solve::solve_field(*index_, field, cfg);
  double elapsed = solve::ms_since(t0);
  SolveOutput o;
  o.solution = std::move(solution);
  o.timings.solve_ms = elapsed;
  o.timings.total_ms = elapsed;
  o.timings.attempts = 1;
  o.n_stars = field.stars.size();
  return o;
}

}  // namespace skyshot
