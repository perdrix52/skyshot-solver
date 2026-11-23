// Blind solver. Port of src/solve.rs (and the public parts of src/api).
#pragma once
#include <chrono>
#include <memory>

#include "hints.hpp"
#include "index.hpp"
#include "json.hpp"
#include "skymask.hpp"
#include "types.hpp"

namespace skyshot {

enum class SolveMode { Auto, Standard, Deep };
const char* solve_mode_name(SolveMode m);

namespace solve {

struct SolveConfig {
  std::optional<SolveHint> hint;
  SolveMode mode = SolveMode::Auto;
  std::shared_ptr<const index::Index> deep_index;
  double min_fov_deg = 0.15;
  double max_fov_deg = 75.0;
  double deadline_ms = 2000.0;
  size_t max_stars = 6000;
  size_t target_pool = 150;
  size_t min_pool = 10;
  size_t max_pool = 800;
  size_t trigger_anchors = 2;
  double rung_tol = 1.0;
  double dom_tol = 0.2;
  double member_tol = 0.3;
  size_t n_refs = 3;
  size_t n_rest = 5;
  bool exhaustive = true;
  double min_log_odds = 30.0;
  bool keep_matches = false;
  size_t learned_k = 8;
  double min_margin_nats = 0.0;
  std::optional<float> mag_limit;
  double continuation_factor = 0.5;
  std::shared_ptr<const SkyMask> mask;
  bool windows = true;
  size_t extra_trials = 0;
  bool allow_arc = true;
  bool allow_local_grid = false;
  bool allow_panorama = true;
  bool auto_mask = true;
  std::shared_ptr<const index::Index> wide_index;
};

inline constexpr std::pair<double, double> WIDE_INDEX_FOV{10.0, 100.0};

struct SolveStats {
  bool hint_used = false, hint_fallback = false, deep_search = false;
  size_t n_stars = 0, rungs_tried = 0, anchors = 0, keys = 0, probes = 0, records = 0, hits = 0, bins = 0;
  size_t verifications = 0, cheap_rejects = 0, stage2 = 0;
  bool exhaustive_pass = false;
  size_t branch2 = 0;
  double keygen_ms = 0, select_ms = 0, probe_ms = 0, verify_ms = 0, total_ms = 0;
  Json to_json() const;
};

struct MatchOut {
  double x, y, ra, dec, residual_px;
  Json to_json() const;
};

struct Solution {
  std::string status;
  uint32_t width = 0, height = 0;
  std::optional<Wcs> wcs;
  double ra = NaN, dec = NaN, scale_arcsec = NaN, rotation_deg = NaN;
  int32_t parity = 0;
  double fov_width_deg = NaN;
  size_t n_matches = 0, n_catalog = 0;
  double rms_px = NaN, log_odds = 0.0;
  size_t cluster_anchors = 0;
  uint32_t verify_level = 0;
  SolveStats stats;
  std::vector<MatchOut> matches;
  double margin_nats = NaN;
  uint32_t n_candidates = 0;
  std::optional<std::array<double, 4>> window;
  std::optional<Refraction> refraction;
  std::optional<std::array<double, 4>> sky_extent;
  Json to_json() const;
};

Solution blank_solution(const std::string& status, uint32_t width, uint32_t height);

Solution solve_field(const index::Index& idx, const StarField& field, const SolveConfig& cfg);
Solution track_field(const index::Index& idx, const StarField& field, const Wcs& prior, double search_px, bool refine,
                     const SolveConfig& cfg);
Solution solve_windows_only(const index::Index& idx, const StarField& field, const SolveConfig& cfg, size_t spent_verifications);

double poisson_log_odds(size_t k, double lambda);
double log_binom_tail(size_t n, size_t k, double p);

using Clock = std::chrono::steady_clock;
inline double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

}  // namespace solve
}  // namespace skyshot
