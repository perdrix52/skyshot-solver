// Internal solver state shared by solve.cpp, solve_wide.cpp and solve_panorama.cpp.
#pragma once
#include <unordered_map>
#include <unordered_set>

#include "skyshot/fingerprint.hpp"
#include "skyshot/geometry.hpp"
#include "skyshot/learned.hpp"
#include "skyshot/select.hpp"
#include "skyshot/solve.hpp"
#include "skyshot/sphere.hpp"
#include "skyshot/wcsfit.hpp"

namespace skyshot::solve {

using index::Index;
using wcsfit::Pair;
using wcsfit::Pairs;

constexpr uint32_t NONE = UINT32_MAX;

inline uint64_t hash_u64(uint64_t x) {
  uint64_t z = x + 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

struct Hit {
  uint32_t img_anchor = 0, cat_anchor = 0;
  Vec3 center{};
  double s = 0, phi = 0;
  int32_t parity = 0;
  uint32_t next = NONE;
};

struct Bin {
  std::array<uint32_t, 8> anchors{};
  uint8_t n_anchors = 0;
  uint32_t head = 0;
  uint32_t hits = 0;
  uint8_t verified_at = 0;
  int8_t parity = 0;
};

struct Candidate {
  Solution sol;
  Hit seed;
};

struct FrameCat {
  std::vector<Vec2> stars;
  uint32_t level = 0;
  Vec3 center{};
  double radius = 0;
};

using Match = std::pair<uint32_t, Vec2>;
using Matches = std::vector<Match>;
struct MatchResult {
  size_t n;
  Matches m;
};

struct Grown {
  Wcs wcs;
  Pairs pairs;
  double rms;
};

// (wcs, inlier pairs, rms, sip order)
struct Refined {
  Wcs wcs;
  Pairs pairs;
  double rms;
  uint32_t order;
};

MatchResult match_projected(const std::vector<Vec2>& pts, const select::Grid2& grid, const std::vector<std::pair<Vec2, Vec2>>& projected,
                            double radius);

class Solver {
 public:
  const Index& idx;
  index::ConeCache catalog_cache;
  std::optional<std::pair<size_t, select::Grid2>> inlier_grid;
  const SolveConfig& cfg;
  std::optional<std::pair<Vec3, double>> hint_cone;
  std::vector<Vec2> pts;
  std::vector<double> sig, d, edge;
  double w = 0, h = 0, open_area = 0;
  SolveStats stats;
  std::vector<Hit> hits;
  std::unordered_map<uint64_t, Bin> bins;
  Clock::time_point t0;
  std::vector<double> pa_sin, pa_cos;
  size_t n_refs, n_rest;
  learned::ElemCache elem_cache;
  std::vector<Candidate> cands;
  std::optional<double> accept_ms;

  Solver(const Index& idx_, const StarField& field, const SolveConfig& cfg_, Clock::time_point t0_);

  double elapsed_ms() const { return ms_since(t0); }
  std::optional<Refraction> measure_refraction(const Wcs& wcs, const Pairs& pairs) const;
  std::optional<std::array<double, 4>> sky_extent() const;
  double fov_width(const Wcs& wcs) const;
  double trials_ln() const { return std::log(double(std::max<size_t>(stats.verifications + cfg.extra_trials, 1))); }
  bool visible(Vec2 px) const;
  double frame_radius(const Wcs& wcs, const Vec3& c, double margin) const;
  double max_frame_radius(const Wcs& wcs, double tan_limit) const;
  std::optional<std::pair<Vec3, double>> hypothesis(Vec2 a_px, std::pair<double, double> dir, const Vec3& av, uint32_t pa_code,
                                                    double s, double p) const;
  uint64_t bin_key(const Vec3& center, double s, double phi, double p) const;
  std::optional<uint64_t> add_hit(const Hit& hit);
  std::optional<Solution> process_anchor(uint32_t a, const std::vector<std::pair<double, uint32_t>>& nb, double r,
                                         std::vector<fingerprint::KeyVariant>& variants, std::unordered_set<uint64_t>& seen);
  std::optional<std::pair<Solution, Hit>> verify_bin(uint64_t key);
  bool continuing() const { return accept_ms.has_value(); }
  bool continuation_over() const;
  bool matches_candidate(const Hit& hit) const;
  void accept(Solution sol, const Hit& seed);
  std::pair<size_t, double> margin() const;
  void sweep_pending_bins();
  std::optional<Solution> conclude();
  std::optional<Solution> verify_wcs(const Wcs& wcs0, size_t cluster);
  std::optional<Solution> verify_wcs_within(const Wcs& wcs0, size_t cluster, double min_r1);
  struct Refit {
    Wcs wcs;
    double rms;
    Pairs inl;
  };
  std::optional<Refit> refit_linear(const Wcs& prior, const Pairs& pairs) const;
  std::optional<Vec2> offset_vote(const Wcs& prior, const Pairs& cat, const select::Grid2& g, double radius) const;
  std::optional<Solution> verify_tracked(const Wcs& prior, double search_px);
  std::vector<double> weights(const Matches& m) const;
  Pairs pairs_of(const Matches& m) const;
  Refined refine(Wcs wcs0, double start_radius);
  size_t inliers_at(const Wcs& wcs, double tol);
  std::optional<Pairs> catalog_in_frame_margin(const Wcs& wcs, double rm, double margin);
  std::vector<index::StarInfo> catalog_cone(const Vec3& c, double radius, double rm);
  std::optional<Pairs> catalog_in_frame(const Wcs& wcs, double rm);
  MatchResult count_matches(const select::Grid2& g, const Pairs& cat, double radius, const Wcs& wcs) const;
  std::vector<MatchOut> match_out(const Pairs& pairs, const Wcs& wcs) const;

  // ---- wide.rs ----
  uint32_t frame_level(const Wcs& wcs) const;
  std::vector<Vec2> cone_radec(const Vec3& c, double radius, uint32_t level);
  std::optional<FrameCat> frame_catalog(const Wcs& wcs, uint32_t level);
  Pairs zone_stars(const FrameCat& fc, const Wcs& wcs, Vec2 pc, double r) const;
  size_t inliers_in(const FrameCat& fc, const Wcs& wcs, std::array<double, 4> rect, double tol) const;
  std::optional<Grown> grow_stereographic(const Wcs& seed);
  bool extension_evidence(const Wcs& wcs, std::array<double, 4> seed);
  std::optional<Grown> grow(const Wcs& w0, std::array<double, 4> win, double rms0, FrameCat& fc, const geometry::Projection& start);
  double zone_angle(const Wcs& wcs, Vec2 pc, double r) const;
};

// Shared helpers (solve.cpp)
std::unique_ptr<Solver> make_context(const Index& idx, const StarField& field, const SolveConfig& cfg, Clock::time_point t0);

// wide.rs
namespace wide {
inline constexpr double LANDSCAPE_WHOLE_SHARE = 0.25;
constexpr size_t MIN_WINDOW_STARS = 40;
constexpr double MIN_WINDOW_MS = 40.0;
std::optional<Solution> solve_windows(const Index& idx, const StarField& field, const SolveConfig& cfg, Clock::time_point t0,
                                      const SolveStats& spent);
std::optional<Solution> solve_windows_within(const Index& idx, const StarField& field, const SolveConfig& cfg, Clock::time_point t0,
                                             const SolveStats& spent, bool uncapped);
void accumulate(SolveStats& total, const SolveStats& s);
bool mapping_holds(const Solver& ctx, const Wcs& wcs);
bool grown_holds(const Solver& ctx, const Wcs& wcs, const Pairs& pairs, double rms);
std::optional<Solution> refit_whole(Solver& ctx, const Solution& f);
std::optional<Solution> extend(const Index& idx, const StarField& field, const SolveConfig& cfg, const Solution& ws,
                               std::array<double, 4> win);
std::optional<Solution> finish(Solver& ctx, const StarField& field, const SolveConfig& cfg, const Solution& ws, std::array<double, 4> win,
                               const Grown& grown);
// panorama.rs: blind identification on tangent tiles of a cylinder
std::optional<Solution> panorama_search(const Index& idx, const Index* wide, const StarField& field, const SolveConfig& cfg,
                                        Clock::time_point t0, double end, SolveStats& stats, size_t& trials);
}  // namespace wide

namespace refract {
std::optional<Refraction> measure(const Wcs& wcs, const Pairs& pairs, double w, double h);
}

}  // namespace skyshot::solve
