// Image -> solution pipeline with the re-detection fallback. Port of src/pipeline.rs and src/attempt.rs.
#pragma once
#include <filesystem>

#include "extract.hpp"
#include "image_io.hpp"
#include "solve.hpp"

namespace skyshot::pipeline {

// Detection attempts in order with the share of the deadline each gets when another follows.
struct Attempt {
  const char* name;
  double share;
};
inline constexpr Attempt ATTEMPTS[3] = {{"default", 0.25}, {"local_background_compact", 0.25}, {"highpass_dog", 1.0}};
std::optional<double> attempt_share(const std::string& name);
std::optional<std::string> next_attempt(const std::string& name);

// The star list of detection attempt `name` (nullopt for an unknown name).
std::optional<StarField> attempt_field(const image_io::GrayImage& gray, const image_io::SourceInfo& info, const std::string& name);

struct PipelineTimings {
  double decode_ms = 0, extract_ms = 0;
  std::optional<double> mask_ms;
  double solve_ms = 0, total_ms = 0;
  uint32_t attempts = 0;
  std::optional<std::string> fallback;
  std::optional<double> queue_ms, network_ms;
  bool tracked = false;
  Json to_json() const;
};

struct PipelineResult {
  solve::Solution solution;
  PipelineTimings timings;
  size_t n_stars = 0;
  std::shared_ptr<const SkyMask> mask;
  std::optional<StarField> stars;
};

inline constexpr double FINAL_WINDOWS_MS = 700.0;

std::pair<solve::Solution, size_t> solve_attempts(const index::Index& idx, const std::vector<std::pair<std::string, const StarField*>>& attempts,
                                                  const solve::SolveConfig& cfg);
PipelineResult solve_gray(const index::Index& idx, const image_io::GrayImage& gray, const image_io::SourceInfo& info,
                          const solve::SolveConfig& cfg);

struct Prior {
  Wcs wcs;
  double search_px = 0;
  bool refine = false;
  double fallback_ms = 0;
  std::shared_ptr<const SkyMask> mask;
};
PipelineResult solve_gray_tracked(const index::Index& idx, const image_io::GrayImage& gray, const image_io::SourceInfo& info,
                                  const solve::SolveConfig& cfg, const Prior* prior);

PipelineResult solve_path(const index::Index& idx, const std::filesystem::path& path, const solve::SolveConfig& cfg);

// Approximate pointing supplied by capture software. Hints change only the order of the attempts,
// never the acceptance test.
struct Hint {
  double ra = 0, dec = 0, radius_deg = 0;
  std::optional<double> fov_deg;
};
inline constexpr double DEEP_FIRST_FOV_DEG = 0.3;

// Blind solve; if unsolved and a hint is given, build (or reuse from `cache`) a regional deep index from
// D80 around the hint and solve with it. With a field-size hint below DEEP_FIRST_FOV_DEG the order is reversed.
PipelineResult solve_path_hinted(const index::Index& idx, const std::filesystem::path& path, const solve::SolveConfig& cfg,
                                 const std::optional<Hint>& hint, const std::filesystem::path& d80, const std::filesystem::path& cache);

}  // namespace skyshot::pipeline
