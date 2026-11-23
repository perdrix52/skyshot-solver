// Index builder. Port of src/builder.rs.
#pragma once
#include <chrono>
#include <filesystem>

#include "d80.hpp"
#include "index.hpp"
#include "sphere.hpp"

namespace skyshot::builder {

struct BuildConfig {
  std::filesystem::path d80 = "work/astap";
  std::filesystem::path out = "work/index";
  double r7_arcmin = 1.94;
  uint32_t max_pool_level = 8;
  std::string profile = "full";
  bool dump_levels = true;
  size_t max_bucket = 4096;
  std::vector<float> caps = {18.0f, 16.5f, 15.0f, 13.0f, 11.0f};
  uint32_t caps_min_level = 1;
  std::optional<std::filesystem::path> coder;
};

struct PlanEntry {
  size_t ref;
  std::array<size_t, 3> sub;
};
const std::vector<PlanEntry>& record_plan(uint32_t m, bool dominant);

// Write the index files from level-8 survivors (exact unit vectors, dominance distances in radians,
// magnitudes; any order). `anchor_cone` restricts record anchors to a cap (regional builds).
index::Manifest write_index(const BuildConfig& cfg, std::chrono::steady_clock::time_point t0, const std::vector<Vec3>& exact,
                            const std::vector<double>& dist, const std::vector<float>& mags, uint64_t n_cat, const std::string& fingerprint,
                            std::optional<std::pair<Vec3, double>> anchor_cone);

index::Manifest build(const BuildConfig& cfg);
index::Manifest build_region(const BuildConfig& cfg, double ra, double dec, double radius_deg);

}  // namespace skyshot::builder
