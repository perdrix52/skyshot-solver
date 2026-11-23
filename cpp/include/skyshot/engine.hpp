// Owns a memory-mapped index and optional wide/deep companions across many solves. Port of src/engine.rs.
#pragma once
#include <filesystem>

#include "pipeline.hpp"

namespace skyshot {

struct SolveOutput {
  uint32_t schema_version = 1;
  solve::Solution solution;
  pipeline::PipelineTimings timings;
  size_t n_stars = 0;
  Json to_json() const;
};

class Engine {
 public:
  // Throws skyshot::Error.
  static Engine open(const std::filesystem::path& path);
  const index::Index& index() const { return *index_; }

  SolveOutput solve_path(const std::filesystem::path& path, const solve::SolveConfig& cfg) const;
  // Source positions are zero-based native pixels; brightness is expressed as flux.
  SolveOutput solve_sources(const StarField& field, const solve::SolveConfig& cfg) const;

 private:
  solve::SolveConfig config(const solve::SolveConfig& cfg) const;
  std::shared_ptr<const index::Index> index_;
  std::shared_ptr<const index::Index> deep_;
  std::shared_ptr<const index::Index> wide_;
};

}  // namespace skyshot
