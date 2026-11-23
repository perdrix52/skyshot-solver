// Image-side dominance and rung ladder. Port of src/select.rs.
#pragma once
#include "common.hpp"

namespace skyshot::select {

using P2 = Vec2;

class Grid2 {
 public:
  double cell = 1.0;
  size_t nx = 1, ny = 1;
  std::vector<uint32_t> offsets, items;

  static Grid2 build(const std::vector<P2>& pts, const std::vector<uint32_t>& ids, double width, double height,
                     double per_cell);
  std::optional<std::pair<double, uint32_t>> nearest_within(const std::vector<P2>& pts, P2 q, double radius,
                                                            uint32_t skip) const;
  void knn(const std::vector<P2>& pts, P2 q, size_t k, uint32_t skip, std::vector<std::pair<double, uint32_t>>& out) const;

 private:
  double cell_axis_distance(double q, size_t c, size_t count) const;
  double outside_distance(P2 q, int64_t cx, int64_t cy, int64_t ring) const;
};

std::vector<double> dominance_px(const std::vector<P2>& pts, double width, double height);
std::vector<double> edge_px(const std::vector<P2>& pts, double width, double height);

inline constexpr double RUNG_STEP_LOG2 = 0.25;
inline double rung_radius(int k) { return std::exp2(double(k) * RUNG_STEP_LOG2); }
std::vector<int> rung_order(const std::vector<double>& d, size_t target, size_t min_pool, size_t max_pool);

}  // namespace skyshot::select
