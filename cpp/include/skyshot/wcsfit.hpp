// Robust TAN + SIP fitting from pixel <-> sky correspondences. Port of src/wcsfit.rs.
#pragma once
#include "geometry.hpp"
#include "types.hpp"

namespace skyshot::wcsfit {

using Pair = std::array<double, 4>;  // x, y, ra, dec
using Pairs = std::vector<Pair>;
using geometry::Projection;

struct ModelChoice {
  Wcs wcs;
  std::vector<bool> keep;
  double err;
  std::pair<uint32_t, uint32_t> model;  // (SIP order, radial degree)
};

struct RobustFit {
  Wcs wcs;
  std::vector<bool> keep;
};

size_t model_columns(uint32_t order, uint32_t radial);

std::optional<Wcs> fit_tan_sip(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent, uint32_t order);
std::optional<Wcs> fit_tan_sip_model(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                     uint32_t order, uint32_t radial);
std::optional<Wcs> fit_model(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                             uint32_t order, uint32_t radial, const Projection& proj);

std::vector<double> residuals(const Wcs& wcs, const Pairs& pairs);
double rms(const std::vector<double>& r);
double robust_test_rms(const std::vector<double>& r, const std::vector<double>& w);

std::optional<RobustFit> robust_fit(const Pairs& pairs, Vec2 crpix, Vec2 tangent, uint32_t order);
std::optional<RobustFit> robust_fit_w(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent, uint32_t order);
std::optional<RobustFit> robust_fit_wm(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                       uint32_t order, uint32_t radial);
std::optional<RobustFit> robust_fit_wmp(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                        uint32_t order, uint32_t radial, const Projection& proj);

struct OrderChoice {
  Wcs wcs;
  std::vector<bool> keep;
  double err;
};
std::optional<OrderChoice> select_order(const Pairs& pairs, Vec2 crpix, Vec2 tangent, uint32_t max_order);
std::optional<OrderChoice> select_order_w(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                          uint32_t max_order);
std::optional<ModelChoice> select_model_w(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                          uint32_t max_order, uint32_t radial);
std::optional<ModelChoice> select_model_wp(const Pairs& pairs, const std::vector<double>& weights, Vec2 crpix, Vec2 tangent,
                                           uint32_t max_order, uint32_t radial, const Projection& proj);

// wcsfit_panorama.cpp
std::optional<Wcs> cylinder_seed(const Pairs& pairs, Vec2 crpix, Vec2 tangent, Vec2 pole);
std::vector<Projection> panorama_candidates(const Pairs& pairs, Vec2 crpix, Vec2 tangent);
// wcsfit_stereographic.cpp
std::optional<RobustFit> fit_stereographic(const Pairs& pairs, const std::vector<double>& weights);
// wcsfit_lens.cpp
std::optional<Wcs> select_lens(const Pairs& pairs, const std::vector<double>& weights, const Wcs& base,
                               std::pair<uint32_t, uint32_t> model, double width, double height);

}  // namespace skyshot::wcsfit
