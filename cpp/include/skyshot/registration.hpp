// Residual registration after blind identification. Port of src/wcsfit/registration.rs.
#pragma once
#include <functional>

#include "skymask.hpp"
#include "solve.hpp"

namespace skyshot::registration {

// Catalogue rows [base x, base y, ra, dec, magnitude]; image points in brightness order.
std::optional<PixelGrid> fit(const std::vector<std::array<double, 5>>& cat, const std::vector<Vec2>& image, double width,
                             double height, const std::function<bool(Vec2)>& visible, const SkyMask* mask, double budget_ms);

// Run once after the image pipeline has accepted a blind solution.
void refine_solution(const index::Index& idx, solve::Solution& solution, const StarField& field, const solve::SolveConfig& cfg,
                     const SkyMask* mask, double remaining_ms);

}  // namespace skyshot::registration
