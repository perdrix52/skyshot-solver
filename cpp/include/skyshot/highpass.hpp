// High-pass (difference-of-Gaussians) star detection. Port of src/highpass.rs.
#pragma once
#include "image_io.hpp"
#include "types.hpp"

namespace skyshot::highpass {

size_t box_radius(float sigma);
std::vector<float> box_blur3(const std::vector<float>& src, size_t w, size_t h, float sigma);
StarField detect_highpass(const image_io::GrayImage& img, float fwhm_px, float threshold);

}  // namespace skyshot::highpass
