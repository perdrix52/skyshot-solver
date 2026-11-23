// Baseline star detector (comparison baseline of the extractor). Port of src/detect.rs.
#pragma once
#include <atomic>
#include <chrono>

#include "image_io.hpp"
#include "types.hpp"

namespace skyshot::detect {

using Deadline = std::chrono::steady_clock::time_point;
StarField detect(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline);
// Alternate measurement for undersampled stars and structured nebular backgrounds.
StarField detect_compact(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline);
// Measure sources against a surrounding annulus on structured backgrounds.
StarField detect_local(const image_io::GrayImage& image, const std::atomic<bool>& cancel, Deadline deadline);

}  // namespace skyshot::detect
