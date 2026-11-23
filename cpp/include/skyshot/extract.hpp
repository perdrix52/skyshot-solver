// Fast, robust star extractor. Port of src/extract.rs (with vet.rs and lines.rs).
#pragma once
#include "image_io.hpp"
#include "json.hpp"
#include "types.hpp"

namespace skyshot {

struct ExtractOptions {
  float threshold_sigma = 5.0f;
  bool compact = false;
  bool local_background = false;
  size_t max_per_tile = 64;
  size_t bin = 0;
  float lsb = 0.0f;
  bool keep_rejected = false;
  bool remove_lines = true;

  static ExtractOptions for_source(const image_io::SourceInfo& info) {
    ExtractOptions o;
    o.lsb = info.lsb;
    return o;
  }
};

StarField extract(const image_io::GrayImage& image, const ExtractOptions& opts);
size_t bin_factor(size_t w, size_t h, const ExtractOptions& opts);
float detect_lattice(const std::vector<float>& px);

}  // namespace skyshot
