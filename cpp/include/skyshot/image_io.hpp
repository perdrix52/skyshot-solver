// Image decoding to a luminance plane. Port of src/image_io.rs (FITS, PNG, JPEG, TIFF).
#pragma once
#include <filesystem>

#include "common.hpp"
#include "hints.hpp"
#include "json.hpp"

namespace skyshot::image_io {

inline constexpr size_t MAX_PIXELS = 100000000;

inline constexpr uint32_t PREVIEW_MAX_SIDE = 1600;

struct GrayImage {
  uint32_t width = 0, height = 0;
  std::vector<float> pixels;
  std::optional<double> epoch;

  // Size of preview(): the frame scaled down (never up) so that its longer side is at most PREVIEW_MAX_SIDE.
  std::pair<uint32_t, uint32_t> preview_size() const;
  // 8-bit greyscale PNG preview (area-average downscale plus an automatic asinh stretch). Throws on a frame without finite pixels.
  std::vector<uint8_t> preview() const;
  std::vector<float> box_downsample(uint32_t w, uint32_t h) const;
};

// Minimal PNG encoder (stored deflate blocks): 8-bit samples, `channels` 1..4.
std::vector<uint8_t> encode_png(uint32_t width, uint32_t height, int channels, const uint8_t* data);

enum class SourceFormat { Fits, Png, Jpeg, Tiff, Other };
const char* source_format_name(SourceFormat f);

struct SourceInfo {
  SourceFormat format = SourceFormat::Other;
  int32_t bitpix = 8;
  float lsb = 0.0f;
  std::optional<SolveHint> hint;
};

struct Loaded {
  GrayImage image;
  SourceInfo info;
};

// Throw skyshot::Error on failure.
Loaded load_with_info(const std::filesystem::path& path);
GrayImage load(const std::filesystem::path& path);
Loaded decode_bytes(const uint8_t* bytes, size_t len);
std::optional<std::string> fits_row_order(const uint8_t* src, size_t len);

// Decoders for the non-FITS formats (implemented in image_formats.cpp).
struct NativeImage {
  uint32_t width = 0, height = 0;
  int channels = 1;       // 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA
  int bits = 8;           // 8, 16 or -32 (float samples)
  std::vector<float> samples;  // interleaved, native sample values (not normalised)
  int orientation = 1;    // EXIF orientation 1..8
};
NativeImage decode_png_jpeg(const uint8_t* bytes, size_t len);
NativeImage decode_tiff(const uint8_t* bytes, size_t len);
int exif_orientation_from_jpeg(const uint8_t* bytes, size_t len);
int exif_orientation_from_png(const uint8_t* bytes, size_t len);

}  // namespace skyshot::image_io
