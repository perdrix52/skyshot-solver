#include "skyshot/image_io.hpp"

#include <bit>
#include <cstdlib>
#include <fstream>
#include <map>

#include "skyshot/parallel.hpp"

namespace skyshot::image_io {

const char* source_format_name(SourceFormat f) {
  switch (f) {
    case SourceFormat::Fits: return "fits";
    case SourceFormat::Png: return "png";
    case SourceFormat::Jpeg: return "jpeg";
    case SourceFormat::Tiff: return "tiff";
    default: return "other";
  }
}

namespace {

// Channel weights for colour inputs: Rec.709 luma unless SA2_RGB_WEIGHTS=r,g,b overrides it.
const std::array<float, 3>& rgb_weights() {
  static const std::array<float, 3> W = [] {
    std::array<float, 3> dflt = {0.2126f, 0.7152f, 0.0722f};
    const char* e = std::getenv("SA2_RGB_WEIGHTS");
    if (!e) return dflt;
    std::vector<float> p;
    std::string s(e);
    size_t i = 0;
    while (i <= s.size()) {
      size_t j = s.find(',', i);
      if (j == std::string::npos) j = s.size();
      if (auto v = parse_f64(trim(std::string_view(s).substr(i, j - i)))) p.push_back(float(*v));
      i = j + 1;
    }
    float t = 0;
    for (float x : p) t += x;
    if (p.size() == 3 && t > 0.0f) return std::array<float, 3>{p[0] / t, p[1] / t, p[2] / t};
    return dflt;
  }();
  return W;
}

// ---------- luma conversion ----------
std::vector<float> luma_plane(const NativeImage& n, int32_t& bitpix, float& lsb) {
  const float B8 = 1.0f / 255.0f, B16 = 1.0f / 65535.0f;
  float scale = 1.0f;
  if (n.bits == 8) {
    bitpix = 8;
    lsb = B8;
    scale = B8;
  } else if (n.bits == 16) {
    bitpix = 16;
    lsb = B16;
    scale = B16;
  } else {
    bitpix = -32;
    lsb = 0.0f;
    scale = 1.0f;
  }
  const int C = n.channels;
  size_t count = size_t(n.width) * n.height;
  std::vector<float> out(count);
  const auto& w = rgb_weights();
  const float* s = n.samples.data();
  parallel_chunks(count, size_t(1) << 16, [&](size_t a, size_t b) {
    for (size_t i = a; i < b; ++i) {
      const float* p = s + i * size_t(C);
      float v;
      if (C == 1) {
        v = p[0] * scale;
      } else if (C == 2) {
        v = p[1] <= 0.0f ? std::numeric_limits<float>::quiet_NaN() : p[0] * scale;
      } else if (C == 4 && p[3] <= 0.0f) {
        v = std::numeric_limits<float>::quiet_NaN();
      } else {
        v = (w[0] * p[0] + w[1] * p[1] + w[2] * p[2]) * scale;
      }
      out[i] = v;
    }
  });
  return out;
}

void apply_orientation(std::vector<float>& px, uint32_t& w, uint32_t& h, int orientation) {
  if (orientation < 2 || orientation > 8) return;
  uint32_t nw = (orientation >= 5) ? h : w;
  uint32_t nh = (orientation >= 5) ? w : h;
  std::vector<float> out(px.size());
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      uint32_t dx, dy;
      switch (orientation) {
        case 2: dx = w - 1 - x; dy = y; break;
        case 3: dx = w - 1 - x; dy = h - 1 - y; break;
        case 4: dx = x; dy = h - 1 - y; break;
        case 5: dx = y; dy = x; break;
        case 6: dx = h - 1 - y; dy = x; break;
        case 7: dx = h - 1 - y; dy = w - 1 - x; break;
        default: dx = y; dy = w - 1 - x; break;  // 8
      }
      out[size_t(dy) * nw + dx] = px[size_t(y) * w + x];
    }
  }
  px.swap(out);
  w = nw;
  h = nh;
}

SourceFormat format_of(const uint8_t* m, size_t len) {
  auto starts = [&](const char* s, size_t n) { return len >= n && std::memcmp(m, s, n) == 0; };
  if (starts("\x89PNG", 4)) return SourceFormat::Png;
  if (starts("\xFF\xD8\xFF", 3)) return SourceFormat::Jpeg;
  if (starts("II*\0", 4) || starts("MM\0*", 4)) return SourceFormat::Tiff;
  return SourceFormat::Other;
}

Loaded decode_image(const uint8_t* bytes, size_t len, SourceFormat format) {
  NativeImage n;
  if (format == SourceFormat::Tiff) n = decode_tiff(bytes, len);
  else n = decode_png_jpeg(bytes, len);
  if (size_t(n.width) * size_t(n.height) > MAX_PIXELS) throw Error("Image exceeds 100 MP limit");
  Loaded r;
  r.info.format = format;
  r.image.pixels = luma_plane(n, r.info.bitpix, r.info.lsb);
  r.image.width = n.width;
  r.image.height = n.height;
  apply_orientation(r.image.pixels, r.image.width, r.image.height, n.orientation);
  return r;
}

// ---------- FITS ----------
struct FitsScaling {
  int64_t bitpix;
  double scale, zero;
  std::optional<int64_t> blank;
};

inline uint16_t be16(const uint8_t* p) { return uint16_t((p[0] << 8) | p[1]); }
inline uint32_t be32(const uint8_t* p) { return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | p[3]; }
inline uint64_t be64(const uint8_t* p) { return (uint64_t(be32(p)) << 32) | be32(p + 4); }

inline void put(float& o, float v, float weight, bool acc) {
  if (acc) o += v * weight;
  else o = v * weight;
}

void convert_fits_band(const uint8_t* src, float* out, size_t count, const FitsScaling& f, float weight, bool acc) {
  float s = float(f.scale), z = float(f.zero);
  switch (f.bitpix) {
    case 8: {
      bool has_blank = f.blank && *f.blank >= 0 && *f.blank <= 255;
      for (size_t i = 0; i < count; ++i) {
        uint8_t raw = src[i];
        float v = (has_blank && raw == uint8_t(*f.blank)) ? std::numeric_limits<float>::quiet_NaN() : float(raw) * s + z;
        put(out[i], v, weight, acc);
      }
      break;
    }
    case 16: {
      bool has_blank = f.blank && *f.blank >= -32768 && *f.blank <= 32767;
      for (size_t i = 0; i < count; ++i) {
        int16_t raw = int16_t(be16(src + 2 * i));
        float v = (has_blank && raw == int16_t(*f.blank)) ? std::numeric_limits<float>::quiet_NaN() : float(raw) * s + z;
        put(out[i], v, weight, acc);
      }
      break;
    }
    case 32: {
      bool has_blank = f.blank && *f.blank >= INT32_MIN && *f.blank <= INT32_MAX;
      for (size_t i = 0; i < count; ++i) {
        int32_t raw = int32_t(be32(src + 4 * i));
        float v = (has_blank && raw == int32_t(*f.blank)) ? std::numeric_limits<float>::quiet_NaN()
                                                          : float(double(raw) * f.scale + f.zero);
        put(out[i], v, weight, acc);
      }
      break;
    }
    case 64: {
      for (size_t i = 0; i < count; ++i) {
        int64_t raw = int64_t(be64(src + 8 * i));
        float v = (f.blank && raw == *f.blank) ? std::numeric_limits<float>::quiet_NaN()
                                               : float(double(raw) * f.scale + f.zero);
        put(out[i], v, weight, acc);
      }
      break;
    }
    case -32: {
      bool plain = s == 1.0f && z == 0.0f && weight == 1.0f && !acc;
      for (size_t i = 0; i < count; ++i) {
        float v = std::bit_cast<float>(be32(src + 4 * i));
        if (plain) out[i] = v;
        else put(out[i], v * s + z, weight, acc);
      }
      break;
    }
    case -64: {
      for (size_t i = 0; i < count; ++i) {
        double d = std::bit_cast<double>(be64(src + 8 * i));
        put(out[i], float(d * f.scale + f.zero), weight, acc);
      }
      break;
    }
  }
}

std::optional<int64_t> parse_i64(const std::string& s) {
  if (s.empty()) return std::nullopt;
  size_t i = 0;
  bool neg = false;
  if (s[0] == '+' || s[0] == '-') {
    neg = s[0] == '-';
    i = 1;
  }
  if (i >= s.size()) return std::nullopt;
  uint64_t v = 0;
  const uint64_t limit = neg ? (uint64_t(1) << 63) : uint64_t(INT64_MAX);
  for (; i < s.size(); ++i) {
    if (s[i] < '0' || s[i] > '9') return std::nullopt;
    uint64_t d = uint64_t(s[i] - '0');
    if (v > (limit - d) / 10) return std::nullopt;
    v = v * 10 + d;
  }
  if (neg) return v == (uint64_t(1) << 63) ? INT64_MIN : -int64_t(v);
  return int64_t(v);
}

std::string strip_quotes(std::string s) {
  size_t a = 0, b = s.size();
  while (a < b && s[a] == '\'') ++a;
  while (b > a && s[b - 1] == '\'') --b;
  return s.substr(a, b - a);
}

Loaded load_fits(const uint8_t* file, size_t len) {
  uint64_t pos = 0;
  std::optional<FitsHeader> primary;
  for (;;) {
    FitsHeader header;
    bool ended = false;
    int blocks = 0;
    while (!ended) {
      if (pos + 2880 > len) throw Error("Truncated FITS header");
      const uint8_t* block = file + pos;
      pos += 2880;
      ++blocks;
      if (blocks >= 4096) throw Error("FITS header too long");
      for (int c = 0; c < 36; ++c) {
        const uint8_t* card = block + c * 80;
        std::string key = trim(std::string(reinterpret_cast<const char*>(card), 8));
        if (key == "END") {
          ended = true;
          break;
        }
        if (card[8] == '=') {
          std::string v(reinterpret_cast<const char*>(card + 10), 70);
          size_t slash = v.find('/');
          if (slash != std::string::npos) v.resize(slash);
          v = trim(strip_quotes(trim(v)));
          header[key] = v;
        }
      }
    }
    if (!primary) primary = header;
    auto number = [&](const std::string& name, int64_t dflt) -> int64_t {
      auto it = header.find(name);
      if (it == header.end()) return dflt;
      auto v = parse_i64(it->second);
      if (!v) throw Error("Invalid FITS " + name);
      return *v;
    };
    auto fl = [&](const std::string& name, double dflt) -> double {
      auto it = header.find(name);
      if (it == header.end()) return dflt;
      std::string v = it->second;
      for (auto& ch : v)
        if (ch == 'D') ch = 'E';
      auto d = parse_f64(v);
      if (!d) throw Error("Invalid FITS " + name);
      return *d;
    };
    int64_t bitpix = number("BITPIX", 8);
    int64_t naxis = number("NAXIS", 0);
    if (!(naxis >= 0 && naxis <= 9)) throw Error("Invalid FITS NAXIS");
    uint64_t count = naxis == 0 ? 0 : 1;
    std::vector<size_t> dims;
    for (int64_t i = 1; i <= naxis; ++i) {
      int64_t d = number("NAXIS" + std::to_string(i), 0);
      if (d < 0) throw Error("Negative FITS dimension");
      if (d != 0 && count > UINT64_MAX / uint64_t(d)) throw Error("FITS dimensions overflow");
      count *= uint64_t(d);
      dims.push_back(size_t(d));
    }
    if (!(bitpix == 8 || bitpix == 16 || bitpix == 32 || bitpix == 64 || bitpix == -32 || bitpix == -64))
      throw Error("Unsupported FITS BITPIX");
    size_t bpp = size_t(bitpix < 0 ? -bitpix : bitpix) / 8;
    int64_t pcount = number("PCOUNT", 0), gcount = number("GCOUNT", 1);
    if (!(pcount >= 0 && gcount >= 1)) throw Error("Invalid FITS group size");
    if (count > UINT64_MAX - uint64_t(pcount)) throw Error("FITS size overflow");
    uint64_t data_bytes = count + uint64_t(pcount);
    if (data_bytes != 0 && data_bytes > UINT64_MAX / uint64_t(gcount)) throw Error("FITS size overflow");
    data_bytes *= uint64_t(gcount);
    if (data_bytes != 0 && data_bytes > UINT64_MAX / bpp) throw Error("FITS size overflow");
    data_bytes *= bpp;
    if (pos + data_bytes < pos || pos + data_bytes > len) throw Error("Truncated FITS data");
    auto xt = header.find("XTENSION");
    bool image_hdu = xt == header.end() || xt->second == "IMAGE";
    if (image_hdu && naxis >= 2) {
      if (!(naxis == 2 || (naxis == 3 && dims[2] == 3))) throw Error("FITS cubes require a 2D image or three RGB planes");
      size_t width = dims[0], height = dims[1];
      if (!(width > 0 && height > 0 && width <= MAX_PIXELS / height))
        throw Error("Invalid image dimensions or image exceeds 100 MP");
      double scale = fl("BSCALE", 1.0), zero = fl("BZERO", 0.0);
      if (!(std::isfinite(scale) && std::isfinite(zero))) throw Error("Invalid FITS scaling");
      std::optional<int64_t> blank;
      if (auto it = header.find("BLANK"); it != header.end()) blank = parse_i64(it->second);
      FitsScaling sc{bitpix, scale, zero, bitpix > 0 ? blank : std::nullopt};
      size_t planes = naxis == 3 ? 3 : 1;
      size_t size = width * height;
      Loaded out;
      out.image.pixels.assign(size, 0.0f);
      size_t rows = std::max<size_t>((size_t(1) << 20) / std::max<size_t>(width * bpp, 1), 1);
      size_t nbands = (height + rows - 1) / rows;
      const auto& W = rgb_weights();
      uint64_t base = pos;
      float* px = out.image.pixels.data();
      parallel_for(nbands, [&](size_t band) {
        size_t y0 = band * rows, y1 = std::min(height, y0 + rows);
        for (size_t ch = 0; ch < planes; ++ch) {
          const uint8_t* src = file + base + (ch * size + y0 * width) * bpp;
          float weight = planes == 1 ? 1.0f : W[ch];
          convert_fits_band(src, px + y0 * width, (y1 - y0) * width, sc, weight, ch > 0);
        }
      });
      FitsHeader metadata = *primary;
      for (auto& kv : header) metadata[kv.first] = kv.second;
      std::optional<double> epoch;
      if (auto it = metadata.find("MJD-OBS"); it != metadata.end()) {
        if (auto mjd = parse_f64(it->second)) epoch = 2000.0 + (*mjd - 51544.5) / 365.25;
      }
      if (!epoch) {
        if (auto it = metadata.find("DATE-OBS"); it != metadata.end() && it->second.size() >= 4) {
          if (auto y = parse_f64(it->second.substr(0, 4))) epoch = *y + 0.5;
        }
      }
      if (epoch && !std::isfinite(*epoch)) epoch.reset();
      out.image.width = uint32_t(width);
      out.image.height = uint32_t(height);
      out.image.epoch = epoch;
      out.info.format = SourceFormat::Fits;
      out.info.bitpix = int32_t(bitpix);
      out.info.lsb = bitpix > 0 ? float(std::fabs(scale)) : 0.0f;
      out.info.hint = hint_from_fits(metadata, uint32_t(width), uint32_t(height));
      return out;
    }
    uint64_t next = pos + (data_bytes + 2879) / 2880 * 2880;
    if (next >= len) throw Error("No supported image HDU in FITS");
    pos = next;
  }
}

}  // namespace

std::optional<std::string> fits_row_order(const uint8_t* src, size_t len) {
  if (len < 8 || std::memcmp(src, "SIMPLE  ", 8) != 0) return std::nullopt;
  size_t blocks = std::min<size_t>(len / 2880, 64);
  for (size_t b = 0; b < blocks; ++b) {
    for (int c = 0; c < 36; ++c) {
      const uint8_t* card = src + b * 2880 + size_t(c) * 80;
      if (std::memcmp(card, "END     ", 8) == 0) return std::nullopt;
      if (std::memcmp(card, "ROWORDER", 8) == 0 && card[8] == '=') {
        std::string v(reinterpret_cast<const char*>(card + 10), 70);
        size_t slash = v.find('/');
        if (slash != std::string::npos) v.resize(slash);
        v = trim(strip_quotes(trim(v)));
        for (auto& ch : v) ch = char(std::toupper(static_cast<unsigned char>(ch)));
        return v;
      }
    }
  }
  return std::nullopt;
}

Loaded decode_bytes(const uint8_t* bytes, size_t len) {
  if (len >= 8 && std::memcmp(bytes, "SIMPLE  ", 8) == 0) return load_fits(bytes, len);
  SourceFormat f = format_of(bytes, len);
  if (f == SourceFormat::Other) throw Error("The image format could not be determined");
  return decode_image(bytes, len, f);
}

Loaded load_with_info(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("open " + path.string());
  f.seekg(0, std::ios::end);
  auto n = static_cast<std::streamoff>(f.tellg());
  f.seekg(0);
#ifdef _WIN32
  if (n > std::streamoff(300) * 1024 * 1024) throw Error("File exceeds desktop upload limit");
#endif
  std::vector<uint8_t> bytes(static_cast<size_t>(n));
  if (n > 0) f.read(reinterpret_cast<char*>(bytes.data()), n);
  if (!f && n > 0) throw Error("read " + path.string());
  return decode_bytes(bytes.data(), bytes.size());
}

GrayImage load(const std::filesystem::path& path) { return load_with_info(path).image; }

}  // namespace skyshot::image_io
