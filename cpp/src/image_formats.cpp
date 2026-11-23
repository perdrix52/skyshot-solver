// PNG / JPEG (via stb_image), TIFF (own baseline reader) and EXIF orientation.
#include <bit>
#include <cstdlib>
#include <cstring>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_FAILURE_STRINGS
#include <stb_image.h>

#include "skyshot/image_io.hpp"

namespace skyshot::image_io {

namespace {

// ---------- TIFF-structured EXIF / IFD helpers ----------
struct TiffReader {
  const uint8_t* d;
  size_t n;
  bool le;
  uint16_t u16(size_t o) const {
    if (o + 2 > n) throw Error("Truncated TIFF");
    return le ? uint16_t(d[o] | (d[o + 1] << 8)) : uint16_t((d[o] << 8) | d[o + 1]);
  }
  uint32_t u32(size_t o) const {
    if (o + 4 > n) throw Error("Truncated TIFF");
    return le ? (uint32_t(d[o]) | (uint32_t(d[o + 1]) << 8) | (uint32_t(d[o + 2]) << 16) | (uint32_t(d[o + 3]) << 24))
              : ((uint32_t(d[o]) << 24) | (uint32_t(d[o + 1]) << 16) | (uint32_t(d[o + 2]) << 8) | d[o + 3]);
  }
};

bool tiff_header(const uint8_t* d, size_t n, TiffReader& r, size_t& ifd0) {
  if (n < 8) return false;
  if (d[0] == 'I' && d[1] == 'I') r.le = true;
  else if (d[0] == 'M' && d[1] == 'M') r.le = false;
  else return false;
  r.d = d;
  r.n = n;
  if (r.u16(2) != 42) return false;
  ifd0 = r.u32(4);
  return true;
}

struct Entry {
  uint16_t tag, type;
  uint32_t count;
  size_t value_off;  // offset of the value bytes (inline or pointed)
};

size_t type_size(uint16_t t) {
  switch (t) {
    case 1: case 2: case 6: case 7: return 1;
    case 3: case 8: return 2;
    case 4: case 9: case 11: return 4;
    case 5: case 10: case 12: return 8;
    case 16: case 17: case 18: return 8;
    default: return 0;
  }
}

std::vector<Entry> read_ifd(const TiffReader& r, size_t off) {
  std::vector<Entry> out;
  uint16_t cnt = r.u16(off);
  for (uint16_t i = 0; i < cnt; ++i) {
    size_t e = off + 2 + size_t(i) * 12;
    Entry en;
    en.tag = r.u16(e);
    en.type = r.u16(e + 2);
    en.count = r.u32(e + 4);
    size_t total = type_size(en.type) * size_t(en.count);
    en.value_off = total <= 4 ? e + 8 : size_t(r.u32(e + 8));
    out.push_back(en);
  }
  return out;
}

const Entry* find(const std::vector<Entry>& v, uint16_t tag) {
  for (auto& e : v)
    if (e.tag == tag) return &e;
  return nullptr;
}

uint32_t entry_u32(const TiffReader& r, const Entry& e, size_t idx = 0) {
  if (idx >= e.count) throw Error("TIFF tag index out of range");
  switch (e.type) {
    case 1: case 7: return r.d[e.value_off + idx];
    case 3: return r.u16(e.value_off + idx * 2);
    case 4: return r.u32(e.value_off + idx * 4);
    default: throw Error("Unsupported TIFF tag type");
  }
}

int orientation_from_ifd(const uint8_t* d, size_t n) {
  try {
    TiffReader r{};
    size_t ifd0;
    if (!tiff_header(d, n, r, ifd0)) return 1;
    auto entries = read_ifd(r, ifd0);
    if (const Entry* e = find(entries, 274)) {
      uint32_t v = entry_u32(r, *e);
      if (v >= 1 && v <= 8) return int(v);
    }
  } catch (const Error&) {
  }
  return 1;
}

// ---------- LZW (TIFF flavour: MSB first, early change) ----------
std::vector<uint8_t> lzw_decode(const uint8_t* src, size_t n, size_t expected) {
  std::vector<uint8_t> out;
  out.reserve(expected);
  struct Ent {
    int prefix;
    uint8_t ch;
    uint32_t len;
  };
  std::vector<Ent> table(4096);
  auto reset = [&] {
    for (int i = 0; i < 256; ++i) table[size_t(i)] = {-1, uint8_t(i), 1};
  };
  reset();
  int next = 258, bits = 9;
  uint32_t buf = 0;
  int nbuf = 0;
  size_t pos = 0;
  int prev = -1;
  std::vector<uint8_t> tmp;
  auto emit = [&](int code) {
    uint32_t len = table[size_t(code)].len;
    size_t base = out.size();
    out.resize(base + len);
    int c = code;
    for (uint32_t i = len; i-- > 0;) {
      out[base + i] = table[size_t(c)].ch;
      c = table[size_t(c)].prefix;
    }
  };
  auto first_char = [&](int code) {
    int c = code;
    while (table[size_t(c)].prefix >= 0) c = table[size_t(c)].prefix;
    return table[size_t(c)].ch;
  };
  while (out.size() < expected) {
    while (nbuf < bits && pos < n) {
      buf = (buf << 8) | src[pos++];
      nbuf += 8;
    }
    if (nbuf < bits) break;
    int code = int((buf >> (nbuf - bits)) & ((1u << bits) - 1));
    nbuf -= bits;
    if (code == 257) break;
    if (code == 256) {
      reset();
      next = 258;
      bits = 9;
      prev = -1;
      continue;
    }
    if (prev < 0) {
      if (code >= 256) throw Error("Corrupt TIFF LZW data");
      emit(code);
      prev = code;
      continue;
    }
    if (code < next) {
      emit(code);
      if (next < 4096) {
        table[size_t(next)] = {prev, first_char(code), table[size_t(prev)].len + 1};
        ++next;
      }
    } else if (code == next && next < 4096) {
      table[size_t(next)] = {prev, first_char(prev), table[size_t(prev)].len + 1};
      ++next;
      emit(code);
    } else {
      throw Error("Corrupt TIFF LZW data");
    }
    prev = code;
    if (next + 1 >= 4096) bits = 12;
    else if (next + 1 >= 2048) bits = 12;
    else if (next + 1 >= 1024) bits = 11;
    else if (next + 1 >= 512) bits = 10;
    else bits = 9;
  }
  return out;
}

std::vector<uint8_t> packbits_decode(const uint8_t* s, size_t n, size_t expected) {
  std::vector<uint8_t> out;
  out.reserve(expected);
  size_t i = 0;
  while (i < n && out.size() < expected) {
    int8_t c = int8_t(s[i++]);
    if (c >= 0) {
      size_t cnt = size_t(c) + 1;
      if (i + cnt > n) throw Error("Corrupt TIFF PackBits data");
      out.insert(out.end(), s + i, s + i + cnt);
      i += cnt;
    } else if (c != -128) {
      size_t cnt = size_t(1 - c);
      if (i >= n) throw Error("Corrupt TIFF PackBits data");
      out.insert(out.end(), cnt, s[i++]);
    }
  }
  return out;
}

}  // namespace

// ---------- EXIF orientation ----------
int exif_orientation_from_jpeg(const uint8_t* b, size_t n) {
  size_t i = 2;
  while (i + 4 <= n) {
    if (b[i] != 0xFF) return 1;
    uint8_t marker = b[i + 1];
    if (marker == 0xFF) {
      ++i;
      continue;
    }
    if (marker == 0xD8 || (marker >= 0xD0 && marker <= 0xD7) || marker == 0x01) {
      i += 2;
      continue;
    }
    if (marker == 0xDA || marker == 0xD9) return 1;
    size_t len = (size_t(b[i + 2]) << 8) | b[i + 3];
    if (len < 2 || i + 2 + len > n) return 1;
    if (marker == 0xE1 && len >= 8 && std::memcmp(b + i + 4, "Exif\0\0", 6) == 0)
      return orientation_from_ifd(b + i + 10, len - 8);
    i += 2 + len;
  }
  return 1;
}

int exif_orientation_from_png(const uint8_t* b, size_t n) {
  size_t i = 8;
  while (i + 12 <= n) {
    uint32_t len = (uint32_t(b[i]) << 24) | (uint32_t(b[i + 1]) << 16) | (uint32_t(b[i + 2]) << 8) | b[i + 3];
    if (i + 12 + size_t(len) > n) return 1;
    if (std::memcmp(b + i + 4, "eXIf", 4) == 0) return orientation_from_ifd(b + i + 8, len);
    if (std::memcmp(b + i + 4, "IDAT", 4) == 0) return 1;
    i += 12 + size_t(len);
  }
  return 1;
}

// ---------- PNG / JPEG ----------
NativeImage decode_png_jpeg(const uint8_t* bytes, size_t len) {
  if (len > size_t(INT32_MAX)) throw Error("Image too large");
  bool png = len >= 4 && std::memcmp(bytes, "\x89PNG", 4) == 0;
  NativeImage out;
  int w = 0, h = 0, comp = 0;
  bool is16 = false;
  if (png) {
    if (len >= 26) is16 = bytes[24] == 16;
    out.orientation = exif_orientation_from_png(bytes, len);
  } else {
    out.orientation = exif_orientation_from_jpeg(bytes, len);
  }
  if (!stbi_info_from_memory(bytes, int(len), &w, &h, &comp)) throw Error("The image could not be decoded");
  if (w > 50000 || h > 50000) throw Error("Image dimensions exceed the limit");
  if (size_t(w) * size_t(h) > MAX_PIXELS) throw Error("Image exceeds 100 MP limit");
  if (is16) {
    stbi_us* p = stbi_load_16_from_memory(bytes, int(len), &w, &h, &comp, 0);
    if (!p) throw Error("The image could not be decoded");
    out.bits = 16;
    out.samples.resize(size_t(w) * h * comp);
    for (size_t i = 0; i < out.samples.size(); ++i) out.samples[i] = float(p[i]);
    stbi_image_free(p);
  } else {
    stbi_uc* p = stbi_load_from_memory(bytes, int(len), &w, &h, &comp, 0);
    if (!p) throw Error("The image could not be decoded");
    out.bits = 8;
    out.samples.resize(size_t(w) * h * comp);
    for (size_t i = 0; i < out.samples.size(); ++i) out.samples[i] = float(p[i]);
    stbi_image_free(p);
  }
  out.width = uint32_t(w);
  out.height = uint32_t(h);
  out.channels = comp;
  return out;
}

// ---------- TIFF ----------
NativeImage decode_tiff(const uint8_t* d, size_t n) {
  TiffReader r{};
  size_t ifd0;
  if (!tiff_header(d, n, r, ifd0)) throw Error("Invalid TIFF header");
  auto E = read_ifd(r, ifd0);
  auto need = [&](uint16_t tag) -> const Entry& {
    const Entry* e = find(E, tag);
    if (!e) throw Error("TIFF is missing tag " + std::to_string(tag));
    return *e;
  };
  auto opt = [&](uint16_t tag, uint32_t dflt) -> uint32_t {
    const Entry* e = find(E, tag);
    return e ? entry_u32(r, *e) : dflt;
  };
  uint32_t width = entry_u32(r, need(256)), height = entry_u32(r, need(257));
  if (width == 0 || height == 0) throw Error("Invalid TIFF dimensions");
  if (width > 50000 || height > 50000 || uint64_t(width) * height > MAX_PIXELS) throw Error("Image exceeds 100 MP limit");
  uint32_t spp = opt(277, 1);
  uint32_t bps = opt(258, 1);
  if (const Entry* e = find(E, 258)) {
    bps = entry_u32(r, *e, 0);
    for (uint32_t i = 1; i < e->count; ++i)
      if (entry_u32(r, *e, i) != bps) throw Error("TIFF with mixed sample sizes is unsupported");
  }
  uint32_t comp = opt(259, 1), photo = opt(262, 1), planar = opt(284, 1), pred = opt(317, 1), fmt = opt(339, 1);
  if (planar != 1) throw Error("Planar TIFF is unsupported");
  if (!(bps == 8 || bps == 16 || bps == 32)) throw Error("Unsupported TIFF bit depth");
  if (bps == 32 && fmt != 3) throw Error("Unsupported TIFF sample format");
  if (photo == 5 || photo == 6 || photo == 8) throw Error("Unsupported TIFF colour model");
  if (spp < 1 || spp > 4) throw Error("Unsupported TIFF sample count");
  if (pred != 1 && pred != 2) throw Error("Unsupported TIFF predictor");
  if (pred == 2 && bps == 32) throw Error("Unsupported TIFF predictor");
  size_t bytes_per_sample = bps / 8;
  size_t pixel_bytes = bytes_per_sample * spp;

  struct Chunk {
    size_t off, len;
    uint32_t w, h;  // pixel extent of the chunk data
    uint32_t x0, y0;
  };
  std::vector<Chunk> chunks;
  if (find(E, 322)) {
    uint32_t tw = entry_u32(r, need(322)), th = entry_u32(r, need(323));
    if (tw == 0 || th == 0) throw Error("Invalid TIFF tile size");
    const Entry& to = need(324);
    const Entry& tc = need(325);
    uint32_t across = (width + tw - 1) / tw, down = (height + th - 1) / th;
    if (to.count < uint64_t(across) * down) throw Error("Truncated TIFF tile table");
    for (uint32_t ty = 0; ty < down; ++ty)
      for (uint32_t tx = 0; tx < across; ++tx) {
        size_t i = size_t(ty) * across + tx;
        chunks.push_back({entry_u32(r, to, i), entry_u32(r, tc, i), tw, th, tx * tw, ty * th});
      }
  } else {
    uint32_t rps = opt(278, height);
    if (rps == 0 || rps > height) rps = height;
    const Entry& so = need(273);
    const Entry* sc = find(E, 279);
    uint32_t nstrips = (height + rps - 1) / rps;
    if (so.count < nstrips) throw Error("Truncated TIFF strip table");
    for (uint32_t s = 0; s < nstrips; ++s) {
      uint32_t rows = std::min(rps, height - s * rps);
      size_t off = entry_u32(r, so, s);
      size_t len = sc ? entry_u32(r, *sc, s) : size_t(rows) * width * pixel_bytes;
      chunks.push_back({off, len, width, rows, 0, s * rps});
    }
  }

  NativeImage out;
  out.width = width;
  out.height = height;
  bool palette = photo == 3;
  if (palette && (spp != 1 || bps != 8)) throw Error("Unsupported TIFF palette image");
  out.channels = palette ? 3 : int(spp);
  out.bits = bps == 32 ? -32 : int(bps);
  out.samples.assign(size_t(width) * height * size_t(out.channels), 0.0f);
  out.orientation = int(opt(274, 1));
  if (out.orientation < 1 || out.orientation > 8) out.orientation = 1;
  std::vector<uint16_t> cmap;
  if (palette) {
    const Entry& cm = need(320);
    if (cm.count < 768) throw Error("Invalid TIFF colour map");
    for (uint32_t i = 0; i < 768; ++i) cmap.push_back(uint16_t(entry_u32(r, cm, i)));
  }
  for (const Chunk& c : chunks) {
    if (c.off + c.len > n) throw Error("Truncated TIFF data");
    size_t expected = size_t(c.w) * c.h * pixel_bytes;
    std::vector<uint8_t> raw;
    const uint8_t* src = d + c.off;
    switch (comp) {
      case 1:
        raw.assign(src, src + std::min(c.len, expected));
        break;
      case 5:
        raw = lzw_decode(src, c.len, expected);
        break;
      case 32773:
        raw = packbits_decode(src, c.len, expected);
        break;
      case 8:
      case 32946: {
        int olen = 0;
        char* z = stbi_zlib_decode_malloc_guesssize_headerflag(reinterpret_cast<const char*>(src), int(c.len), int(expected), &olen, 1);
        if (!z) throw Error("Corrupt TIFF deflate data");
        raw.assign(reinterpret_cast<uint8_t*>(z), reinterpret_cast<uint8_t*>(z) + olen);
        free(z);
        break;
      }
      default:
        throw Error("Unsupported TIFF compression");
    }
    if (raw.size() < expected) throw Error("Truncated TIFF data");
    if (pred == 2) {
      for (uint32_t y = 0; y < c.h; ++y) {
        uint8_t* row = raw.data() + size_t(y) * c.w * pixel_bytes;
        if (bps == 8) {
          for (size_t i = spp; i < size_t(c.w) * spp; ++i) row[i] = uint8_t(row[i] + row[i - spp]);
        } else {
          for (size_t i = spp; i < size_t(c.w) * spp; ++i) {
            uint16_t a = r.le ? uint16_t(row[2 * i] | (row[2 * i + 1] << 8)) : uint16_t((row[2 * i] << 8) | row[2 * i + 1]);
            uint16_t b = r.le ? uint16_t(row[2 * (i - spp)] | (row[2 * (i - spp) + 1] << 8))
                              : uint16_t((row[2 * (i - spp)] << 8) | row[2 * (i - spp) + 1]);
            uint16_t v = uint16_t(a + b);
            if (r.le) {
              row[2 * i] = uint8_t(v);
              row[2 * i + 1] = uint8_t(v >> 8);
            } else {
              row[2 * i] = uint8_t(v >> 8);
              row[2 * i + 1] = uint8_t(v);
            }
          }
        }
      }
    }
    for (uint32_t y = 0; y < c.h; ++y) {
      uint32_t iy = c.y0 + y;
      if (iy >= height) break;
      for (uint32_t x = 0; x < c.w; ++x) {
        uint32_t ix = c.x0 + x;
        if (ix >= width) break;
        const uint8_t* px = raw.data() + (size_t(y) * c.w + x) * pixel_bytes;
        float* dst = out.samples.data() + (size_t(iy) * width + ix) * size_t(out.channels);
        if (palette) {
          uint8_t idx = px[0];
          for (int k = 0; k < 3; ++k) dst[k] = float(cmap[size_t(k) * 256 + idx] >> 8);
          continue;
        }
        for (uint32_t k = 0; k < spp; ++k) {
          const uint8_t* s = px + k * bytes_per_sample;
          float v;
          if (bps == 8) {
            v = float(s[0]);
          } else if (bps == 16) {
            v = float(r.le ? uint16_t(s[0] | (s[1] << 8)) : uint16_t((s[0] << 8) | s[1]));
          } else {
            v = std::bit_cast<float>(r.u32(size_t(s - d)));
          }
          dst[k] = v;
        }
      }
    }
  }
  if (photo == 0 && !palette && bps != 32) {  // WhiteIsZero
    float maxv = bps == 8 ? 255.0f : 65535.0f;
    int colour = spp == 2 ? 1 : int(spp);
    if (spp == 1 || spp == 2) {
      for (size_t i = 0; i < out.samples.size(); i += size_t(out.channels))
        for (int k = 0; k < colour && k < 1; ++k) out.samples[i + size_t(k)] = maxv - out.samples[i + size_t(k)];
    }
  }
  return out;
}

}  // namespace skyshot::image_io
