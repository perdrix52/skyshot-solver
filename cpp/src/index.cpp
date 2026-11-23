#include "skyshot/index.hpp"

#include <algorithm>

#include "skyshot/sha256.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace skyshot::index {

namespace fs = std::filesystem;

// ---------- Record ----------
Record Record::make_deep(uint32_t fp, uint32_t anchor, uint32_t ar, uint32_t pa, bool flip, uint32_t level,
                         bool dominant, uint32_t dcode) {
  if (!(fp < (1u << 10) && anchor < (1u << 28) && ar < 1024 && pa < 1024)) throw Error("deep record field out of range");
  if (!(level >= 7 && level <= 8 && dcode < 8)) throw Error("deep record field out of range");
  return Record{uint64_t(fp) | uint64_t(anchor) << 10 | uint64_t(ar) << 38 | uint64_t(pa) << 48 | uint64_t(flip) << 58 |
                uint64_t(level - 7) << 59 | uint64_t(dominant) << 60 | uint64_t(dcode) << 61};
}
Record Record::make(uint32_t fp, uint32_t anchor, uint32_t ar, uint32_t pa, bool flip, uint32_t level, bool dominant,
                    uint32_t dcode) {
  return Record{uint64_t(fp) | uint64_t(anchor) << 12 | uint64_t(ar) << 36 | uint64_t(pa) << 46 | uint64_t(flip) << 56 |
                uint64_t(level - 1) << 57 | uint64_t(dominant) << 60 | uint64_t(dcode) << 61};
}

// ---------- Manifest JSON ----------
Json Manifest::to_json() const {
  Json o = Json::object();
  o.set("format", format).set("version", version).set("profile", profile).set("r7_arcmin", r7_arcmin);
  o.set("key_spec", key_spec);
  Json lv = Json::array();
  for (const auto& l : levels) {
    Json x = Json::object();
    x.set("m", l.m).set("radius_arcmin", l.radius_arcmin).set("stars", l.stars).set("anchors", l.anchors).set("records", l.records);
    lv.push(std::move(x));
  }
  o.set("levels", std::move(lv));
  o.set("n_records", n_records).set("n_stars", n_stars).set("n_catalog", n_catalog).set("star_cell_n", star_cell_n);
  o.set("catalog", catalog).set("catalog_fingerprint", catalog_fingerprint).set("build_seconds", build_seconds);
  Json fl = Json::array();
  for (const auto& f : files) {
    Json p = Json::array();
    p.push(Json(f.first)).push(Json(f.second));
    fl.push(std::move(p));
  }
  o.set("files", std::move(fl));
  o.set("stop_words", stop_words);
  if (key_type != "v1") o.set("key_type", key_type);
  if (coder_sha256) o.set("coder_sha256", *coder_sha256);
  return o;
}

namespace {
const Json& need(const Json& o, const char* k) {
  const Json* v = o.get(k);
  if (!v) throw Error(std::string("missing field `") + k + "`");
  return *v;
}
std::string need_str(const Json& o, const char* k) {
  const Json& v = need(o, k);
  if (!v.is_string()) throw Error(std::string("invalid type for field `") + k + "`, expected a string");
  return v.as_str();
}
double need_f64(const Json& o, const char* k) {
  const Json& v = need(o, k);
  if (!v.is_number()) throw Error(std::string("invalid type for field `") + k + "`, expected a number");
  return v.as_f64();
}
uint64_t need_u64(const Json& o, const char* k) {
  const Json& v = need(o, k);
  auto u = v.as_u64();
  if (!u) throw Error(std::string("invalid type for field `") + k + "`, expected an unsigned integer");
  return *u;
}
}  // namespace

Manifest Manifest::from_json(const Json& j) {
  if (!j.is_object()) throw Error("invalid type: expected a manifest object");
  Manifest m;
  m.format = need_str(j, "format");
  m.version = uint32_t(need_u64(j, "version"));
  m.profile = need_str(j, "profile");
  m.r7_arcmin = need_f64(j, "r7_arcmin");
  m.key_spec = need_str(j, "key_spec");
  const Json& lv = need(j, "levels");
  if (!lv.is_array()) throw Error("invalid type for field `levels`, expected a sequence");
  for (const auto& x : lv.items()) {
    LevelInfo l;
    l.m = uint32_t(need_u64(x, "m"));
    l.radius_arcmin = need_f64(x, "radius_arcmin");
    l.stars = need_u64(x, "stars");
    l.anchors = need_u64(x, "anchors");
    l.records = need_u64(x, "records");
    m.levels.push_back(l);
  }
  m.n_records = need_u64(j, "n_records");
  m.n_stars = need_u64(j, "n_stars");
  m.n_catalog = need_u64(j, "n_catalog");
  m.star_cell_n = uint32_t(need_u64(j, "star_cell_n"));
  m.catalog = need_str(j, "catalog");
  m.catalog_fingerprint = need_str(j, "catalog_fingerprint");
  m.build_seconds = need_f64(j, "build_seconds");
  const Json& fl = need(j, "files");
  if (!fl.is_array()) throw Error("invalid type for field `files`, expected a sequence");
  for (const auto& f : fl.items()) {
    if (!f.is_array() || f.items().size() != 2 || !f.items()[0].is_string() || !f.items()[1].as_u64())
      throw Error("invalid entry in `files`");
    m.files.emplace_back(f.items()[0].as_str(), *f.items()[1].as_u64());
  }
  if (const Json* s = j.get("stop_words")) {
    auto u = s->as_u64();
    if (!u) throw Error("invalid type for field `stop_words`");
    m.stop_words = *u;
  }
  if (const Json* k = j.get("key_type")) {
    if (!k->is_string()) throw Error("invalid type for field `key_type`");
    m.key_type = k->as_str();
  }
  if (const Json* c = j.get("coder_sha256"); c && !c->is_null()) {
    if (!c->is_string()) throw Error("invalid type for field `coder_sha256`");
    m.coder_sha256 = c->as_str();
  }
  return m;
}

// ---------- MappedFile ----------
MappedFile::~MappedFile() {
#ifdef _WIN32
  if (data_) UnmapViewOfFile(data_);
  if (os_a_) CloseHandle(static_cast<HANDLE>(os_a_));
#else
  if (data_) munmap(const_cast<uint8_t*>(data_), size_);
#endif
}

MappedFile MappedFile::open(const fs::path& path) {
  MappedFile m;
#ifdef _WIN32
  HANDLE f = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (f == INVALID_HANDLE_VALUE) throw Error("open " + path.string());
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(f, &sz)) {
    CloseHandle(f);
    throw Error("open " + path.string());
  }
  m.size_ = size_t(sz.QuadPart);
  if (m.size_ > 0) {
    HANDLE h = CreateFileMappingW(f, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(f);
    if (!h) throw Error("map " + path.string());
    void* p = MapViewOfFile(h, FILE_MAP_READ, 0, 0, 0);
    if (!p) {
      CloseHandle(h);
      throw Error("map " + path.string());
    }
    m.os_a_ = h;
    m.data_ = static_cast<const uint8_t*>(p);
  } else {
    CloseHandle(f);
  }
#else
  int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw Error("open " + path.string());
  struct stat st;
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    throw Error("open " + path.string());
  }
  m.size_ = size_t(st.st_size);
  if (m.size_ > 0) {
    void* p = mmap(nullptr, m.size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) throw Error("map " + path.string());
    m.data_ = static_cast<const uint8_t*>(p);
  } else {
    ::close(fd);
  }
#endif
  return m;
}

// ---------- companions ----------
std::optional<fs::path> companion_wide_dir(const fs::path& main) {
  fs::path parent = main.parent_path();
  fs::path fname = main.filename();
  if (parent.empty() || fname.empty()) return std::nullopt;
  std::string name = fname.string();
  bool v1 = name.size() >= 3 && name.compare(name.size() - 3, 3, "-v1") == 0;
  std::string wide = v1 ? "index-phone-v1" : "index-phone";
  fs::path dir = parent / wide;
  std::error_code ec;
  if (name != wide && fs::exists(dir / "manifest.json", ec)) return dir;
  return std::nullopt;
}
std::optional<fs::path> companion_deep_dir(const fs::path& main) {
  fs::path parent = main.parent_path();
  if (parent.empty()) return std::nullopt;
  fs::path dir = parent / "index-deep";
  std::error_code ec;
  if (dir != main && fs::exists(dir / "manifest.json", ec)) return dir;
  return std::nullopt;
}

// ---------- Index ----------
std::shared_ptr<Index> Index::open(const fs::path& dir) {
  auto ix = std::make_shared<Index>();
  Index& I = *ix;
  {
    std::string text;
    try {
      text = read_file_string((dir / "manifest.json").string());
    } catch (const Error&) {
      throw Error("open " + (dir / "manifest.json").string());
    }
    try {
      I.manifest = Manifest::from_json(parse_json(text));
    } catch (const Error& e) {
      throw Error(std::string("manifest.json: ") + e.what());
    }
  }
  const Manifest& m = I.manifest;
  uint32_t expected = m.key_type == "v1" ? FORMAT_VERSION : FORMAT_VERSION_LEARNED;
  if (!(m.format == "skyanchor2-index" &&
        (m.version == expected || m.version == FORMAT_VERSION || (m.version == FORMAT_VERSION_DEEP && m.profile == "deep"))))
    throw Error("unsupported index (format " + m.format + " version " + std::to_string(m.version) + ", key type " + m.key_type + ")");
  if (m.key_type == "v1") {
  } else if (m.key_type == learned::KEY_TYPE) {
    I.coder = learned::load_index_coder(dir, m.coder_sha256);
  } else {
    throw Error("unsupported key type " + m.key_type);
  }
  I.codes_ = MappedFile::open(dir / "codes.bin");
  I.keys_ = MappedFile::open(dir / "keys.bin");
  I.stars_ = MappedFile::open(dir / "stars.bin");
  I.cells_ = MappedFile::open(dir / "cells.bin");
  size_t buckets = m.version == FORMAT_VERSION_DEEP ? (size_t(1) << DEEP_BUCKET_BITS) : N_BUCKETS;
  if (I.keys_.size() != (buckets + 1) * 4) throw Error("keys.bin size");
  if (!(I.codes_.size() % 8 == 0 && uint64_t(I.codes_.size()) / 8 == m.n_records)) throw Error("codes.bin size");
  if (I.stars_.size() != size_t(m.n_stars) * STAR_BYTES) throw Error("stars.bin size");
  I.star_grid = sphere::CubeGrid(m.star_cell_n);
  if (I.cells_.size() != (I.star_grid.cells() + 1) * 4) throw Error("cells.bin size");
  if (reinterpret_cast<uintptr_t>(I.codes_.data()) % 8 != 0 || reinterpret_cast<uintptr_t>(I.keys_.data()) % 4 != 0)
    throw Error("alignment");
  {
    const uint32_t* off = I.cell_offsets();
    uint32_t n_stars = uint32_t(m.n_stars);
    size_t ncells = I.star_grid.cells();
    I.cell_hint_.reserve((n_stars >> HINT_SHIFT) + 1);
    size_t c = 0;
    uint32_t id = 0;
    while (id < n_stars) {
      while (c < ncells && off[c + 1] <= id) ++c;
      if (c >= ncells) throw Error("cells.bin offsets are inconsistent with stars.bin");
      I.cell_hint_.push_back(uint32_t(c));
      id += 1u << HINT_SHIFT;
    }
  }
  std::error_code ec;
  if (fs::exists(dir / "deep.bin", ec) && fs::exists(dir / "deep_cells.bin", ec)) {
    DeepFiles d;
    d.stars = MappedFile::open(dir / "deep.bin");
    d.cells = MappedFile::open(dir / "deep_cells.bin");
    if (!(d.cells.size() == (I.star_grid.cells() + 1) * 4 && d.stars.size() % STAR_BYTES == 0)) throw Error("deep files");
    I.deep_stars_ = std::move(d);
  }
  I.r7 = to_radians(m.r7_arcmin) / 60.0;
  return ix;
}

uint64_t Index::mapped_bytes() const {
  uint64_t n = codes_.size() + keys_.size() + stars_.size() + cells_.size();
  if (deep_stars_) n += deep_stars_->stars.size() + deep_stars_->cells.size();
  return n;
}

uint64_t Index::touch() const {
  uint8_t sum = 0;
  auto walk = [&](const MappedFile& f) {
    for (size_t i = 0; i < f.size(); i += 4096) sum = uint8_t(sum + f.data()[i]);
  };
  walk(codes_);
  walk(keys_);
  walk(stars_);
  walk(cells_);
  if (deep_stars_) {
    walk(deep_stars_->stars);
    walk(deep_stars_->cells);
  }
  volatile uint8_t sink = sum;
  (void)sink;
  return mapped_bytes();
}

std::string Index::build_id() const {
  Sha256 h;
  h.update(manifest.catalog_fingerprint);
  h.update(manifest.coder_sha256 ? std::string_view(*manifest.coder_sha256) : std::string_view());
  h.update(manifest.profile);
  uint8_t le[8];
  for (int i = 0; i < 8; ++i) le[i] = uint8_t(manifest.n_records >> (8 * i));
  h.update(le, 8);
  uint8_t d[32];
  h.finalize(d);
  static const char* hx = "0123456789abcdef";
  std::string s = "idx-";
  for (int i = 0; i < 6; ++i) {
    s.push_back(hx[d[i] >> 4]);
    s.push_back(hx[d[i] & 15]);
  }
  return s;
}

void Index::probe(uint32_t key, std::vector<ProbeRecord>& out) const {
  out.clear();
  probe_each(key, [&](ProbeRecord r) { out.push_back(r); });
}

uint32_t Index::star_cell(uint32_t id) const {
  const uint32_t* off = cell_offsets();
  size_t c = cell_hint_[id >> HINT_SHIFT];
  while (off[c + 1] <= id) ++c;
  return uint32_t(c);
}

StarInfo Index::decode_star(const uint8_t* data, uint32_t id, uint32_t cell) const {
  const uint8_t* b = data + size_t(id) * STAR_BYTES;
  double fa = double(uint16_t(b[0] | (b[1] << 8)));
  double fb = double(uint16_t(b[2] | (b[3] << 8)));
  auto s = star_grid.split(cell);
  double step = FRAC_PI_2 / double(star_grid.n);
  double a = -FRAC_PI_4 + (double(s[1]) + (fa + 0.5) / 65536.0) * step;
  double bb = -FRAC_PI_4 + (double(s[2]) + (fb + 0.5) / 65536.0) * step;
  StarInfo si;
  si.v = sphere::face_vector(s[0], a, bb);
  si.d_rad = level_radius(r7, 8) * std::exp2(double(b[4]) / 16.0);
  si.mag = float(b[5]) / 10.0f - 2.0f;
  return si;
}

void Index::deep_cone(const Vec3& c, double radius, std::vector<uint32_t>& cells, std::vector<StarInfo>& out) const {
  out.clear();
  if (!deep_stars_) return;
  const uint32_t* off = reinterpret_cast<const uint32_t*>(deep_stars_->cells.data());
  if (radius >= 1.0) {
    cells.clear();
    for (uint32_t i = 0; i < star_grid.cells(); ++i) cells.push_back(i);
  } else {
    star_grid.cap_cells(c, radius, cells);
  }
  for (uint32_t cell : cells)
    for (uint32_t id = off[cell]; id < off[size_t(cell) + 1]; ++id) out.push_back(decode_star(deep_stars_->stars.data(), id, cell));
}

void Index::cone_candidates(const Vec3& c, double radius, std::vector<uint32_t>& cells,
                            std::vector<std::pair<uint32_t, uint32_t>>& out) const {
  out.clear();
  if (radius >= 1.0) {
    cells.clear();
    for (uint32_t i = 0; i < star_grid.cells(); ++i) cells.push_back(i);
  } else {
    star_grid.cap_cells(c, radius, cells);
  }
  const uint32_t* off = cell_offsets();
  for (uint32_t cell : cells)
    for (uint32_t id = off[cell]; id < off[size_t(cell) + 1]; ++id) out.emplace_back(id, cell);
}

static inline float decode_mag(uint8_t code) { return float(code) / 10.0f - 2.0f; }
static inline double decode_d(double r8, uint8_t code) { return r8 * std::exp2(double(code) / 16.0); }
static std::optional<uint8_t> mag_code_max(float mag_max) {
  for (int c = 255; c >= 0; --c)
    if (decode_mag(uint8_t(c)) <= mag_max) return uint8_t(c);
  return std::nullopt;
}

std::optional<uint8_t> Index::d_code_min(double d_min) const {
  double r8 = level_radius(r7, 8);
  for (int c = 0; c <= 255; ++c)
    if (decode_d(r8, uint8_t(c)) >= d_min) return uint8_t(c);
  return std::nullopt;
}

std::vector<StarInfo> Index::filtered_cone_candidates(const Vec3& c, double radius, double d_min,
                                                      std::optional<float> mag_limit, bool include_deep) const {
  auto dc = d_code_min(d_min);
  if (!dc) return {};
  uint8_t mc = 255;
  if (mag_limit) {
    auto code = mag_code_max(*mag_limit);
    if (!code) return {};
    mc = *code;
  }
  std::vector<uint32_t> cells;
  if (radius >= 1.0) {
    for (uint32_t i = 0; i < star_grid.cells(); ++i) cells.push_back(i);
  } else {
    star_grid.cap_cells(c, radius, cells);
  }
  std::vector<StarInfo> out;
  auto append = [&](const uint8_t* data, const uint32_t* offsets) {
    for (uint32_t cell : cells)
      for (uint32_t id = offsets[cell]; id < offsets[size_t(cell) + 1]; ++id) {
        const uint8_t* b = data + size_t(id) * STAR_BYTES;
        if (b[4] >= *dc && b[5] <= mc) out.push_back(decode_star(data, id, cell));
      }
  };
  append(stars_.data(), cell_offsets());
  if (include_deep && deep_stars_) append(deep_stars_->stars.data(), reinterpret_cast<const uint32_t*>(deep_stars_->cells.data()));
  return out;
}

std::vector<StarInfo> Index::cached_cone_candidates(const Vec3& c, double radius, double d_min,
                                                    std::optional<float> mag_limit, bool include_deep,
                                                    ConeCache& cache) const {
  auto dc = d_code_min(d_min);
  if (!dc) return {};
  uint8_t mc = 255;
  if (mag_limit) {
    auto code = mag_code_max(*mag_limit);
    if (!code) return {};
    mc = *code;
  }
  if (!(cache.has_cuts && cache.cut_d == *dc && cache.cut_m == mc)) {
    cache.clear();
    cache.has_cuts = true;
    cache.cut_d = *dc;
    cache.cut_m = mc;
  }
  if (radius >= 1.0) return filtered_cone_candidates(c, radius, d_min, mag_limit, include_deep);
  std::vector<uint32_t> cells;
  star_grid.cap_cells(c, radius, cells);
  std::vector<StarInfo> out;
  auto append = [&](const uint8_t* data, const uint32_t* offsets, CachedCells& part) {
    bool fill = part.cells.empty();
    size_t cursor = 0;
    for (uint32_t cell : cells) {
      if (!fill) {
        while (cursor < part.cells.size() && part.cells[cursor] < cell) ++cursor;
        if (cursor < part.cells.size() && part.cells[cursor] == cell) {
          auto [a, b] = part.ranges[cursor];
          out.insert(out.end(), part.stars.begin() + long(a), part.stars.begin() + long(b));
          continue;
        }
      }
      size_t start = out.size();
      for (uint32_t id = offsets[cell]; id < offsets[size_t(cell) + 1]; ++id) {
        const uint8_t* b = data + size_t(id) * STAR_BYTES;
        if (b[4] >= *dc && b[5] <= mc) out.push_back(decode_star(data, id, cell));
      }
      size_t n = out.size() - start;
      if (fill && part.cells.size() < 4096 && part.stars.size() + n <= 32768) {
        part.cells.push_back(cell);
        part.ranges.emplace_back(part.stars.size(), part.stars.size() + n);
        part.stars.insert(part.stars.end(), out.begin() + long(start), out.end());
      }
    }
  };
  append(stars_.data(), cell_offsets(), cache.base);
  if (include_deep && deep_stars_)
    append(deep_stars_->stars.data(), reinterpret_cast<const uint32_t*>(deep_stars_->cells.data()), cache.deep);
  return out;
}

// ---------- bright table ----------
void Index::set_bright_cut(float mag) {
  std::lock_guard<std::mutex> g(bright_mu_);
  bright_cut_ = mag;
  bright_.reset();
}

const Index::BrightTable& Index::bright_table() const {
  std::lock_guard<std::mutex> g(bright_mu_);
  if (!bright_) bright_ = std::make_unique<BrightTable>(decode_bright(bright_cut_));
  return *bright_;
}

Index::BrightTable Index::decode_bright(float cut) const {
  BrightTable t;
  t.cut = cut;
  t.cut_code = mag_code_max(cut);
  if (t.cut_code) {
    uint8_t mc = *t.cut_code;
    const uint32_t* off = cell_offsets();
    const uint8_t* data = stars_.data();
    for (size_t cell = 0; cell < star_grid.cells(); ++cell)
      for (uint32_t id = off[cell]; id < off[cell + 1]; ++id) {
        if (data[size_t(id) * STAR_BYTES + 5] > mc) continue;
        StarInfo st = decode_star(data, id, uint32_t(cell));
        t.stars.push_back(BrightStar{st.v, st.mag, st.d_rad, id});
      }
    std::stable_sort(t.stars.begin(), t.stars.end(), [](const BrightStar& a, const BrightStar& b) {
      int c = total_cmp(a.mag, b.mag);
      return c != 0 ? c < 0 : a.id < b.id;
    });
  }
  return t;
}

std::pair<const BrightStar*, size_t> Index::bright_stars(float mag_max) const {
  const BrightTable& t = bright_table();
  size_t lo = 0, hi = t.stars.size();
  while (lo < hi) {
    size_t mid = (lo + hi) / 2;
    if (t.stars[mid].mag <= mag_max) lo = mid + 1;
    else hi = mid;
  }
  return {t.stars.data(), lo};
}

void Index::cone_stars(const Vec3& c, double radius, float mag_max, double d_min, std::vector<BrightStar>& out) const {
  out.clear();
  if (!std::isfinite(radius) || radius < 0.0) return;
  const BrightTable& t = bright_table();
  if (mag_max > t.cut && radius <= BRIGHT_PATH_RADIUS) {
    cone_stars_cells(c, radius, mag_max, d_min, out);
    return;
  }
  double cos_r = std::cos(std::min(radius, PI));
  auto [p, n] = bright_stars(mag_max);
  for (size_t i = 0; i < n; ++i)
    if (p[i].d_rad >= d_min && sphere::dot(p[i].v, c) >= cos_r) out.push_back(p[i]);
  if (mag_max > t.cut) {
    uint8_t lo;
    if (t.cut_code) {
      if (*t.cut_code == 255) return;
      lo = uint8_t(*t.cut_code + 1);
    } else {
      lo = 0;
    }
    auto hi = mag_code_max(mag_max);
    if (!hi) return;
    auto dc = d_code_min(d_min);
    if (!dc) return;
    const uint32_t* off = cell_offsets();
    const uint8_t* data = stars_.data();
    for (size_t cell = 0; cell < star_grid.cells(); ++cell)
      for (uint32_t id = off[cell]; id < off[cell + 1]; ++id) {
        const uint8_t* b = data + size_t(id) * STAR_BYTES;
        if (b[5] < lo || b[5] > *hi || b[4] < *dc) continue;
        StarInfo st = decode_star(data, id, uint32_t(cell));
        if (sphere::dot(st.v, c) >= cos_r) out.push_back(BrightStar{st.v, st.mag, st.d_rad, id});
      }
  }
}

void Index::cone_stars_cells(const Vec3& c, double radius, float mag_max, double d_min,
                             std::vector<BrightStar>& out) const {
  out.clear();
  if (!std::isfinite(radius) || radius < 0.0) return;
  auto mc = mag_code_max(mag_max);
  if (!mc) return;
  auto dc = d_code_min(d_min);
  if (!dc) return;
  double cos_r = std::cos(std::min(radius, PI));
  std::vector<uint32_t> cells;
  if (radius >= 1.0) {
    for (uint32_t i = 0; i < star_grid.cells(); ++i) cells.push_back(i);
  } else {
    star_grid.cap_cells(c, radius, cells);
  }
  const uint32_t* off = cell_offsets();
  const uint8_t* data = stars_.data();
  for (uint32_t cell : cells)
    for (uint32_t id = off[cell]; id < off[size_t(cell) + 1]; ++id) {
      const uint8_t* b = data + size_t(id) * STAR_BYTES;
      if (b[5] > *mc || b[4] < *dc) continue;
      StarInfo st = decode_star(data, id, cell);
      if (sphere::dot(st.v, c) >= cos_r) out.push_back(BrightStar{st.v, st.mag, st.d_rad, id});
    }
}

}  // namespace skyshot::index
