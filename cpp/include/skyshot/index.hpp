// On-disk index format (little-endian, memory-mapped). Port of src/index.rs.
#pragma once
#include <filesystem>
#include <memory>
#include <mutex>

#include "common.hpp"
#include "json.hpp"
#include "learned.hpp"
#include "sphere.hpp"

namespace skyshot::index {

inline constexpr uint32_t FORMAT_VERSION = 1;
inline constexpr uint32_t FORMAT_VERSION_LEARNED = 2;
inline constexpr uint32_t FORMAT_VERSION_DEEP = 3;
inline constexpr uint32_t DEEP_BUCKET_BITS = 22;
inline constexpr uint32_t BUCKET_BITS = 20;
inline constexpr size_t N_BUCKETS = size_t(1) << BUCKET_BITS;
inline constexpr uint32_t FP_MASK = 0xfff;
inline constexpr uint32_t STAR_CELL_N = 256;
inline constexpr size_t N_LEVELS = 9;
inline constexpr double AR_SCALE = 256.0;
inline constexpr size_t STAR_BYTES = 6;
inline constexpr float BRIGHT_CUT_MAG = 7.5f;
inline constexpr double BRIGHT_PATH_RADIUS = 0.3;

inline double level_radius(double r7, uint32_t m) { return r7 * powi(2.0, 7 - int(m)); }

struct Record {
  uint64_t raw = 0;
  static Record make_deep(uint32_t fp, uint32_t anchor, uint32_t ar, uint32_t pa, bool flip, uint32_t level,
                          bool dominant, uint32_t dcode);
  static Record make(uint32_t fp, uint32_t anchor, uint32_t ar, uint32_t pa, bool flip, uint32_t level, bool dominant,
                     uint32_t dcode);
  uint32_t fp() const { return uint32_t(raw & 0xfff); }
  uint32_t anchor() const { return uint32_t((raw >> 12) & 0xffffff); }
  uint32_t ar_code() const { return uint32_t((raw >> 36) & 0x3ff); }
  uint32_t pa_code() const { return uint32_t((raw >> 46) & 0x3ff); }
  bool flip() const { return ((raw >> 56) & 1) == 1; }
  uint32_t level() const { return uint32_t((raw >> 57) & 7) + 1; }
  bool dominant() const { return ((raw >> 60) & 1) == 1; }
  uint32_t dcode() const { return uint32_t(raw >> 61); }
  double ar_rad(double r7) const { return level_radius(r7, level()) * std::exp2((double(ar_code()) + 0.5) / AR_SCALE); }
  double pa_rad() const { return (double(pa_code()) + 0.5) * TAU / 1024.0; }
};

struct ProbeRecord {
  uint64_t raw = 0;
  bool deep = false;
  uint32_t anchor() const { return deep ? uint32_t((raw >> 10) & 0x0fffffff) : Record{raw}.anchor(); }
  uint32_t level() const { return deep ? 7 + uint32_t((raw >> 59) & 1) : Record{raw}.level(); }
  uint32_t pa_code() const { return deep ? uint32_t((raw >> 48) & 1023) : Record{raw}.pa_code(); }
  bool flip() const { return (raw & (uint64_t(1) << (deep ? 58 : 56))) != 0; }
  uint32_t dcode() const { return uint32_t(raw >> 61); }
  double ar_rad(double r7) const {
    uint64_t ar = (raw >> (deep ? 38 : 36)) & 1023;
    return level_radius(r7, level()) * std::exp2((double(ar) + 0.5) / AR_SCALE);
  }
};

inline std::pair<size_t, uint32_t> bucket_fp(uint32_t key) {
  uint32_t h = fingerprint::mix32(key);
  return {size_t(h >> 12), h & FP_MASK};
}
inline std::pair<size_t, uint32_t> bucket_fp_layout(uint32_t key, bool deep) {
  if (deep) {
    uint32_t h = fingerprint::mix32(key);
    return {size_t(h >> 10), h & 1023};
  }
  return bucket_fp(key);
}

struct LevelInfo {
  uint32_t m = 0;
  double radius_arcmin = 0;
  uint64_t stars = 0, anchors = 0, records = 0;
};

struct Manifest {
  std::string format;
  uint32_t version = 0;
  std::string profile;
  double r7_arcmin = 0;
  std::string key_spec;
  std::vector<LevelInfo> levels;
  uint64_t n_records = 0, n_stars = 0, n_catalog = 0;
  uint32_t star_cell_n = 0;
  std::string catalog, catalog_fingerprint;
  double build_seconds = 0;
  std::vector<std::pair<std::string, uint64_t>> files;
  uint64_t stop_words = 0;
  std::string key_type = "v1";
  std::optional<std::string> coder_sha256;

  Json to_json() const;
  static Manifest from_json(const Json& j);
};

struct StarInfo {
  Vec3 v{};
  double d_rad = 0;
  float mag = 0;
  bool operator==(const StarInfo& o) const { return v == o.v && d_rad == o.d_rad && mag == o.mag; }
};

struct BrightStar {
  Vec3 v{};
  float mag = 0;
  double d_rad = 0;
  uint32_t id = 0;
};

struct CachedCells {
  std::vector<uint32_t> cells;
  std::vector<std::pair<size_t, size_t>> ranges;
  std::vector<StarInfo> stars;
};

struct ConeCache {
  bool has_cuts = false;
  uint8_t cut_d = 0, cut_m = 0;
  CachedCells base, deep;
  void clear() {
    has_cuts = false;
    for (CachedCells* p : {&base, &deep}) {
      p->cells.clear();
      p->ranges.clear();
      p->stars.clear();
    }
  }
};

// Read-only memory-mapped file.
class MappedFile {
 public:
  MappedFile() = default;
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& o) noexcept { swap(o); }
  MappedFile& operator=(MappedFile&& o) noexcept {
    swap(o);
    return *this;
  }
  static MappedFile open(const std::filesystem::path& path);
  const uint8_t* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void swap(MappedFile& o) noexcept {
    std::swap(data_, o.data_);
    std::swap(size_, o.size_);
    std::swap(os_a_, o.os_a_);
    std::swap(os_b_, o.os_b_);
  }
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  void* os_a_ = nullptr;  // file mapping handle (Windows) / unused
  void* os_b_ = nullptr;
};

std::optional<std::filesystem::path> companion_wide_dir(const std::filesystem::path& main);
std::optional<std::filesystem::path> companion_deep_dir(const std::filesystem::path& main);

class Index {
 public:
  Manifest manifest;
  double r7 = 0;
  sphere::CubeGrid star_grid;
  std::optional<learned::Coder> coder;

  static std::shared_ptr<Index> open(const std::filesystem::path& dir);

  uint64_t mapped_bytes() const;
  uint64_t touch() const;
  std::string build_id() const;

  const uint64_t* codes() const { return reinterpret_cast<const uint64_t*>(codes_.data()); }
  size_t n_codes() const { return codes_.size() / 8; }
  const uint32_t* offsets() const { return reinterpret_cast<const uint32_t*>(keys_.data()); }
  const uint32_t* cell_offsets() const { return reinterpret_cast<const uint32_t*>(cells_.data()); }

  // Records whose key equals `key` exactly, appended to `out`.
  void probe(uint32_t key, std::vector<ProbeRecord>& out) const;
  template <class F>
  void probe_each(uint32_t key, F&& f) const {
    bool deep = manifest.version == FORMAT_VERSION_DEEP;
    uint64_t mask = deep ? 1023 : 0xfff;
    auto [b, fp] = bucket_fp_layout(key, deep);
    const uint32_t* off = offsets();
    const uint64_t* c = codes() + off[b];
    size_t n = off[b + 1] - off[b];
    size_t lo = 0, hi = n;
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (uint32_t(c[mid] & mask) < fp) lo = mid + 1;
      else hi = mid;
    }
    for (size_t i = lo; i < n && uint32_t(c[i] & mask) == fp; ++i) f(ProbeRecord{c[i], deep});
  }
  size_t n_buckets() const { return keys_.size() / 4 - 1; }
  size_t bucket_len(size_t b) const { return offsets()[b + 1] - offsets()[b]; }
  size_t n_stars() const { return stars_.size() / STAR_BYTES; }
  uint32_t star_cell(uint32_t id) const;
  StarInfo star_in_cell(uint32_t id, uint32_t cell) const { return decode_star(stars_.data(), id, cell); }
  StarInfo star(uint32_t id) const { return star_in_cell(id, star_cell(id)); }
  bool has_deep() const { return deep_stars_.has_value(); }
  void deep_cone(const Vec3& c, double radius, std::vector<uint32_t>& cells, std::vector<StarInfo>& out) const;
  void cone_candidates(const Vec3& c, double radius, std::vector<uint32_t>& cells,
                       std::vector<std::pair<uint32_t, uint32_t>>& out) const;
  std::vector<StarInfo> filtered_cone_candidates(const Vec3& c, double radius, double d_min,
                                                 std::optional<float> mag_limit, bool include_deep) const;
  std::vector<StarInfo> cached_cone_candidates(const Vec3& c, double radius, double d_min,
                                               std::optional<float> mag_limit, bool include_deep,
                                               ConeCache& cache) const;
  double level_radius_m(uint32_t m) const { return level_radius(r7, m); }

  float bright_cut() const { return bright_cut_; }
  void set_bright_cut(float mag);
  // Stars with mag <= min(mag_max, bright_cut()), sorted by (magnitude, id).
  std::pair<const BrightStar*, size_t> bright_stars(float mag_max) const;
  void cone_stars(const Vec3& c, double radius, float mag_max, double d_min, std::vector<BrightStar>& out) const;
  void cone_stars_cells(const Vec3& c, double radius, float mag_max, double d_min, std::vector<BrightStar>& out) const;

 private:
  struct BrightTable {
    float cut = 0;
    std::optional<uint8_t> cut_code;
    std::vector<BrightStar> stars;
  };
  StarInfo decode_star(const uint8_t* data, uint32_t id, uint32_t cell) const;
  const BrightTable& bright_table() const;
  BrightTable decode_bright(float cut) const;
  std::optional<uint8_t> d_code_min(double d_min) const;

  MappedFile codes_, keys_, stars_, cells_;
  std::vector<uint32_t> cell_hint_;
  struct DeepFiles {
    MappedFile stars, cells;
  };
  std::optional<DeepFiles> deep_stars_;
  mutable std::mutex bright_mu_;
  mutable std::unique_ptr<BrightTable> bright_;
  float bright_cut_ = BRIGHT_CUT_MAG;
  static constexpr uint32_t HINT_SHIFT = 6;
};

}  // namespace skyshot::index
