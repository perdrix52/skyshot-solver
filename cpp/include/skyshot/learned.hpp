// Learned 32-bit key LGQ-GD. Port of src/learned.rs.
#pragma once
#include <filesystem>

#include "common.hpp"
#include "fingerprint.hpp"

namespace skyshot::learned {

inline constexpr const char* KEY_TYPE = "lgq-gd";
inline constexpr const char* INDEX_CODER = "coder";
inline constexpr size_t N_Q = 4097;
inline constexpr float SIGMA_FLOOR = 0.002f;
inline constexpr size_t M_BITS = 12;
inline constexpr float PRIOR_CUT = 12.0f;
inline constexpr double PARITY_C = 1.0;
inline constexpr double PARITY_S0 = 0.005;

inline constexpr size_t FE = 16, FG = 2, FN = 3, H1 = 64, H2 = 128, DIMS = 8, TIN = H1 + FG, NIN = H2 + FN;

struct Raw {
  std::array<std::array<double, 2>, 4> xy{};
  std::array<double, 5> d{};
  double sigma = 0;
  double edge = INF;
};

struct Basis {
  double len = 0;
  std::array<double, 3> u{}, v{};
  double spar = 0;
};

struct Feats {
  std::array<std::array<float, FE>, 3> elem{};
  std::array<float, FG> glob{};
  std::array<float, FN> noise{};
  bool flip = false;
  double spar = 0, spar_sigma = 0, len = 0;
};

std::optional<Basis> basis(const Raw& raw);
std::array<float, FE> elem_features(double u, double v, double d, double len);
std::optional<Feats> featurize(const Raw& raw, bool invert);
Feats featurize_basis(const Raw& raw, const Basis& b, bool invert);
double spar_sigma(const Basis& b, double sigma);
float parity_prior(double spar, double spar_sigma);
double ln_erfc(double x);
double erfc(double x);

struct ElemCache {
  std::vector<std::array<uint64_t, 4>> keys;
  std::vector<std::pair<std::array<float, FE>, std::array<float, H1>>> vals;
  void clear() {
    keys.clear();
    vals.clear();
  }
  size_t len() const { return keys.size(); }
  bool is_empty() const { return keys.empty(); }
};

struct ImageCode {
  Feats feats;
  std::array<float, DIMS> e{}, sigma{};
  std::array<double, DIMS> cdf{};
  uint32_t key = 0;
  std::array<float, 32> logodds{};
};

struct Probe {
  uint32_t key;
  bool flip;
  double cost;
};

class Coder {
 public:
  std::string sha256;

  static Coder load(const std::filesystem::path& prefix);
  static Coder from_bytes(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& json);
  static void copy_files(const std::filesystem::path& prefix, const std::filesystem::path& dir);
  std::array<uint32_t, DIMS> bits() const { return bits_; }

  std::array<float, H1> phi(const std::array<float, FE>& e) const;
  static std::array<float, TIN> pool(const std::array<float, H1>& h0, const std::array<float, H1>& h1,
                                     const std::array<float, H1>& h2, const std::array<float, FG>& glob);
  void heads(const std::array<float, TIN>& p, const std::array<float, FN>* noise, std::array<float, DIMS>& e,
             std::array<float, DIMS>& sig) const;
  std::pair<std::array<float, DIMS>, std::array<float, DIMS>> embed(const Feats& f, bool with_noise) const;
  double cdf(size_t j, float e) const;
  struct Quant {
    std::array<double, DIMS> c;
    std::array<uint32_t, DIMS> b;
    uint32_t key;
  };
  Quant quantise(const std::array<float, DIMS>& e) const;
  std::array<float, 32> logodds(const std::array<double, DIMS>& c, const std::array<uint32_t, DIMS>& b,
                                const std::array<float, DIMS>& s) const;
  std::optional<std::pair<uint32_t, bool>> catalogue_key(const Raw& raw) const;
  std::optional<ImageCode> image_code_cached(const Raw& raw, bool invert, ElemCache& cache) const;
  // Returns nullopt for a degenerate configuration, else whether the opposite-parity branch was enumerated.
  std::optional<bool> image_probes_cached(const Raw& raw, size_t k, ElemCache& cache, std::vector<Probe>& out) const;
  std::optional<ImageCode> image_code(const Raw& raw, bool invert) const;
  std::optional<bool> image_probes(const Raw& raw, size_t k, std::vector<Probe>& out) const;
  std::optional<bool> image_variants(const Raw& raw, size_t k, ElemCache& cache,
                                     std::vector<fingerprint::KeyVariant>& out) const;

 private:
  std::vector<float> w1_, b1_, w2_, b2_, w3_, b3_, w4_, b4_, we_, be_, ws_, bs_;
  std::vector<float> q_;
  std::array<uint32_t, DIMS> bits_{};
  std::array<uint32_t, DIMS> shift_{};
  uint32_t bin(size_t j, double c) const;
};

void branch_probes(const std::array<float, 32>& l, size_t kmax, double prior, bool flip, std::vector<Probe>& out);
Coder load_index_coder(const std::filesystem::path& dir, const std::optional<std::string>& sha256);

}  // namespace skyshot::learned
