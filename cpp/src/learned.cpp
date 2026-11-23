#include "skyshot/learned.hpp"

#include <bit>
#include <filesystem>
#include <queue>

#include "skyshot/json.hpp"
#include "skyshot/sha256.hpp"

namespace skyshot::learned {

namespace {
inline double nan0(double x) { return std::isnan(x) ? 0.0 : x; }
inline uint32_t gray(uint32_t n) { return n ^ (n >> 1); }
}  // namespace

std::optional<Basis> basis(const Raw& raw) {
  const auto& r = raw.xy[0];
  double len = std::hypot(r[0], r[1]);
  if (std::isnan(len) || len <= 0.0) return std::nullopt;
  double ex[2] = {r[0] / len, r[1] / len};
  double ey[2] = {-ex[1], ex[0]};
  Basis b;
  b.len = len;
  for (size_t j = 0; j < 3; ++j) {
    const auto& p = raw.xy[j + 1];
    b.u[j] = (p[0] * ex[0] + p[1] * ex[1]) / len;
    b.v[j] = (p[0] * ey[0] + p[1] * ey[1]) / len;
    double r2 = b.u[j] * b.u[j] + b.v[j] * b.v[j];
    if (!(r2 > 0.0 && std::isfinite(r2))) return std::nullopt;
  }
  double spar = -0.0;
  for (size_t j = 0; j < 3; ++j) spar += b.v[j] / (b.u[j] * b.u[j] + b.v[j] * b.v[j]);
  b.spar = spar;
  return b;
}

std::array<float, FE> elem_features(double u, double v, double d, double len) {
  double rho = 0.5 * std::log(u * u + v * v);
  double th = std::atan2(v, u);
  double f[FE] = {};
  f[0] = rho;
  for (int k = 1; k <= 4; ++k) {
    double a = double(k) * th;
    f[2 * k - 1] = std::cos(a);
    f[2 * k] = std::sin(a);
  }
  const double ws[3] = {PI, 2.0 * PI, 4.0 * PI};
  for (int i = 0; i < 3; ++i) {
    f[9 + 2 * i] = std::sin(ws[i] * rho);
    f[10 + 2 * i] = std::cos(ws[i] * rho);
  }
  f[15] = clampd(std::log(d / len), -2.0, 3.0);
  std::array<float, FE> out;
  for (size_t i = 0; i < FE; ++i) out[i] = float(nan0(f[i]));
  return out;
}

std::optional<Feats> featurize(const Raw& raw, bool invert) {
  auto b = basis(raw);
  if (!b) return std::nullopt;
  return featurize_basis(raw, *b, invert);
}

static Feats featurize_basis_tail(const Raw& raw, const Basis& b, bool flip) {
  double len = b.len;
  Feats f;
  f.glob = {float(nan0(clampd(std::log(raw.d[0] / len), -2.0, 4.0))), float(nan0(clampd(std::log(raw.d[1] / len), -2.0, 3.0)))};
  f.noise = {float(clampd(std::log(fmax_r(raw.sigma, 1e-300) / len), -12.0, 0.0)), raw.d[0] >= raw.edge ? 1.0f : 0.0f,
             float(clampd(std::log(raw.edge / len), -1.0, 4.0))};
  f.flip = flip;
  f.spar = b.spar;
  f.spar_sigma = spar_sigma(b, raw.sigma);
  f.len = len;
  return f;
}

Feats featurize_basis(const Raw& raw, const Basis& b, bool invert) {
  bool flip = (b.spar < 0.0) != invert;
  double sgn = flip ? -1.0 : 1.0;
  Feats f = featurize_basis_tail(raw, b, flip);
  for (size_t j = 0; j < 3; ++j) f.elem[j] = elem_features(b.u[j], b.v[j] * sgn, raw.d[j + 2], b.len);
  return f;
}

double spar_sigma(const Basis& b, double sigma) {
  const auto& us = b.u;
  const auto& vs = b.v;
  double gu[3], gv[3];
  for (size_t j = 0; j < 3; ++j) {
    double r2 = us[j] * us[j] + vs[j] * vs[j];
    gu[j] = -2.0 * us[j] * vs[j] / (r2 * r2);
    gv[j] = (us[j] * us[j] - vs[j] * vs[j]) / (r2 * r2);
  }
  auto sum = [](auto f) { return f(0) + (f(1) + f(2)); };
  double s1 = sum([&](size_t j) { return gu[j] * gu[j] + gv[j] * gv[j]; });
  double s2 = sum([&](size_t j) { return gu[j]; });
  double s3 = sum([&](size_t j) { return gv[j]; });
  double s4 = sum([&](size_t j) { return -gv[j] * us[j] + gu[j] * vs[j]; });
  double var = s1 + s2 * s2 + s3 * s3 + s4 * s4 + b.spar * b.spar;
  return sigma / b.len * std::sqrt(var);
}

float parity_prior(double spar, double spar_sigma_) {
  double a = PARITY_C * spar_sigma_;
  double sig = std::sqrt(a * a + PARITY_S0 * PARITY_S0);
  double t = std::fabs(spar) / sig;
  if (!std::isfinite(t)) return std::numeric_limits<float>::infinity();
  double ln_q = ln_erfc(t / SQRT_2) - 0.693147180559945309417232121458176568;
  return float(std::log1p(-std::exp(ln_q)) - ln_q);
}

double ln_erfc(double x) {
  if (x < 2.0) {
    double x2 = x * x;
    double term = x, sum = x, n = 0.0;
    while (term > 1e-17 * sum) {
      n += 1.0;
      term *= 2.0 * x2 / (2.0 * n + 1.0);
      sum += term;
    }
    return std::log1p(-(2.0 / std::sqrt(PI)) * std::exp(-x2) * sum);
  }
  double f = x;
  for (int n = 80; n >= 1; --n) f = x + (double(n) / 2.0) / f;
  return -x * x - std::log(std::sqrt(PI) * f);
}
double erfc(double x) { return std::exp(ln_erfc(x)); }

// ---------------------------------------------------------------------------------------------
// y[o] = act(sum_i w[i*OUT+o] * x[i] + bias[o]); inputs in index order, then the bias (as the Rust code).
template <size_t IN, size_t OUT>
static void dense(const std::vector<float>& w, const std::vector<float>& bias, const float* x, float* y, bool relu) {
  float acc[OUT];
  for (size_t o = 0; o < OUT; ++o) acc[o] = 0.0f;
  for (size_t i = 0; i < IN; ++i) {
    const float xv = x[i];
    const float* wr = w.data() + i * OUT;
    for (size_t o = 0; o < OUT; ++o) acc[o] += wr[o] * xv;
  }
  for (size_t o = 0; o < OUT; ++o) {
    float v = acc[o] + bias[o];
    y[o] = (relu && (std::isnan(v) || v <= 0.0f)) ? 0.0f : v;
  }
}

Coder Coder::load(const std::filesystem::path& prefix) {
  auto with_ext = [&](const char* ext) {
    std::filesystem::path p = prefix;
    p.replace_extension(ext);
    return p;
  };
  auto jp = with_ext(".json"), bp = with_ext(".f32");
  std::vector<uint8_t> jbytes, blob;
  try {
    jbytes = read_file_bytes(jp.string());
  } catch (const Error&) {
    throw Error("read " + jp.string());
  }
  try {
    blob = read_file_bytes(bp.string());
  } catch (const Error&) {
    throw Error("read " + bp.string());
  }
  return from_bytes(blob, jbytes);
}

Coder Coder::from_bytes(const std::vector<uint8_t>& blob, const std::vector<uint8_t>& json) {
  Json meta;
  try {
    meta = parse_json(std::string_view(reinterpret_cast<const char*>(json.data()), json.size()));
  } catch (const Error& e) {
    throw Error(std::string("coder json: ") + e.what());
  }
  auto str_is = [](const Json* j, const char* s) { return j && j->is_string() && j->as_str() == s; };
  const Json* ver = meta.get("version");
  if (!str_is(meta.get("format"), "skyanchor2-learned-coder-lgq") || !ver || !ver->as_u64() || *ver->as_u64() != 1)
    throw Error("unsupported coder format");
  const Json* a = meta.get("arch");
  auto num = [&](const char* k) -> uint64_t {
    const Json* v = a ? a->get(k) : nullptr;
    if (v) {
      if (auto u = v->as_u64()) return *u;
    }
    return UINT64_MAX;
  };
  const Json* use_extra = a ? a->get("use_extra") : nullptr;
  if (!(num("fe") == FE && num("fg") == FG && num("fn") == FN && num("h1") == H1 && num("h2") == H2 &&
        num("dims") == DIMS && use_extra && use_extra->is_bool() && !use_extra->as_bool()))
    throw Error("coder architecture is not LGQ-GD (fe 16, fg 2, h1 64, h2 128, dims 8)");
  auto cols = [&](const char* k) {
    std::vector<uint64_t> out;
    const Json* v = a ? a->get(k) : nullptr;
    if (v && v->is_array())
      for (const auto& x : v->items())
        if (auto u = x.as_u64()) out.push_back(*u);
    return out;
  };
  std::vector<uint64_t> ec = cols("elem_cols"), gc = cols("glob_cols");
  bool ok = ec.size() == 16 && gc.size() == 2 && gc[0] == 0 && gc[1] == 1;
  for (size_t i = 0; ok && i < 16; ++i) ok = ec[i] == i;
  if (!ok) throw Error("coder feature columns are not the GD set");
  const Json* bj = meta.get("bits_per_dim");
  if (!bj || !bj->is_array()) throw Error("bits_per_dim");
  std::vector<uint32_t> bits;
  for (const auto& x : bj->items()) bits.push_back(uint32_t(x.as_u64().value_or(0)));
  uint32_t bsum = 0;
  bool bok = bits.size() == DIMS;
  for (uint32_t b : bits) {
    bsum += b;
    bok = bok && b >= 1;
  }
  if (!bok || bsum != 32) throw Error("bits_per_dim");
  if (blob.size() % 4 != 0) throw Error("coder blob size");
  std::vector<float> floats(blob.size() / 4);
  std::memcpy(floats.data(), blob.data(), blob.size());  // little-endian host
  const Json* tensors = meta.get("tensors");
  if (!tensors || !tensors->is_array()) throw Error("tensors");
  auto get = [&](const std::string& name, size_t rows, size_t colsn) -> const float* {
    const Json* t = nullptr;
    for (const auto& x : tensors->items()) {
      const Json* n = x.get("name");
      if (n && n->is_string() && n->as_str() == name) {
        t = &x;
        break;
      }
    }
    if (!t) throw Error("tensor " + name);
    const Json* sh = t->get("shape");
    if (!sh || !sh->is_array()) throw Error("shape");
    size_t n = 1;
    for (const auto& x : sh->items()) n *= size_t(x.as_u64().value_or(0));
    if (n != rows * colsn) throw Error("tensor " + name + ": bad shape");
    const Json* offj = t->get("offset_floats");
    if (!offj || !offj->as_u64()) throw Error("offset");
    size_t off = size_t(*offj->as_u64());
    if (off + n > floats.size()) throw Error("tensor " + name + " out of range");
    return floats.data() + off;
  };
  auto tr = [&](const std::string& name, size_t out, size_t inp) {
    const float* w = get(name, out, inp);
    std::vector<float> t(out * inp);
    for (size_t o = 0; o < out; ++o)
      for (size_t i = 0; i < inp; ++i) t[i * out + o] = w[o * inp + i];
    return t;
  };
  auto vecf = [&](const std::string& name, size_t n) {
    const float* p = get(name, n, 1);
    return std::vector<float>(p, p + n);
  };
  Coder c;
  const float* q32 = get("quantiles", DIMS, N_Q);
  c.q_.assign(q32, q32 + DIMS * N_Q);
  for (size_t j = 0; j < DIMS; ++j) {
    const float* row = c.q_.data() + j * N_Q;
    for (size_t i = 0; i + 1 < N_Q; ++i)
      if (!(row[i] <= row[i + 1])) throw Error("quantiles of dim " + std::to_string(j) + " not sorted");
  }
  for (size_t j = 0; j < DIMS; ++j) c.bits_[j] = bits[j];
  c.shift_[0] = 0;
  for (size_t j = 1; j < DIMS; ++j) c.shift_[j] = c.shift_[j - 1] + c.bits_[j - 1];
  Sha256 h;
  h.update(blob.data(), blob.size());
  h.update(json.data(), json.size());
  c.sha256 = h.hex();
  c.w1_ = tr("phi1.weight", H1, FE);
  c.b1_ = vecf("phi1.bias", H1);
  c.w2_ = tr("phi2.weight", H1, H1);
  c.b2_ = vecf("phi2.bias", H1);
  c.w3_ = tr("t1.weight", H2, TIN);
  c.b3_ = vecf("t1.bias", H2);
  c.w4_ = tr("t2.weight", H2, H2);
  c.b4_ = vecf("t2.bias", H2);
  c.we_ = tr("emb.weight", DIMS, H2);
  c.be_ = vecf("emb.bias", DIMS);
  c.ws_ = tr("noise.weight", DIMS, NIN);
  c.bs_ = vecf("noise.bias", DIMS);
  return c;
}

void Coder::copy_files(const std::filesystem::path& prefix, const std::filesystem::path& dir) {
  namespace fs = std::filesystem;
  for (const char* ext : {"f32", "json"}) {
    fs::path src = prefix;
    src.replace_extension(ext);
    fs::path dst = dir / (std::string(INDEX_CODER) + "." + ext);
    std::error_code e1, e2;
    auto a = fs::canonical(src, e1), b = fs::canonical(dst, e2);
    if (!e1 && !e2 && a == b) continue;
    std::error_code ec;
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing, ec);
    if (ec) throw Error("copy " + src.string() + " -> " + dst.string());
  }
}

std::array<float, H1> Coder::phi(const std::array<float, FE>& e) const {
  float h[H1];
  dense<FE, H1>(w1_, b1_, e.data(), h, true);
  std::array<float, H1> o;
  dense<H1, H1>(w2_, b2_, h, o.data(), true);
  return o;
}

std::array<float, TIN> Coder::pool(const std::array<float, H1>& h0, const std::array<float, H1>& h1,
                                   const std::array<float, H1>& h2, const std::array<float, FG>& glob) {
  std::array<float, TIN> p;
  for (size_t o = 0; o < H1; ++o) p[o] = h0[o] + h1[o] + h2[o];
  for (size_t i = 0; i < FG; ++i) p[H1 + i] = glob[i];
  return p;
}

void Coder::heads(const std::array<float, TIN>& p, const std::array<float, FN>* noise, std::array<float, DIMS>& e,
                  std::array<float, DIMS>& sig) const {
  float x1[H2], x[H2];
  dense<TIN, H2>(w3_, b3_, p.data(), x1, true);
  dense<H2, H2>(w4_, b4_, x1, x, true);
  dense<H2, DIMS>(we_, be_, x, e.data(), false);
  if (noise) {
    float xn[NIN];
    for (size_t i = 0; i < H2; ++i) xn[i] = x[i];
    for (size_t i = 0; i < FN; ++i) xn[H2 + i] = (*noise)[i];
    float z[DIMS];
    dense<NIN, DIMS>(ws_, bs_, xn, z, false);
    for (size_t j = 0; j < DIMS; ++j) sig[j] = SIGMA_FLOOR + std::exp(z[j]);
  }
}

std::pair<std::array<float, DIMS>, std::array<float, DIMS>> Coder::embed(const Feats& f, bool with_noise) const {
  auto h0 = phi(f.elem[0]), h1 = phi(f.elem[1]), h2 = phi(f.elem[2]);
  auto p = pool(h0, h1, h2, f.glob);
  std::array<float, DIMS> e{}, s{};
  heads(p, with_noise ? &f.noise : nullptr, e, s);
  return {e, s};
}

double Coder::cdf(size_t j, float e) const {
  double x = double(e);
  if (std::isnan(x)) return x;
  const float* q = q_.data() + j * N_Q;
  size_t pp = size_t(std::upper_bound(q, q + N_Q, e) - q);  // count of v <= e
  if (pp == 0) return 0.0;
  if (pp == N_Q) return 1.0;
  size_t k = pp - 1;
  double yk = double(k) / double(N_Q - 1);
  double q0 = double(q[k]), q1 = double(q[k + 1]);
  if (q0 == x) return yk;
  double slope = (1.0 / double(N_Q - 1)) / (q1 - q0);
  return slope * (x - q0) + yk;
}

uint32_t Coder::bin(size_t j, double c) const {
  double n = double(1u << bits_[j]);
  return sat_cast<uint32_t>(fmin_r(std::floor(c * n), n - 1.0));
}

Coder::Quant Coder::quantise(const std::array<float, DIMS>& e) const {
  Quant q{};
  q.key = 0;
  for (size_t j = 0; j < DIMS; ++j) {
    q.c[j] = cdf(j, e[j]);
    q.b[j] = bin(j, q.c[j]);
    q.key |= gray(q.b[j]) << shift_[j];
  }
  return q;
}

std::array<float, 32> Coder::logodds(const std::array<double, DIMS>& c, const std::array<uint32_t, DIMS>& b,
                                     const std::array<float, DIMS>& s) const {
  std::array<float, 32> l{};
  double p_sat = 1e-12;
  double saturated = std::log((1.0 - p_sat) / p_sat);
  for (size_t j = 0; j < DIMS; ++j) {
    uint32_t nbit = bits_[j];
    int64_t n = int64_t(1) << nbit;
    double w = 1.0 / double(n);
    int64_t bj = int64_t(b[j]);
    uint32_t g = gray(b[j]);
    double sj = double(std::isnan(s[j]) ? 1e-4f : std::max(s[j], 1e-4f));
    for (uint32_t k = 0; k < nbit; ++k) {
      int64_t step = int64_t(1) << (k + 1);
      int64_t first = int64_t(1) << k;
      int64_t up = bj + 1 + rem_euclid_i(first - (bj + 1), step);
      int64_t dn = bj - rem_euclid_i(bj - first, step);
      double m_up = up < n ? double(up) * w - c[j] : INF;
      double m_dn = dn >= 1 ? c[j] - double(dn) * w : INF;
      double m = fmax_r(fmin_r(m_up, m_dn), 0.0);
      double z = -m / sj;
      double mag;
      if (z <= -27.0) {
        mag = saturated;
      } else {
        double p = clampd(0.5 * std::exp(z), 1e-12, 0.5 - 1e-7);
        mag = std::log((1.0 - p) / p);
      }
      uint32_t bit = (g >> k) & 1;
      l[size_t(shift_[j] + k)] = float(bit == 1 ? mag : -mag);
    }
  }
  return l;
}

std::optional<std::pair<uint32_t, bool>> Coder::catalogue_key(const Raw& raw) const {
  auto f = featurize(raw, false);
  if (!f) return std::nullopt;
  auto [e, s] = embed(*f, false);
  (void)s;
  auto q = quantise(e);
  return std::make_pair(q.key, f->flip);
}

std::optional<ImageCode> Coder::image_code_cached(const Raw& raw, bool invert, ElemCache& cache) const {
  auto bo = basis(raw);
  if (!bo) return std::nullopt;
  const Basis& b = *bo;
  double len = b.len;
  bool flip = (b.spar < 0.0) != invert;
  double sgn = flip ? -1.0 : 1.0;
  std::array<std::array<float, FE>, 3> elem;
  std::array<std::array<float, H1>, 3> h;
  auto bits_of = [](double v) { return std::bit_cast<uint64_t>(v); };
  for (size_t j = 0; j < 3; ++j) {
    double u = b.u[j], v = b.v[j] * sgn, d = raw.d[j + 2];
    std::array<uint64_t, 4> key = {bits_of(len), bits_of(u), bits_of(v), bits_of(d)};
    size_t slot = cache.keys.size();
    for (size_t i = 0; i < cache.keys.size(); ++i)
      if (cache.keys[i] == key) {
        slot = i;
        break;
      }
    if (slot == cache.keys.size()) {
      auto f = elem_features(u, v, d, len);
      cache.keys.push_back(key);
      cache.vals.emplace_back(f, phi(f));
    }
    elem[j] = cache.vals[slot].first;
    h[j] = cache.vals[slot].second;
  }
  ImageCode ic;
  ic.feats = featurize_basis_tail(raw, b, flip);
  ic.feats.elem = elem;
  auto p = pool(h[0], h[1], h[2], ic.feats.glob);
  heads(p, &ic.feats.noise, ic.e, ic.sigma);
  auto q = quantise(ic.e);
  ic.cdf = q.c;
  ic.key = q.key;
  ic.logodds = logodds(q.c, q.b, ic.sigma);
  return ic;
}

std::optional<bool> Coder::image_probes_cached(const Raw& raw, size_t k, ElemCache& cache,
                                               std::vector<Probe>& out) const {
  out.clear();
  auto c1 = image_code_cached(raw, false, cache);
  if (!c1) return std::nullopt;
  float prior2 = parity_prior(c1->feats.spar, c1->feats.spar_sigma);
  branch_probes(c1->logodds, k, 0.0, c1->feats.flip, out);
  bool both = prior2 < PRIOR_CUT;
  if (both) {
    auto c2 = image_code_cached(raw, true, cache);
    if (!c2) return std::nullopt;
    branch_probes(c2->logodds, k, double(prior2), c2->feats.flip, out);
    std::stable_sort(out.begin(), out.end(), [](const Probe& a, const Probe& b) { return total_cmp(a.cost, b.cost) < 0; });
    if (out.size() > k) out.resize(k);
  }
  return both;
}

std::optional<ImageCode> Coder::image_code(const Raw& raw, bool invert) const {
  auto feats = featurize(raw, invert);
  if (!feats) return std::nullopt;
  auto [e, sigma] = embed(*feats, true);
  auto q = quantise(e);
  ImageCode ic;
  ic.feats = *feats;
  ic.e = e;
  ic.sigma = sigma;
  ic.cdf = q.c;
  ic.key = q.key;
  ic.logodds = logodds(q.c, q.b, sigma);
  return ic;
}

std::optional<bool> Coder::image_probes(const Raw& raw, size_t k, std::vector<Probe>& out) const {
  out.clear();
  auto c1 = image_code(raw, false);
  if (!c1) return std::nullopt;
  float prior2 = parity_prior(c1->feats.spar, c1->feats.spar_sigma);
  branch_probes(c1->logodds, k, 0.0, c1->feats.flip, out);
  bool both = prior2 < PRIOR_CUT;
  if (both) {
    auto c2 = image_code(raw, true);
    if (!c2) return std::nullopt;
    branch_probes(c2->logodds, k, double(prior2), c2->feats.flip, out);
    std::stable_sort(out.begin(), out.end(), [](const Probe& a, const Probe& b) { return total_cmp(a.cost, b.cost) < 0; });
    if (out.size() > k) out.resize(k);
  }
  return both;
}

std::optional<bool> Coder::image_variants(const Raw& raw, size_t k, ElemCache& cache,
                                          std::vector<fingerprint::KeyVariant>& out) const {
  std::vector<Probe> p;
  p.reserve(2 * k);
  auto r = image_probes_cached(raw, k, cache, p);
  out.clear();
  for (const auto& pr : p) out.push_back(fingerprint::KeyVariant{pr.key, pr.flip});
  return r;
}

void branch_probes(const std::array<float, 32>& l, size_t kmax, double prior, bool flip, std::vector<Probe>& out) {
  if (kmax == 0) return;
  uint32_t base = 0;
  for (size_t i = 0; i < 32; ++i)
    if (l[i] > 0.0f) base |= 1u << i;
  std::array<uint8_t, 32> idx;
  for (size_t i = 0; i < 32; ++i) idx[i] = uint8_t(i);
  std::stable_sort(idx.begin(), idx.end(), [&](uint8_t a, uint8_t b) {
    int c = total_cmp(std::fabs(l[a]), std::fabs(l[b]));
    return c != 0 ? c < 0 : a < b;
  });
  double a[M_BITS];
  uint32_t bit[M_BITS];
  for (size_t i = 0; i < M_BITS; ++i) {
    a[i] = double(std::fabs(l[idx[i]]));
    bit[i] = 1u << idx[i];
  }
  out.push_back(Probe{base, flip, prior});
  if (kmax == 1) return;
  struct E {
    double cost;
    uint8_t last;
    uint32_t mask;
  };
  // Max-heap whose top is the entry with the lowest cost, then the lowest mask.
  auto less = [](const E& x, const E& y) {
    int c = total_cmp(y.cost, x.cost);
    if (c != 0) return c < 0;
    return y.mask < x.mask;
  };
  std::priority_queue<E, std::vector<E>, decltype(less)> heap(less);
  heap.push(E{a[0], 0, bit[0]});
  size_t n = 1;
  while (n < kmax) {
    if (heap.empty()) break;
    E top = heap.top();
    heap.pop();
    out.push_back(Probe{base ^ top.mask, flip, prior + top.cost});
    ++n;
    size_t nx = size_t(top.last) + 1;
    if (nx < M_BITS) {
      heap.push(E{top.cost + a[nx], uint8_t(nx), top.mask | bit[nx]});
      heap.push(E{top.cost - a[top.last] + a[nx], uint8_t(nx), (top.mask ^ bit[top.last]) | bit[nx]});
    }
  }
}

Coder load_index_coder(const std::filesystem::path& dir, const std::optional<std::string>& sha256) {
  Coder c = Coder::load(dir / INDEX_CODER);
  if (sha256 && *sha256 == c.sha256) return c;
  if (sha256) throw Error("coder SHA-256 mismatch: manifest " + *sha256 + ", files " + c.sha256);
  throw Error("learned index without coder_sha256");
}

}  // namespace skyshot::learned
