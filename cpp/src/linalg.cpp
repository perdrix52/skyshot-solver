#include "skyshot/linalg.hpp"

namespace skyshot::linalg {

std::optional<Mat> cholesky(const Mat& m) {
  size_t n = m.r;
  // Column Cholesky in the operation order of nalgebra's lower-triangular variant.
  Mat l(n, n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j <= i; ++j) l(i, j) = m(i, j);
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < j; ++k) {
      double factor = -l(j, k);
      for (size_t i = j; i < n; ++i) l(i, j) += factor * l(i, k);
    }
    double diag = l(j, j);
    if (!(diag > 0.0)) return std::nullopt;
    diag = std::sqrt(diag);
    double inv = 1.0 / diag;
    l(j, j) = diag;
    for (size_t i = j + 1; i < n; ++i) l(i, j) *= inv;
  }
  return l;
}

std::vector<double> cholesky_solve(const Mat& l, const std::vector<double>& b) {
  size_t n = l.r;
  std::vector<double> x = b;
  for (size_t i = 0; i < n; ++i) {  // L y = b
    double s = x[i];
    for (size_t k = 0; k < i; ++k) s -= l(i, k) * x[k];
    x[i] = s / l(i, i);
  }
  for (size_t ii = n; ii-- > 0;) {  // L^T x = y
    double s = x[ii];
    for (size_t k = ii + 1; k < n; ++k) s -= l(k, ii) * x[k];
    x[ii] = s / l(ii, ii);
  }
  return x;
}

// One-sided Jacobi SVD (Hestenes) of an r x c matrix with r >= c; transposes when r < c.
Svd svd(const Mat& m) {
  if (m.r < m.c) {
    Mat t(m.c, m.r);
    for (size_t i = 0; i < m.r; ++i)
      for (size_t j = 0; j < m.c; ++j) t(j, i) = m(i, j);
    Svd s = svd(t);
    std::swap(s.u, s.v);
    return s;
  }
  size_t r = m.r, c = m.c;
  Mat u = m;
  Mat v(c, c);
  for (size_t i = 0; i < c; ++i) v(i, i) = 1.0;
  for (int sweep = 0; sweep < 60; ++sweep) {
    double off = 0.0;
    for (size_t p = 0; p + 1 < c; ++p) {
      for (size_t q = p + 1; q < c; ++q) {
        double alpha = 0, beta = 0, gamma = 0;
        for (size_t i = 0; i < r; ++i) {
          alpha += u(i, p) * u(i, p);
          beta += u(i, q) * u(i, q);
          gamma += u(i, p) * u(i, q);
        }
        if (gamma == 0.0) continue;
        double denom = std::sqrt(alpha * beta);
        if (denom > 0.0) off = std::max(off, std::fabs(gamma) / denom);
        if (std::fabs(gamma) <= 1e-17 * denom) continue;
        double zeta = (beta - alpha) / (2.0 * gamma);
        double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1.0 + zeta * zeta));
        double cs = 1.0 / std::sqrt(1.0 + t * t), sn = cs * t;
        for (size_t i = 0; i < r; ++i) {
          double up = u(i, p), uq = u(i, q);
          u(i, p) = cs * up - sn * uq;
          u(i, q) = sn * up + cs * uq;
        }
        for (size_t i = 0; i < c; ++i) {
          double vp = v(i, p), vq = v(i, q);
          v(i, p) = cs * vp - sn * vq;
          v(i, q) = sn * vp + cs * vq;
        }
      }
    }
    if (off < 1e-15) break;
  }
  std::vector<double> sv(c);
  for (size_t j = 0; j < c; ++j) {
    double s = 0;
    for (size_t i = 0; i < r; ++i) s += u(i, j) * u(i, j);
    sv[j] = std::sqrt(s);
  }
  std::vector<size_t> order(c);
  for (size_t j = 0; j < c; ++j) order[j] = j;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return sv[a] > sv[b]; });
  Svd out;
  out.u = Mat(r, c);
  out.v = Mat(c, c);
  out.s.resize(c);
  for (size_t k = 0; k < c; ++k) {
    size_t j = order[k];
    out.s[k] = sv[j];
    for (size_t i = 0; i < r; ++i) out.u(i, k) = sv[j] > 0 ? u(i, j) / sv[j] : 0.0;
    for (size_t i = 0; i < c; ++i) out.v(i, k) = v(i, j);
  }
  return out;
}

std::optional<std::vector<double>> svd_solve(const Svd& d, const std::vector<double>& b, double eps) {
  if (!(eps >= 0.0)) return std::nullopt;
  size_t k = d.s.size();
  size_t rows = d.u.r, cols = d.v.r;
  if (b.size() != rows) return std::nullopt;
  std::vector<double> x(cols, 0.0);
  for (size_t j = 0; j < k; ++j) {
    if (d.s[j] <= eps) continue;
    double ub = 0;
    for (size_t i = 0; i < rows; ++i) ub += d.u(i, j) * b[i];
    double f = ub / d.s[j];
    for (size_t i = 0; i < cols; ++i) x[i] += d.v(i, j) * f;
  }
  return x;
}

std::optional<M2> inverse2(const M2& m) {
  double det = m[0][0] * m[1][1] - m[1][0] * m[0][1];
  if (det == 0.0) return std::nullopt;
  M2 o;
  o[0][0] = m[1][1] / det;
  o[0][1] = -m[0][1] / det;
  o[1][0] = -m[1][0] / det;
  o[1][1] = m[0][0] / det;
  return o;
}

std::optional<M3> inverse3(const M3& m) {
  double m11 = m[0][0], m12 = m[0][1], m13 = m[0][2];
  double m21 = m[1][0], m22 = m[1][1], m23 = m[1][2];
  double m31 = m[2][0], m32 = m[2][1], m33 = m[2][2];
  double minor_m12_m23 = m22 * m33 - m32 * m23;
  double minor_m11_m23 = m21 * m33 - m31 * m23;
  double minor_m11_m22 = m21 * m32 - m31 * m22;
  double det = m11 * minor_m12_m23 - m12 * minor_m11_m23 + m13 * minor_m11_m22;
  if (det == 0.0) return std::nullopt;
  M3 o;
  o[0][0] = minor_m12_m23 / det;
  o[0][1] = (m13 * m32 - m33 * m12) / det;
  o[0][2] = (m12 * m23 - m22 * m13) / det;
  o[1][0] = -minor_m11_m23 / det;
  o[1][1] = (m11 * m33 - m31 * m13) / det;
  o[1][2] = (m13 * m21 - m23 * m11) / det;
  o[2][0] = minor_m11_m22 / det;
  o[2][1] = (m12 * m31 - m32 * m11) / det;
  o[2][2] = (m11 * m22 - m21 * m12) / det;
  return o;
}

}  // namespace skyshot::linalg
