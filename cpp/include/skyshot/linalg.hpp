// Small dense linear algebra replacing the nalgebra subset used by the solver.
#pragma once
#include "common.hpp"

namespace skyshot::linalg {

// Row-major dynamic matrix.
struct Mat {
  size_t r = 0, c = 0;
  std::vector<double> a;
  Mat() = default;
  Mat(size_t r_, size_t c_) : r(r_), c(c_), a(r_ * c_, 0.0) {}
  double& operator()(size_t i, size_t j) { return a[i * c + j]; }
  double operator()(size_t i, size_t j) const { return a[i * c + j]; }
};

// Lower Cholesky factor of a symmetric positive-definite matrix (uses the lower triangle,
// like nalgebra). nullopt when the matrix is not positive definite.
std::optional<Mat> cholesky(const Mat& m);
std::vector<double> cholesky_solve(const Mat& l, const std::vector<double>& b);

struct Svd {
  Mat u;                       // r x k
  std::vector<double> s;       // k singular values, descending
  Mat v;                       // c x k   (A = U diag(s) V^T)
};
Svd svd(const Mat& m);
// Least-squares solve with singular values <= eps treated as zero (nalgebra SVD::solve).
std::optional<std::vector<double>> svd_solve(const Svd& d, const std::vector<double>& b, double eps);

using M2 = std::array<std::array<double, 2>, 2>;
using M3 = std::array<std::array<double, 3>, 3>;
std::optional<M2> inverse2(const M2& m);
std::optional<M3> inverse3(const M3& m);

}  // namespace skyshot::linalg
