// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
#include <cmath>

// Define a RESTRICT macro for portability.
#if defined(__GNUG__) || defined(__clang__)
#  define RESTRICT __restrict__
#else
#  define RESTRICT __restrict
#endif

using namespace Rcpp;
using namespace RcppParallel;

// -------------------- Serial version --------------------
NumericVector kde_cpp(const NumericVector y) {
  int n = y.size();
  const double* RESTRICT yptr = y.begin();
  
  // mean
  double sum_y = 0.0;
  for (int i = 0; i < n; i++) sum_y += yptr[i];
  double mean_y = sum_y / n;
  
  // variance
  double ssq = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = yptr[i] - mean_y;
    ssq += diff * diff;
  }
  double sd_y = std::sqrt(ssq / (n - 1));
  
  // bandwidth (Silverman's rule)
  double h = 1.06 * sd_y * std::pow(n, -0.2);
  
  NumericVector est(n);
  double sqrt_2pi = std::sqrt(2.0 * M_PI);
  double norm_const = 1.0 / (n * h * sqrt_2pi);
  
  for (int i = 0; i < n; i++) {
    double sum_kernel = 0.0;
    double yi = yptr[i];
    for (int j = 0; j < n; j++) {
      double z = (yi - yptr[j]) / h;
      sum_kernel += std::exp(-0.5 * z * z);
    }
    est[i] = norm_const * sum_kernel;
  }
  
  return est;
}

// -------------------- Parallel version --------------------

// Worker for parallelReduce (sum and sumsq)
struct AccumWorker : public Worker {
  const RVector<double> y;
  double sum;
  double sumsq;
  
  // Normal constructor
  AccumWorker(const NumericVector& y)
    : y(y), sum(0.0), sumsq(0.0) {}
  
  // Split constructor
  AccumWorker(const AccumWorker& other, Split)
    : y(other.y), sum(0.0), sumsq(0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    const double* RESTRICT yptr = y.begin();
    for (std::size_t i = begin; i < end; ++i) {
      double v = yptr[i];
      local_sum   += v;
      local_sumsq += v * v;
    }
    sum   += local_sum;
    sumsq += local_sumsq;
  }
  
  void join(const AccumWorker& rhs) {
    sum   += rhs.sum;
    sumsq += rhs.sumsq;
  }
};

// Worker for parallel KDE evaluation
struct KDEWorker : public Worker {
  const RVector<double> y;
  int n;
  double h;
  double norm_const;
  RVector<double> est;
  
  KDEWorker(const NumericVector& y, double h, double norm_const, NumericVector& est)
    : y(y), n(y.length()), h(h), norm_const(norm_const), est(est) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double* RESTRICT yptr = y.begin();
    for (std::size_t i = begin; i < end; i++) {
      double yi = yptr[i];
      double sum_kernel = 0.0;
      for (int j = 0; j < n; j++) {
        double z = (yi - yptr[j]) / h;
        sum_kernel += std::exp(-0.5 * z * z);
      }
      est[i] = norm_const * sum_kernel;
    }
  }
};

NumericVector kde_parallel(const NumericVector y, const int ncores = 1) {
  int n = y.size();
  if (n < 2) stop("y must have length >= 2");
  double double_n = static_cast<double>(n);
  // Parallel reduce for mean and variance
  AccumWorker accum(y);
  parallelReduce(0, n, accum, ncores);
  
  double mean_y = accum.sum / double_n;
  double var_y  = (accum.sumsq - double_n * mean_y * mean_y) / (n - 1);
  double sd_y   = std::sqrt(var_y);
  
  // Bandwidth
  double h = 1.06 * sd_y * std::pow(double_n, -0.2);
  
  NumericVector est(n);
  double sqrt_2pi = std::sqrt(2.0 * M_PI);
  double norm_const = 1.0 / (n * h * sqrt_2pi);
  
  // Parallel KDE evaluation
  KDEWorker worker(y, h, norm_const, est);
  parallelFor(0, n, worker, ncores);
  
  return est;
}

// [[Rcpp::export]]
NumericVector kde(const NumericVector y, const int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1){
    return kde_parallel(y, ncores);
  } else {
    return kde_cpp(y);
  }
#else
  return kde_cpp(y);
#endif
}
