// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <Rcpp.h>
#include <cmath>

#if defined(__GNUG__) || defined(__clang__)
#  define RESTRICT __restrict__
#else
#  define RESTRICT __restrict
#endif

using namespace Rcpp;
using namespace RcppParallel;

struct PolarWorker : public Worker {
  const RVector<double> y;
  const RVector<double> prefix; // precomputed prefix sums
  const double h, inv_nh, sqrt_2pi, m, two_over_n, alpha;
  RVector<double> contrib;
  
  PolarWorker(const NumericVector& y,
              const NumericVector& prefix,
              double h, double inv_nh, double sqrt_2pi,
              double m, double two_over_n, double alpha,
              NumericVector& contrib)
    : y(y), prefix(prefix),
      h(h), inv_nh(inv_nh), sqrt_2pi(sqrt_2pi),
      m(m), two_over_n(two_over_n), alpha(alpha),
      contrib(contrib) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double* RESTRICT yptr   = y.begin();
    const double* RESTRICT preptr = prefix.begin();
    double* RESTRICT outptr       = contrib.begin();
    int n = y.length();
    
    for (std::size_t i = begin; i < end; i++) {
      double yi = yptr[i];
      // kernel density at yi
      double sumk = 0.0;
      for (int j = 0; j < n; j++) {
        double z = (yi - yptr[j]) / h;
        sumk += std::exp(-0.5 * z * z) / sqrt_2pi;
      }
      double fhat = sumk * inv_nh;
      // ayi_i using prefix sum
      double ayi_i = m + (two_over_n * (i+1) - 1.0) * yi - two_over_n * preptr[i];
      outptr[i] = std::pow(fhat, alpha) * ayi_i;
    }
  }
};

NumericVector polarization_parallel(NumericVector y, NumericVector a, int ncores = 1) {
  int n = y.size();
  const double* RESTRICT yptr = y.begin();
  
  // mean
  double sum_y = 0.0;
  for (int i = 0; i < n; i++) sum_y += yptr[i];
  double m = sum_y / n;
  
  // variance
  double ssq = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = yptr[i] - m;
    ssq += diff * diff;
  }
  double sd_y = std::sqrt(ssq / (n - 1));
  
  // prefix sums for ayi
  NumericVector prefix(n);
  double cumsum = 0.0;
  for (int i = 0; i < n; i++) {
    cumsum += yptr[i];
    prefix[i] = cumsum;
  }
  
  int la = a.size();
  NumericVector est(la);
  double sqrt_2pi = std::sqrt(2.0 * M_PI);
  double two_over_n = 2.0 / n;
  
  for (int k = 0; k < la; k++) {
    double h = 4.7 / std::sqrt((double)n) * sd_y * std::pow(a[k], 0.1);
    double inv_nh = 1.0 / (n * h);
    
    NumericVector contrib(n);
    PolarWorker worker(y, prefix, h, inv_nh, sqrt_2pi, m, two_over_n, a[k], contrib);
    
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
    parallelFor(0, n, worker);
    
    double sum_est = std::accumulate(contrib.begin(), contrib.end(), 0.0);
    est[k] = (sum_est / n) / std::pow(m, 1.0 - a[k]);
  }
  
  return est;
}

NumericVector polarization_cpp(const NumericVector y, const NumericVector a) {
  int n = y.size();
  const double* RESTRICT yptr = y.begin();   // raw pointer to y
  const double* RESTRICT aptr = a.begin();   // raw pointer to a
  
  // mean
  double sum_y = 0.0;
  for (int i = 0; i < n; i++) sum_y += yptr[i];
  double m = sum_y / n;
  
  // variance for bandwidth
  double ssq = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = yptr[i] - m;
    ssq += diff * diff;
  }
  double sd_y = std::sqrt(ssq / (n - 1));
  
  int la = a.size();
  NumericVector est(la);
  double sqrt_2pi = std::sqrt(2.0 * M_PI);
  double two_over_n = 2.0 / n;
  
  if (la == 1) {
    double h = 4.7 / std::sqrt((double)n) * sd_y * std::pow(aptr[0], 0.1);
    double one_over_nh = 1 / (n * h);
    
    // accumulate polarization index using on-the-fly ayi
    double cumsum_y = 0.0;
    double sum_est = 0.0;
    for (int i = 0; i < n; i++) {
      double fhat = 0.0;
      for (int j = 0; j < n; j++) {
        double z = (yptr[i] - yptr[j]) / h;
        fhat += std::exp(-0.5 * z * z) / sqrt_2pi;
      }
      fhat *= one_over_nh;
      cumsum_y += yptr[i];
      sum_est += std::pow(fhat, aptr[0]) *
        (m + (two_over_n * (i+1) - 1.0) * yptr[i] - two_over_n * cumsum_y);
    }
    est[0] = sum_est / n;
  } else {
    double com = 4.7 / std::sqrt((double)n) * sd_y;
    double two_over_n = 2.0 / n;
    
    for (int k = 0; k < la; k++) {
      double h = com * std::pow(aptr[k], 0.1);
      double one_over_nh = 1 / (n * h);
      
      double cumsum_y = 0.0;
      double sum_est = 0.0;
      for (int i = 0; i < n; i++) {
        double fhat = 0.0;
        for (int j = 0; j < n; j++) {
          double z = (yptr[i] - yptr[j]) / h;
          fhat += std::exp(-0.5 * z * z) / sqrt_2pi;
        }
        fhat *= one_over_nh;
        cumsum_y += yptr[i];
        sum_est += std::pow(fhat, aptr[k]) *
          (m + (two_over_n * (i+1) - 1.0) * yptr[i] - two_over_n * cumsum_y);
      }
      est[k] = sum_est / n;
    }
  }
  
  // final scaling
  for (int k = 0; k < la; k++) {
    est[k] /= std::pow(m, 1.0 - aptr[k]);
  }
  
  return est;
}

// [[Rcpp::export]]
SEXP pafF_helper(const NumericVector y, const NumericVector a, int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1){
    return polarization_parallel(y, a, ncores);
  } else {
    return polarization_cpp(y, a);
  }
#else
  return polarization_cpp(y, a);
#endif
}
