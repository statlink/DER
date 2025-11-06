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

#ifdef RCPP_PARALLEL_USE_TBB
#include <tbb/global_control.h>
#endif

// ---- Fast mean ----
inline double fast_mean(const double* RESTRICT ptr, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) sum += ptr[i];
  return sum / static_cast<double>(n);
}

// ---- Fast sd ----
inline double fast_sd(const double* RESTRICT yp, int n, double mean) {
  double sumsq = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = yp[i] - mean;
    sumsq += diff * diff;
  }
  return std::sqrt(sumsq / (n - 1));
}

// ---- Serial version ----
NumericMatrix colpafs2_cpp(const NumericMatrix& Y, double a) {
  int n = Y.nrow();
  int p = Y.ncol();
  
  NumericMatrix res(p, 3);
  colnames(res) = CharacterVector({"paf","deprivation","surplus"});
  CharacterVector rn = colnames(Y);
  if (rn.size() == p) rownames(res) = rn;
  
  double cons_sqrt_n = 4.7 / std::sqrt((double)n);
  double sqrt_2pi    = std::sqrt(2.0 * M_PI);
  double inv_n       = 1.0 / (double)n;
  double inv_n2      = 1.0 / ((double)n * (double)n);
  
  for (int col = 0; col < p; col++) {
    NumericVector y = Y(_, col);
    double* RESTRICT yptr = y.begin();
    
    double mean_y = fast_mean(yptr, n);
    for (int i = 0; i < n; i++) yptr[i] /= mean_y;
    
    double sd_y = fast_sd(yptr, n, 1.0);
    double h = cons_sqrt_n * sd_y * std::pow(a, 0.1);
    double inv_h_sq = 1.0 / (h*h);
    double norm_const = 1.0 / (sqrt_2pi * h);
    
    double D = 0.0, S = 0.0;
    
    for (int i = 0; i < n; i++) {
      double yi = yptr[i];
      double fhat_sum = 0.0;
      for (int j = 0; j < n; j++) {
        double d = yi - yptr[j];
        fhat_sum += std::exp(-0.5 * d*d * inv_h_sq);
      }
      double fhat_i  = fhat_sum * norm_const * inv_n;
      double fhata_i = std::pow(fhat_i, a);
      
      for (int j = 0; j < n; j++) {
        double d = yi - yptr[j];
        if (d > 0) S += fhata_i * d;
        else if (d < 0) D += fhata_i * (-d);
      }
    }
    
    D *= inv_n2;
    S *= inv_n2;
    
    res(col,0) = D + S;
    res(col,1) = D;
    res(col,2) = S;
  }
  
  return res;
}

// ---- Parallel worker ----
struct ColPaf2Worker : public Worker {
  const RMatrix<double> Y;
  double a;
  int n;
  double cons_sqrt_n, sqrt_2pi, inv_n, inv_n2;
  RMatrix<double> res;
  
  ColPaf2Worker(const NumericMatrix& Y, double a, NumericMatrix& res)
    : Y(Y), a(a), n(Y.nrow()),
      cons_sqrt_n(4.7 / std::sqrt((double)Y.nrow())),
      sqrt_2pi(std::sqrt(2.0 * M_PI)),
      inv_n(1.0 / (double)Y.nrow()),
      inv_n2(1.0 / ((double)Y.nrow() * (double)Y.nrow())),
      res(res) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t col = begin; col < end; col++) {
      std::vector<double> y(n);
      for (int i = 0; i < n; i++) y[i] = Y(i,col);
      double* RESTRICT yptr = y.data();
      
      double mean_y = fast_mean(yptr, n);
      for (int i = 0; i < n; i++) yptr[i] /= mean_y;
      
      double sd_y = fast_sd(yptr, n, 1.0);
      double h = cons_sqrt_n * sd_y * std::pow(a, 0.1);
      double inv_h_sq = 1.0 / (h*h);
      double norm_const = 1.0 / (sqrt_2pi * h);
      
      double D = 0.0, S = 0.0;
      
      for (int i = 0; i < n; i++) {
        double yi = yptr[i];
        double fhat_sum = 0.0;
        for (int j = 0; j < n; j++) {
          double d = yi - yptr[j];
          fhat_sum += std::exp(-0.5 * d*d * inv_h_sq);
        }
        double fhat_i  = fhat_sum * norm_const * inv_n;
        double fhata_i = std::pow(fhat_i, a);
        
        for (int j = 0; j < n; j++) {
          double d = yi - yptr[j];
          if (d > 0) S += fhata_i * d;
          else if (d < 0) D += fhata_i * (-d);
        }
      }
      
      D *= inv_n2;
      S *= inv_n2;
      
      res(col,0) = D + S;
      res(col,1) = D;
      res(col,2) = S;
    }
  }
};

// ---- Parallel wrapper ----
NumericMatrix colpafs2_parallel(const NumericMatrix& Y, double a, int ncores = 1) {
  int p = Y.ncol();
  NumericMatrix res(p, 3);
  colnames(res) = CharacterVector({"paf","deprivation","surplus"});
  CharacterVector rn = colnames(Y);
  if (rn.size() == p) rownames(res) = rn;
  
  ColPaf2Worker worker(Y, a, res);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelFor(0, p, worker);
  
  return res;
}

// ---- Dispatcher ----
// [[Rcpp::export]]
SEXP colpafs2(const NumericMatrix& y, double a, int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1) {
    return colpafs2_parallel(y, a, ncores);
  } else {
    return colpafs2_cpp(y, a);
  }
#else
  return colpafs2_cpp(y, a);
#endif
}
