// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <Rcpp.h>
#include <cmath>

// Define a RESTRICT macro for portability.
#if defined(__GNUG__) || defined(__clang__)
#  define RESTRICT __restrict__
#else
#  define RESTRICT __restrict
#endif

using namespace Rcpp;
using namespace RcppParallel;

#ifdef RCPP_PARALLEL_USE_TBB
#include <tbb/global_control.h>  // For controlling the number of threads
#endif

// ---- Fast mean ----
inline double fast_mean(const double* RESTRICT ptr, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += ptr[i];
  }
  return sum / static_cast<double>(n);
}

// ---- Fast standard deviation ----
// inline double fast_sd(const NumericVector &x) {
//   int n = x.size();
//   const double * RESTRICT ptr = x.begin();
//   double mean = 0.0, M2 = 0.0;
//   int count = 0;
//   for (int i = 0; i < n; i++) {
//     count++;
//     double delta = ptr[i] - mean;
//     mean += delta / count;
//     M2 += delta * (ptr[i] - mean);
//   }
//   return std::sqrt(M2 / (count - 1));
// }

inline double fast_sd(const double* RESTRICT yp, int n, double mean) {
  double sumsq = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = yp[i] - mean;
    sumsq += diff * diff;
  }
  return std::sqrt(sumsq / (n - 1));
}

NumericMatrix colpafs_cpp(const NumericMatrix& Y, double a) {
  int n = Y.nrow();
  int p = Y.ncol();
  
  NumericMatrix res(p, 4);
  colnames(res) = CharacterVector({"paf","alienation","identification","1 + rho"});
  CharacterVector rn = colnames(Y);
  if (rn.size() == p) rownames(res) = rn;
  
  double double_n  = static_cast<double>(n);
  double double_n2 = double_n * double_n;
  double inv_n     = 1.0 / double_n;
  double inv_n2    = 1.0 / double_n2;
  double cons_sqrt_n = 4.7 / std::sqrt(double_n);
  double sqrt_2pi    = std::sqrt(2.0 * M_PI);
  
  for (int col = 0; col < p; col++) {
    // make a copy of the column
    NumericVector y = Y(_, col);
    double * RESTRICT yptr = y.begin();
    
    // mean
    // double mean_y = 0.0;
    // for (int i = 0; i < n; i++) mean_y += yptr[i];
    // mean_y /= n;
    double mean_y = fast_mean(yptr, n);
    
    // normalize the copy
    for (int i = 0; i < n; i++) yptr[i] /= mean_y;
    
    // compute sd in one pass
    // double mean = 0.0, M2 = 0.0;
    // for (int i = 0; i < n; i++) {
    //   double delta = yptr[i] - mean;
    //   mean += delta / (i+1);
    //   M2 += delta * (yptr[i] - mean);
    // }
    // double sd_y = std::sqrt(M2 / (n-1));
    
    double sd_y = fast_sd(yptr, n, 1.0);
    
    double com = cons_sqrt_n * sd_y;
    double h = com * std::pow(a, 0.1);
    double inv_h_sq = 1.0 / (h*h);
    double norm_const = 1.0 / (sqrt_2pi * h);
    
    double alien = 0.0, paf_sum = 0.0, ident_sum = 0.0;
    
    for (int i = 0; i < n; i++) {
      double yi = yptr[i];
      double fhat_sum = 1.0;
      double row_sum  = 0.0;
      
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        double d = yi - yptr[j];
        double k_val = std::exp(-0.5 * d*d * inv_h_sq);
        fhat_sum += k_val;
        row_sum  += std::fabs(d);
        alien    += std::fabs(d);
      }
      
      double fhat_i = fhat_sum * norm_const * inv_n;
      double fhata_i = std::pow(fhat_i, a);
      
      ident_sum += fhata_i;
      paf_sum   += fhata_i * row_sum;
    }
    
    alien *= inv_n2;
    double ident = ident_sum * inv_n;
    double paf   = paf_sum   * inv_n2;
    double rho   = paf / (alien * ident) - 1.0;
    
    res(col,0) = paf;
    res(col,1) = alien;
    res(col,2) = ident;
    res(col,3) = 1.0 + rho;
  }
  
  return res;
}


// Worker struct
struct ColPafWorker : public Worker {
  const RMatrix<double> Y;
  double a;
  int n;
  double double_n, double_n2, inv_n, inv_n2, cons_sqrt_n, sqrt_2pi;
  
  RMatrix<double> res;
  
  ColPafWorker(const NumericMatrix& Y, double a, NumericMatrix& res)
    : Y(Y), a(a), n(Y.nrow()), 
      double_n(static_cast<double>(Y.nrow())),
      double_n2(double_n * double_n),
      inv_n(1.0 / double_n),
      inv_n2(1.0 / double_n2),
      cons_sqrt_n(4.7 / std::sqrt(double_n)),
      sqrt_2pi(std::sqrt(2.0 * M_PI)),
      res(res) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t col = begin; col < end; col++) {
      // copy column
      std::vector<double> y(n);
      for (int i = 0; i < n; i++) y[i] = Y(i,col);
      double* RESTRICT yptr = y.data();
      
      // mean
      double mean_y = fast_mean(yptr, n);
      
      // normalize
      for (int i = 0; i < n; i++) yptr[i] /= mean_y;
      
      // sd (mean of normalized y is 1.0)
      double sd_y = fast_sd(yptr, n, 1.0);
      
      double com = cons_sqrt_n * sd_y;
      double h = com * std::pow(a, 0.1);
      double inv_h_sq = 1.0 / (h*h);
      double norm_const = 1.0 / (sqrt_2pi * h);
      
      double alien = 0.0, paf_sum = 0.0, ident_sum = 0.0;
      
      for (int i = 0; i < n; i++) {
        double yi = yptr[i];
        double fhat_sum = 1.0;
        double row_sum  = 0.0;
        
        for (int j = 0; j < n; j++) {
          if (i == j) continue;
          double d = yi - yptr[j];
          double k_val = std::exp(-0.5 * d*d * inv_h_sq);
          fhat_sum += k_val;
          row_sum  += std::fabs(d);
          alien    += std::fabs(d);
        }
        
        double fhat_i = fhat_sum * norm_const * inv_n;
        double fhata_i = std::pow(fhat_i, a);
        
        ident_sum += fhata_i;
        paf_sum   += fhata_i * row_sum;
      }
      
      alien *= inv_n2;
      double ident = ident_sum * inv_n;
      double paf   = paf_sum   * inv_n2;
      double rho   = paf / (alien * ident) - 1.0;
      
      res(col,0) = paf;
      res(col,1) = alien;
      res(col,2) = ident;
      res(col,3) = 1.0 + rho;
    }
  }
};

NumericMatrix colpafs_parallel(const NumericMatrix& Y, double a, int ncores = 1) {
  int p = Y.ncol();
  NumericMatrix res(p, 4);
  colnames(res) = CharacterVector({"paf","alienation","identification","1 + rho"});
  CharacterVector rn = colnames(Y);
  if (rn.size() == p) rownames(res) = rn;
  
  ColPafWorker worker(Y, a, res);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelFor(0, p, worker);
  
  return res;
}

// [[Rcpp::export]]
SEXP colpafs(const NumericMatrix& y, double a, int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1){
    return colpafs_parallel(y, a, ncores);
  } else {
    return colpafs_cpp(y, a);
  }
#else
  return colpafs_cpp(y, a);
#endif
}