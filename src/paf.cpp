// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
#include <Rcpp.h>
#include <cmath>
#include <sstream>
#include <iomanip>

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

// format like R: minimal decimals, no trailing zeros
inline std::string format_alpha(double val) {
  std::ostringstream oss;
  oss << std::setprecision(15) << val; // high precision
  std::string s = oss.str();
  // strip trailing zeros
  if (s.find('.') != std::string::npos) {
    while (!s.empty() && s.back() == '0') s.pop_back();
    if (!s.empty() && s.back() == '.') s.pop_back();
  }
  return s;
}

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

SEXP paf_cpp(NumericVector y, NumericVector a) {
  int n = y.size();
  double double_n  = static_cast<double>(n);
  double double_n2 = double_n * double_n;
  double inv_n     = 1.0 / double_n;
  double inv_n2    = 1.0 / double_n2;
  
  // normalize y into a new vector (no in-place modification)
  const double* RESTRICT yptr0 = y.begin();
  double mean_y = fast_mean(yptr0, n);
  NumericVector y_norm(n);
  for (int i = 0; i < n; i++) {
    y_norm[i] = y[i] / mean_y;
  }
  
  const double* RESTRICT yptr = y_norm.begin();
  double sd_y = fast_sd(yptr, n, 1.0);
  
  double com = 4.7 / std::sqrt(double_n) * sd_y;
  
  // alienation (independent of alpha)
  double alien = 0.0;
  for (int i = 0; i < n; i++) {
    double yi = yptr[i];
    for (int j = 0; j < n; j++) {
      alien += std::fabs(yi - yptr[j]);
    }
  }
  alien *= inv_n2;
  
  int lena = a.size();
  NumericVector paf(lena), ident(lena);
  
  for (int k = 0; k < lena; k++) {
    double alpha = a[k];
    double h = com * std::pow(alpha, 0.1);
    double h_sq = h * h;
    double inv_h_sq = 1.0 / h_sq;
    double norm_const = 1.0 / (std::sqrt(2.0 * M_PI) * h);
    
    double paf_sum = 0.0, ident_sum = 0.0;
    
    for (int i = 0; i < n; i++) {
      double yi = yptr[i];
      double fhat_sum = 1.0; // self-term
      double row_sum  = 0.0;
      
      for (int j = 0; j < n; j++) {
        if (i == j) continue;
        double d = yi - yptr[j];
        double k_val = std::exp(-0.5 * (d * d) * inv_h_sq);
        fhat_sum += k_val;
        row_sum  += std::fabs(d);
      }
      
      double fhat_i = fhat_sum * norm_const * inv_n;
      double fhata_i = std::pow(fhat_i, alpha);
      
      ident_sum += fhata_i;
      paf_sum   += fhata_i * row_sum;
    }
    
    ident[k] = ident_sum * inv_n;
    paf[k]   = paf_sum   * inv_n2;
  }
  
  NumericVector rho(lena);
  for (int k = 0; k < lena; k++) {
    rho[k] = paf[k] / (alien * ident[k]) - 1.0;
  }
  
  if (lena == 1) {
    return NumericVector::create(
      _["paf"] = paf[0],
                    _["alienation"] = alien,
                    _["identification"] = ident[0],
                                               _["1 + rho"] = 1.0 + rho[0]
    );
  } else {
    NumericMatrix res(lena, 4);
    for (int k = 0; k < lena; k++) {
      res(k,0) = paf[k];
      res(k,1) = alien;
      res(k,2) = ident[k];
      res(k,3) = 1.0 + rho[k];
    }
    colnames(res) = CharacterVector({"paf","alienation","identification","1 + rho"});
    CharacterVector rn(lena);
    for (int k = 0; k < lena; k++) {
      std::ostringstream oss;
      oss << "alpha=" << std::setprecision(15) << a[k];
      std::string s = oss.str();
      if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
      }
      rn[k] = s;
    }
    rownames(res) = rn;
    return res;
  }
}

// // For a strange reason, this is slower, despite the micro-optimizations...
// // [[Rcpp::export]]
// SEXP paf_cpp(NumericVector y, NumericVector a) {
//   int n = y.size();
//   double double_n  = static_cast<double>(n);
//   double double_n2 = double_n * double_n;
//   double inv_n     = 1.0 / double_n;
//   double inv_n2    = 1.0 / double_n2;
//   
//   // normalize y into a new vector (no in-place modification)
//   const double* RESTRICT yptr0 = y.begin();
//   double mean_y = fast_mean(yptr0, n);
//   NumericVector y_norm(n);
//   for (int i = 0; i < n; i++) {
//     y_norm[i] = y[i] / mean_y;
//   }
//   
//   const double* RESTRICT yptr = y_norm.begin();
//   double sd_y = fast_sd(yptr, n, 1.0);
//   
//   double com = 4.7 / std::sqrt(double_n) * sd_y;
//   
//   // alienation (independent of alpha)
//   double alien = 0.0;
//   for (int i = 0; i < n; i++) {
//     double yi = yptr[i];
//     for (int j = 0; j < n; j++) {
//       alien += std::fabs(yi - yptr[j]);
//     }
//   }
//   alien *= inv_n2;
//   
//   int lena = a.size();
//   NumericVector paf(lena), ident(lena);
//   
//   for (int k = 0; k < lena; k++) {
//     double alpha = a[k];
//     double h = com * std::pow(alpha, 0.1);
//     double h_sq = h * h;
//     double inv_h_sq = 1.0 / h_sq;
//     double norm_const = 1.0 / (std::sqrt(2.0 * M_PI) * h);
//     
//     double paf_sum = 0.0, ident_sum = 0.0;
//     
//     for (int i = 0; i < n; i++) {
//       double yi = yptr[i];
//       double fhat_sum = 1.0; // self-term
//       double row_sum  = 0.0;
//       
//       for (int j = 0; j < n; j++) {
//         if (i == j) continue;
//         double d = yi - yptr[j];
//         double k_val = std::exp(-0.5 * (d * d) * inv_h_sq);
//         fhat_sum += k_val;
//         row_sum  += std::fabs(d);
//       }
//       
//       double fhat_i = fhat_sum * norm_const * inv_n;
//       double fhata_i = std::pow(fhat_i, alpha);
//       
//       ident_sum += fhata_i;
//       paf_sum   += fhata_i * row_sum;
//     }
//     
//     ident[k] = ident_sum * inv_n;
//     paf[k]   = paf_sum   * inv_n2;
//   }
//   
//   NumericVector rho(lena);
//   for (int k = 0; k < lena; k++) {
//     rho[k] = paf[k] / (alien * ident[k]) - 1.0;
//   }
//   
//   if (lena == 1) {
//     return NumericVector::create(
//       _["paf"] = paf[0],
//                     _["alienation"] = alien,
//                     _["identification"] = ident[0],
//                                                _["1 + rho"] = 1.0 + rho[0]
//     );
//   } else {
//     NumericMatrix res(lena, 4);
//     for (int k = 0; k < lena; k++) {
//       res(k,0) = paf[k];
//       res(k,1) = alien;
//       res(k,2) = ident[k];
//       res(k,3) = 1.0 + rho[k];
//     }
//     colnames(res) = CharacterVector({"paf","alienation","identification","1 + rho"});
//     CharacterVector rn(lena);
//     for (int k = 0; k < lena; k++) {
//       std::ostringstream oss;
//       oss << "alpha=" << std::setprecision(15) << a[k];
//       std::string s = oss.str();
//       if (s.find('.') != std::string::npos) {
//         while (!s.empty() && s.back() == '0') s.pop_back();
//         if (!s.empty() && s.back() == '.') s.pop_back();
//       }
//       rn[k] = s;
//     }
//     rownames(res) = rn;
//     return res;
//   }
// }

// ---- Parallel mean worker and function ----
struct MeanWorker : public RcppParallel::Worker {
  const RVector<double> y;
  double sum;
  
  MeanWorker(const NumericVector& y) : y(y), sum(0.0) {}
  MeanWorker(const MeanWorker& other, RcppParallel::Split) : y(other.y), sum(0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double* RESTRICT ptr = y.begin();
    double local_sum = 0.0;
    for (std::size_t i = begin; i < end; i++) {
      local_sum += ptr[i];
    }
    sum += local_sum;
  }
  
  void join(const MeanWorker& rhs) { sum += rhs.sum; }
};

inline double parallel_mean(const NumericVector& y, int ncores) {
  MeanWorker worker(y);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelReduce(0, y.size(), worker);
  return worker.sum / static_cast<double>(y.size());
}

// ---- Parallel variance (second pass) worker and function ----
struct VarWorker : public RcppParallel::Worker {
  const RVector<double> y;
  const double mean;
  double ssq;
  
  VarWorker(const NumericVector& y, double mean) : y(y), mean(mean), ssq(0.0) {}
  VarWorker(const VarWorker& other, RcppParallel::Split) : y(other.y), mean(other.mean), ssq(0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double* RESTRICT ptr = y.begin();
    double local_ssq = 0.0;
    for (std::size_t i = begin; i < end; i++) {
      double diff = ptr[i] - mean;
      local_ssq += diff * diff;
    }
    ssq += local_ssq;
  }
  
  void join(const VarWorker& rhs) { ssq += rhs.ssq; }
};

inline double parallel_sd(const NumericVector& y, double mean, int ncores) {
  VarWorker worker(y, mean);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelReduce(0, y.size(), worker);
  return std::sqrt(worker.ssq / (static_cast<double>(y.size()) - 1.0));
}

// ---- Worker for one alpha (tight inner loop) ----
struct KernelPAFWorker : public Worker {
  const RVector<double> y;
  const double a;
  const double inv_h_sq;
  const double norm_const;
  const double inv_n;
  
  double ident_sum;
  double alien_sum;
  double paf_sum;
  
  KernelPAFWorker(const NumericVector& y, double a,
                  double inv_h_sq, double norm_const, double inv_n)
    : y(y), a(a), inv_h_sq(inv_h_sq), norm_const(norm_const), inv_n(inv_n),
      ident_sum(0.0), alien_sum(0.0), paf_sum(0.0) {}
  
  KernelPAFWorker(const KernelPAFWorker& other, Split)
    : y(other.y), a(other.a), inv_h_sq(other.inv_h_sq),
      norm_const(other.norm_const), inv_n(other.inv_n),
      ident_sum(0.0), alien_sum(0.0), paf_sum(0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    std::size_t n = y.length();
    const double * RESTRICT yptr = y.begin();
    
    for (std::size_t i = begin; i < end; i++) {
      double yi = yptr[i];
      double fhat_sum = 1.0;  // self-term, avoids branch for j==i
      double row_sum  = 0.0;
      
      for (std::size_t j = 0; j < n; j++) {
        if (i == j) continue;
        double d = yi - yptr[j];
        double k = std::exp(-0.5 * (d * d) * inv_h_sq);
        fhat_sum += k;
        row_sum  += std::fabs(d);
      }
      
      double fhat_i  = fhat_sum * norm_const * inv_n;
      double fhata_i = std::pow(fhat_i, a);
      
      ident_sum += fhata_i;
      alien_sum += row_sum;
      paf_sum   += fhata_i * row_sum;
    }
  }
  
  void join(const KernelPAFWorker& rhs) {
    ident_sum += rhs.ident_sum;
    alien_sum += rhs.alien_sum;
    paf_sum   += rhs.paf_sum;
  }
};

// ---- Alienation worker ----
struct AlienWorker : public RcppParallel::Worker {
  const RVector<double> y;
  double sum;
  
  AlienWorker(const NumericVector& y) : y(y), sum(0.0) {}
  AlienWorker(const AlienWorker& other, RcppParallel::Split) : y(other.y), sum(0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const double* RESTRICT yptr = y.begin();
    std::size_t n = y.length();
    double local_sum = 0.0;
    for (std::size_t i = begin; i < end; i++) {
      double yi = yptr[i];
      for (std::size_t j = 0; j < n; j++) {
        local_sum += std::fabs(yi - yptr[j]);
      }
    }
    sum += local_sum;
  }
  
  void join(const AlienWorker& rhs) { sum += rhs.sum; }
};

inline double parallel_alien(const NumericVector& y, int ncores) {
  AlienWorker worker(y);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelReduce(0, y.size(), worker);
  double double_n = static_cast<double>(y.size());
  double inv_n2   = 1.0 / (double_n * double_n);
  return worker.sum * inv_n2;
}


SEXP paf_parallel(NumericVector y, NumericVector a, int ncores = 1) {
  int n = y.size();
  double double_n  = static_cast<double>(n);
  double double_n2 = double_n * double_n;
  double inv_n     = 1.0 / double_n;
  double inv_n2    = 1.0 / double_n2;
  
  // normalize y (parallel mean)
  NumericVector y_norm = clone(y);
  double mean_y = parallel_mean(y_norm, ncores);
  for (int i = 0; i < n; i++) y_norm[i] /= mean_y;
  
  // sd of normalized y_norm has mean 1.0
  double sd_y = parallel_sd(y_norm, 1.0, ncores);
  double com  = 4.7 / std::sqrt(double_n) * sd_y;
  
  // alienation (independent of alpha)
  double alien = parallel_alien(y_norm, ncores);
  
  int lena = a.size();
  NumericVector paf(lena), ident(lena);
  
  for (int k = 0; k < lena; k++) {
    double alpha = a[k];
    double h = com * std::pow(alpha, 0.1);
    double h_sq = h * h;
    double inv_h_sq = 1.0 / h_sq;
    double norm_const = 1.0 / (std::sqrt(2.0 * M_PI) * h);
    
    KernelPAFWorker worker(y_norm, alpha, inv_h_sq, norm_const, inv_n);
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
    parallelReduce(0, n, worker);
    
    ident[k] = worker.ident_sum * inv_n;
    paf[k]   = worker.paf_sum   * inv_n2;
  }
  
  NumericVector rho(lena);
  for (int k = 0; k < lena; k++) {
    rho[k] = paf[k] / (alien * ident[k]) - 1.0;
  }
  
  if (lena == 1) {
    NumericVector res = NumericVector::create(
      _["paf"]            = paf[0],
      _["alienation"]     = alien,
      _["identification"] = ident[0],
      _["1 + rho"]        = 1.0 + rho[0]
    );
    return res;
  } else {
    NumericMatrix res(lena, 4);
    for (int k = 0; k < lena; k++) {
      res(k,0) = paf[k];
      res(k,1) = alien;
      res(k,2) = ident[k];
      res(k,3) = 1.0 + rho[k];
    }
    colnames(res) = CharacterVector({"paf","alienation","identification","1 + rho"});
    CharacterVector rn(lena);
    for (int k = 0; k < lena; k++) {
      rn[k] = "alpha=" + format_alpha(a[k]);
    }
    rownames(res) = rn;
    return res;
  }
}

// [[Rcpp::export]]
SEXP paf(const NumericVector y, const NumericVector a, int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1){
    return paf_parallel(y, a, ncores);
  } else {
    return paf_cpp(y, a);
  }
#else
  return paf_cpp(y, a);
#endif
}