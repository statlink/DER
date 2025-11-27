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

// ---- helpers ----
inline double fast_mean(const double* RESTRICT ptr, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) sum += ptr[i];
  return sum / static_cast<double>(n);
}

inline double fast_sd(const double* RESTRICT ptr, int n, double mean) {
  if (n <= 1) return NA_REAL;
  double sumsq = 0.0;
  for (int i = 0; i < n; i++) {
    double diff = ptr[i] - mean;
    sumsq += diff * diff;
  }
  return std::sqrt(sumsq / (n - 1));
}

SEXP paf2_cpp(NumericVector y, NumericVector a) {
  int n = y.size();
  double double_n  = static_cast<double>(n);
  double double_n2 = double_n * double_n;
  double inv_n     = 1.0 / double_n;
  double inv_n2    = 1.0 / double_n2;
  
  // normalize
  const double* RESTRICT yptr0 = y.begin();
  double mean_y = fast_mean(yptr0, n);
  NumericVector y_norm(n);
  for (int i = 0; i < n; i++) y_norm[i] = y[i] / mean_y;
  
  const double* RESTRICT yptr = y_norm.begin();
  double sd_y = fast_sd(yptr, n, 1.0);
  
  int lena = a.size();
  
  if (lena == 1) {
    double alpha = a[0];
    double h = 4.7 / std::sqrt(double_n) * sd_y * std::pow(alpha, 0.1);
    double inv_h = 1.0 / h;
    double inv_h_sq = inv_h * inv_h;
    double norm_const = 1.0 / (std::sqrt(2.0 * M_PI) * h);
    
    NumericVector fhat(n);
    for (int i = 0; i < n; i++) {
      double yi = yptr[i];
      double sumexp = 0.0;
      for (int j = 0; j < n; j++) {
        double d = yi - yptr[j];
        sumexp += std::exp(-0.5 * d * d * inv_h_sq);
      }
      fhat[i] = sumexp * inv_n * norm_const;
    }
    
    NumericVector fhata(n);
    for (int i = 0; i < n; i++) fhata[i] = std::pow(fhat[i], alpha);
    
    double D = 0.0, S = 0.0;
    for (int i = 0; i < n; i++) {
      double yi = yptr[i];
      for (int j = 0; j < n; j++) {
        double d = yi - yptr[j];
        if (d > 0) {
          S += fhata[i] * d;
        } else if (d < 0) {
          D += fhata[i] * (-d);
        }
      }
    }
    D *= inv_n2;
    S *= inv_n2;
    
    return NumericVector::create(
      _["paf"] = D + S,
      _["deprivation"] = D,
      _["surplus"] = S
    );
    
  } else {
    NumericMatrix res(lena, 3); // output only
    CharacterVector rn(lena);
    
    double com = 4.7 / std::sqrt(double_n) * sd_y;
    
    for (int k = 0; k < lena; k++) {
      double alpha = a[k];
      double h = com * std::pow(alpha, 0.1);
      double inv_h = 1.0 / h;
      double inv_h_sq = inv_h * inv_h;
      double norm_const = 1.0 / (std::sqrt(2.0 * M_PI) * h);
      
      NumericVector fhat(n);
      for (int i = 0; i < n; i++) {
        double yi = yptr[i];
        double sumexp = 0.0;
        for (int j = 0; j < n; j++) {
          double d = yi - yptr[j];
          sumexp += std::exp(-0.5 * d * d * inv_h_sq);
        }
        fhat[i] = sumexp * inv_n * norm_const;
      }
      
      NumericVector fhata(n);
      for (int i = 0; i < n; i++) fhata[i] = std::pow(fhat[i], alpha);
      
      double D = 0.0, S = 0.0;
      for (int i = 0; i < n; i++) {
        double yi = yptr[i];
        for (int j = 0; j < n; j++) {
          double d = yi - yptr[j];
          if (d > 0) {
            S += fhata[i] * d;
          } else if (d < 0) {
            D += fhata[i] * (-d);
          }
        }
      }
      D *= inv_n2;
      S *= inv_n2;
      
      res(k,0) = D + S;
      res(k,1) = D;
      res(k,2) = S;
      
      std::ostringstream oss;
      oss << "alpha=" << std::setprecision(15) << alpha;
      rn[k] = oss.str();
    }
    
    colnames(res) = CharacterVector({"paf","deprivation","surplus"});
    rownames(res) = rn;
    return res;
  }
}

// mean worker
struct MeanWorker : public Worker {
  const RVector<double> y;
  double sum;
  MeanWorker(const NumericVector& y) : y(y), sum(0.0) {}
  MeanWorker(const MeanWorker& other, Split) : y(other.y), sum(0.0) {}
  void operator()(std::size_t b, std::size_t e) {
    const double* RESTRICT ptr = y.begin();
    double local = 0.0;
    for (std::size_t i = b; i < e; ++i) local += ptr[i];
    sum += local;
  }
  void join(const MeanWorker& rhs) { sum += rhs.sum; }
};

inline double parallel_mean(const NumericVector& y, int ncores) {
  MeanWorker w(y);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelReduce(0, y.size(), w);
  return w.sum / static_cast<double>(y.size());
}

// variance worker (second pass)
struct VarWorker : public Worker {
  const RVector<double> y;
  const double mean;
  double ssq;
  VarWorker(const NumericVector& y, double mean) : y(y), mean(mean), ssq(0.0) {}
  VarWorker(const VarWorker& other, Split) : y(other.y), mean(other.mean), ssq(0.0) {}
  void operator()(std::size_t b, std::size_t e) {
    const double* RESTRICT ptr = y.begin();
    double local = 0.0;
    for (std::size_t i = b; i < e; ++i) {
      double d = ptr[i] - mean;
      local += d * d;
    }
    ssq += local;
  }
  void join(const VarWorker& rhs) { ssq += rhs.ssq; }
};

inline double parallel_sd(const NumericVector& y, double mean, int ncores) {
  VarWorker w(y, mean);
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelReduce(0, y.size(), w);
  return std::sqrt(w.ssq / (static_cast<double>(y.size()) - 1.0));
}

// ---- paf2 worker (per alpha, parallel over i) ----
struct PAF2Worker : public Worker {
  const RVector<double> y;   // normalized y
  const double a;            // alpha
  const double inv_h_sq;     // 1/h^2
  const double norm_const;   // 1/(sqrt(2*pi)*h)
  const double inv_n;        // 1/n
  
  double D_sum;              // deprivation accumulator
  double S_sum;              // surplus accumulator
  
  PAF2Worker(const NumericVector& y, double a,
             double inv_h_sq, double norm_const, double inv_n)
    : y(y), a(a), inv_h_sq(inv_h_sq), norm_const(norm_const), inv_n(inv_n),
      D_sum(0.0), S_sum(0.0) {}
  
  PAF2Worker(const PAF2Worker& other, Split)
    : y(other.y), a(other.a),
      inv_h_sq(other.inv_h_sq), norm_const(other.norm_const), inv_n(other.inv_n),
      D_sum(0.0), S_sum(0.0) {}
  
  void operator()(std::size_t begin, std::size_t end) {
    const std::size_t n = y.length();
    const double* RESTRICT yptr = y.begin();
    
    for (std::size_t i = begin; i < end; ++i) {
      const double yi = yptr[i];
      
      // fhat_i: row mean of Gaussian kernel
      double fhat_sum = 0.0;
      double dep_row = 0.0; // sum of (-d) for d<0
      double sup_row = 0.0; // sum of (+d) for d>0
      
      for (std::size_t j = 0; j < n; ++j) {
        const double d = yi - yptr[j];
        fhat_sum += std::exp(-0.5 * (d * d) * inv_h_sq);
        if (d > 0.0) sup_row += d;
        else if (d < 0.0) dep_row += -d;
      }
      
      const double fhat_i  = fhat_sum * norm_const * inv_n; // Rfast::rowmeans(...)/sqrt(2*pi)/h
      const double fhata_i = std::pow(fhat_i, a);
      
      D_sum += fhata_i * dep_row;
      S_sum += fhata_i * sup_row;
    }
  }
  
  void join(const PAF2Worker& rhs) {
    D_sum += rhs.D_sum;
    S_sum += rhs.S_sum;
  }
};

// [[Rcpp::export]]
SEXP paf2_parallel(NumericVector y, NumericVector a, int ncores = 1) {
  const int n = y.size();
  const double double_n  = static_cast<double>(n);
  const double double_n2 = double_n * double_n;
  const double inv_n     = 1.0 / double_n;
  const double inv_n2    = 1.0 / double_n2;
  
  // normalize y by mean (parallel)
  NumericVector y_norm = clone(y);
  const double mean_y = parallel_mean(y_norm, ncores);
  for (int i = 0; i < n; ++i) y_norm[i] /= mean_y;
  
  // sd of normalized y (mean is 1.0)
  const double sd_y = parallel_sd(y_norm, 1.0, ncores);
  const double com  = 4.7 / std::sqrt(double_n) * sd_y;
  
  const int lena = a.size();
  NumericVector D(lena), S(lena);
  
  for (int k = 0; k < lena; ++k) {
    const double alpha = a[k];
    const double h = com * std::pow(alpha, 0.1);
    const double h_sq = h * h;
    const double inv_h_sq = 1.0 / h_sq;
    const double norm_const = 1.0 / (std::sqrt(2.0 * M_PI) * h);
    
    PAF2Worker worker(y_norm, alpha, inv_h_sq, norm_const, inv_n);
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
    parallelReduce(0, n, worker);
    
    D[k] = worker.D_sum * inv_n2;
    S[k] = worker.S_sum * inv_n2;
  }
  
  if (lena == 1) {
    return NumericVector::create(
      _["paf"]        = D[0] + S[0],
                                _["deprivation"]= D[0],
                                                   _["surplus"]    = S[0]
    );
  } else {
    NumericMatrix res(lena, 3);
    for (int k = 0; k < lena; ++k) {
      res(k,0) = D[k] + S[k];
      res(k,1) = D[k];
      res(k,2) = S[k];
    }
    colnames(res) = CharacterVector({"paf","deprivation","surplus"});
    CharacterVector rn(lena);
    for (int k = 0; k < lena; ++k) rn[k] = "alpha=" + format_alpha(a[k]);
    rownames(res) = rn;
    return res;
  }
}

// [[Rcpp::export]]
SEXP paf2(const NumericVector y, const NumericVector a, int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1){
    return paf2_parallel(y, a, ncores);
  } else {
    return paf2_cpp(y, a);
  }
#else
  return paf2_cpp(y, a);
#endif
}
