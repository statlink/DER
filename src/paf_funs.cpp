// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
using namespace Rcpp;
using namespace RcppParallel;

#ifdef RCPP_PARALLEL_USE_TBB
#include <tbb/global_control.h>  // For controlling the number of threads
#endif

// // [[Rcpp::export]]
// List kernel_paf(NumericVector y, double a) {
//   int n = y.size();
//   double h = 4.7 / std::sqrt((double)n) * sd(y) * std::pow(a, 0.1);
//   double h_sq = h * h;
//   double n_norm_const = 1.0 / (n * std::sqrt(2.0 * M_PI) * h);
//
//   // // kernel row sums and distance row sums
//   std::vector<double> fhata(n, 0.0), row_sums(n, 0.0);
//
//   // // loop over upper triangle
//   for (int i = 0; i < n - 1; i++) {
//     for (int j = i + 1; j < n; j++) {
//       double d = std::abs(y[i] - y[j]);
//       double k = std::exp(-0.5 * (d * d / h_sq));
//
//       fhata[i] += k; // still fhat
//       fhata[j] += k; // still fhat
//
//       row_sums[i] += d;
//       row_sums[j] += d;
//     }
//   }
//
//   // // add diagonal contributions (distance=0, kernel=1)
//   // // fhata = fhat^a
//   for (int i = 0; i < n; i++) {
//     fhata[i] = std::pow((fhata[i] + 1.0) * n_norm_const, a);
//   }
//
//   // // paf = sum(fhata * row_sums) / n^2
//   // // compute paf, alien, and ident in the same loop
//   double paf = 0.0, alien = 0.0, ident = 0.0;
//   for (int i = 0; i < n; i++) {
//     paf += fhata[i] * row_sums[i];
//     alien += row_sums[i];
//     ident += fhata[i];
//   }
//
//   paf /= (double)(n * n);
//   alien /= (double)n * (double)n;
//   ident /= (double)n;
//
//   return List::create(
//     _["alien"] = alien,
//     _["ident"] = ident,
//     _["paf"] = paf
//   );
// }

// Define a RESTRICT macro for portability.
#if defined(__GNUG__) || defined(__clang__)
#  define RESTRICT __restrict__
#else
#  define RESTRICT __restrict
#endif

List kernel_paf(NumericVector y, double a, double h) {
  int n = y.size();
  const double * RESTRICT yptr = y.begin();  // raw pointer with restrict

  double h_sq = h * h;
  double n_norm_const = 1.0 / (n * std::sqrt(2.0 * M_PI) * h);

  double ident = 0.0, alien = 0.0, paf = 0.0;

  for (int i = 0; i < n; i++) {
    double fhat_sum = 1.0;
    double row_sum  = 0.0;
    double yi = yptr[i];

    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      double d = std::abs(yi - yptr[j]);
      double k = std::exp(-0.5 * (d * d / h_sq));
      fhat_sum += k;
      row_sum  += d;
    }

    double fhata_i = std::pow(fhat_sum * n_norm_const, a);

    ident += fhata_i;
    alien += row_sum;
    paf   += fhata_i * row_sum;
  }

  ident /= (double)n;
  alien /= (double)n * (double)n;
  paf   /= (double)n * (double)n;

  return List::create(
    _["paf"]   = paf,
    _["alien"] = alien,
    _["ident"] = ident
  );
}

struct KernelPAFWorker : public Worker {
  const RVector<double> y;
  const double a;
  const double h_sq;
  const double n_norm_const;

  double ident_sum;
  double alien_sum;
  double paf_sum;

  KernelPAFWorker(const NumericVector& y, double a, double h_sq,
                  double n_norm_const)
    : y(y), a(a), h_sq(h_sq), n_norm_const(n_norm_const),
      ident_sum(0.0), alien_sum(0.0), paf_sum(0.0) {}

  KernelPAFWorker(const KernelPAFWorker& other, Split)
    : y(other.y), a(other.a), h_sq(other.h_sq), n_norm_const(other.n_norm_const),
      ident_sum(0.0), alien_sum(0.0), paf_sum(0.0) {}

  void operator()(std::size_t begin, std::size_t end) {
    std::size_t n = y.length();
    for (std::size_t i = begin; i < end; i++) {
      double fhat_sum = 1.0;
      double row_sum  = 0.0;

      for (std::size_t j = 0; j < n; j++) {
        if (i == j) continue;
        double d = std::abs(y[i] - y[j]);
        double k = std::exp(-0.5 * (d * d / h_sq));
        fhat_sum += k;
        row_sum  += d;
      }

      double fhata_i = std::pow(fhat_sum * n_norm_const, a);

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

List kernel_paf_parallelReduce(NumericVector y, double a, double h,
                               int ncores) {
  std::size_t n = y.size();
  // double h = 4.7 / std::sqrt((double)n) * sd(y) * std::pow(a, 0.1);
  double h_sq = h * h;
  double n_norm_const = 1.0 / (n * std::sqrt(2.0 * M_PI) * h);
  KernelPAFWorker worker(y, a, h_sq, n_norm_const);

  // Control the maximum number of threads.
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, ncores);
  parallelReduce(0, n, worker);

  double ident = worker.ident_sum / (double)n;
  double alien     = worker.alien_sum / ((double)n * (double)n);
  double paf            = worker.paf_sum   / ((double)n * (double)n);

  return List::create(
    _["paf"]            = paf,
    _["alien"]     = alien,
    _["ident"] = ident
  );
}

// [[Rcpp::export]]
List paf_fun(NumericVector y, double a, double h, int ncores = 1) {
#if RCPP_PARALLEL_USE_TBB
  if (ncores > 1){
    return kernel_paf_parallelReduce(y, a, h, ncores);
  } else {
    return kernel_paf(y, a, h);
  }
#else
  return kernel_paf(y, a, h);
#endif
}

// extern "C" SEXP paf_fun(SEXP ySEXP, SEXP aSEXP, SEXP hSEXP, SEXP ncoresSEXP) {
//   NumericVector y(ySEXP);
//   double a     = as<double>(aSEXP);
//   double h     = as<double>(hSEXP);
//   int ncores   = as<int>(ncoresSEXP);
//   
// #if RCPP_PARALLEL_USE_TBB
//   if (ncores > 1) {
//     return wrap(kernel_paf_parallelReduce(y, a, h, ncores));
//   } else {
//     return wrap(kernel_paf(y, a, h));
//   }
// #else
//   return wrap(kernel_paf(y, a, h));
// #endif
// }
