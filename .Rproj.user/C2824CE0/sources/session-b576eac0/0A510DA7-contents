paf.boot <- function(y, a, R = 1000) {
  index <- paf::paf(y, a)
  boot <- matrix(0, R, 4)
  n <- length(y)
  for (i in 1:R) {
    ind <- Rfast2::Sample.int(n, n, replace = TRUE)
    boot[i, ] <- paf::paf(y[ind], a)
  }
  mesoi <- Rfast::colmeans(boot)
  bias <- index - mesoi
  se <- Rfast::colVars(boot, std = TRUE)
  ci <- Rfast2::Quantile( boot[, 1], probs = c(0.025, 0.975) )
  list(boot = boot, index = index, mesoi = mesoi, bias = bias, se = se, ci = ci)
}

