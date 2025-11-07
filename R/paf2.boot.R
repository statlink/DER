paf2.boot <- function(y, a, R = 1000, ncores = 1) {
  index <- DER::paf2(y, a, ncores)
  boot <- matrix(0, R, 3)
  n <- length(y)
  
  for (i in 1:R) {
    boot[i, ] <-DER::paf2(y[Rfast2::Sample.int(n, n, replace = TRUE)], a, ncores)
  }
  colnames(boot) <- c("paf", "deprivation", "surplus")
  mesoi <- Rfast::colmeans(boot)
  bias <- index - mesoi
  se <- Rfast::colVars(boot, std = TRUE)
  ci <- Rfast2::colQuantile( boot, probs = c(0.025, 0.975) )
  info <- rbind(mesoi, bias, se, ci)
  colnames(info) <- colnames(boot)
  rownames(info) <- c("mesoi", "bias", "se", "2.5%", "97.5%" )
  list(boot = boot, index = index, info = info)
}

