colpafs <- function(y, a) {
  y <- as.matrix(y)
  y <- Rfast::eachrow( y, mean(y), oper = "/" )
  dm <- dim(y)
  n <- dm[1]  ;  p <- dm[2]
  h <- 4.7 / sqrt(n) * Rfast::colVars(y, std = TRUE) * a^0.1  ## bandwidth
  res <- matrix(p, ncol = 4)
  colnames(res) <- c("paf", "alienation", "identification", "1 + rho")
  for ( i in 1:p ) {
    d <- Rfast::vecdist(y[, i])
    fhat <- Rfast::rowmeans( exp( -0.5 * d^2 / h[i]^2 ) ) / sqrt(2 * pi) / h[i]
    fhata <- fhat^a
    paf <- sum( fhata * d ) / n^2
    alien <- mean(d)
    ident <- mean(fhata)
    rho <- paf / (alien * ident) - 1
    res[i, ] <- c(paf, alien, ident, 1 + rho)
  }
  res
}






