colpafs2 <- function(y, a) {
  y <- Rfast::eachrow( y, Rfast::colmeans(y), oper = "/" )
  dm <- dim(y)
  n <- dm[1]  ;  p <- dm[2]
  h <- 4.7 / sqrt(n) * Rfast::colVars(y, std = TRUE) * a^0.1  ## bandwidth

  res <- matrix(0, p, ncol = 3)
  colnames(res) <- c("paf", "deprivation", "surplus")
  rownames(res) <- colnames(y)
  for ( i in 1:p ) {
    dD <- dS <- outer(y[, i], y[, i], "-")
    fhat <- Rfast::rowmeans( exp( -0.5 * dD^2 / h[i]^2 ) ) / sqrt(2 * pi) / h[i]
    fhata <- fhat^a
    dD[dD > 0] <- 0
    dS[dS < 0] <- 0
    D <- sum( fhata * abs(dD) ) / n^2
    S <- sum( fhata * dS ) / n^2
    res[i, ] <- c(D + S, D, S)
  }
  res
}



