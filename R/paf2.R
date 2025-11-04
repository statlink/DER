paf2 <- function(y, a) {
  y <- y / mean(y)
  n <- length(y)

  if ( length(a) == 1 ) {
    h <- 4.7 / sqrt(n) * Rfast::Var(y, std = TRUE) * a^0.1  ## bandwidth
    dD <- dS <- outer(y, y, "-")
    fhat <- Rfast::rowmeans( exp( -0.5 * dD^2 / h^2 ) ) / sqrt(2 * pi) / h
    fhata <- fhat^a
    dD[dD > 0] <- 0
    dS[dS < 0] <- 0
    D <- sum( fhata * abs(dD) ) / n^2
    S <- sum( fhata * dS ) / n^2
    res <- c(D + S, D, S)
    names(res) <- c("paf", "deprivation", "surplus")
  } else {
    dD <- dS <- outer(y, y, "-")
    com <- 4.7 / sqrt(n) * Rfast::Var(y, std = TRUE)
    d2 <-  -0.5 * dD^2
    dD[dD > 0] <- 0
    dS[dS < 0] <- 0
    lena <- length(a)
    D <- S <- numeric(lena)
    for ( i in 1:lena ) {
      h <- com * a[i]^0.1  ## bandwidth
      fhat <- Rfast::rowmeans( exp( d2 / h^2 ) ) / sqrt(2 * pi) / h
      fhata <- fhat^a[i]
      D[i] <- sum( fhata * abs(dD) ) / n^2
      S[i] <- sum( fhata * dS ) / n^2
    }
    res <- cbind(D + S, D, S)
    colnames(res) <- c("paf", "deprivation", "surplus")
    rownames(res) <- paste( "alpha=", a, sep = "" )
  }
  res
}
