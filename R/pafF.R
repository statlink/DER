# Polarization index, Duclos et al. (2004) page 16
pafF <- function(y, a) {
  ## y is the outcome of the aldmck function
  ## a is the a value
  y <- sort(y)  ## the ys must be sorted
  m <- mean(y)  ## mu
  n <- length(y)  ## sample size
  i <- 1:n
  ayi <- m + 2/n * i * y - y - 2 / n * cumsum(y)  ## a(y_i)
  kernel <- function(x)  mean( dnorm( (x - y)/ h ) ) / h
  kpdf <- function(x)  sapply(x, kernel)

  if ( length(a) == 1 ) {
    h <- 4.7 /sqrt(n) * Rfast::Var(y, std = TRUE) * a^0.1  ## bandwidth
    fhat <- kpdf(y)  ## kernel density estimate
    est <- mean(fhat^a * ayi)  ## polarization index
  } else {
    com <- 4.7 /sqrt(n) * Rfast::Var(y, std = TRUE)
    la <- length(a)
    est <- numeric(la)
    for ( i in 1:la ) {
      h <- com * a[i]^0.1  ## bandwidth
      fhat <- kpdf(y)  ## kernel density estimate
      est[i] <- mean( fhat^a[i] * ayi )  ## polarization index
    }
  }
  est / m^(1 - a)
}
