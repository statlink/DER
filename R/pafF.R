# Polarization index, Duclos et al. (2004) page 16
pafF <- function(y, a, ncores = 1) {
  ## y is the outcome of the aldmck function
  ## a is the a value
  return(pafF_helper(Rfast::Sort(y), a, ncores)) ## the ys must be sorted
}
