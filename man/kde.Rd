\name{Kernel Density Estimation}
\alias{kde}
\title{
Kernel Density Estimation (KDE)
}

\description{
Kernel density estimation of a univariate sample using a Gaussian kernel.
}

\usage{
kde(y, ncores = 1)
}

\arguments{
\item{y}{
A numeric vector containing the sample data.
}
\item{ncores}{
The number of cores to use. If greater than 1, parallel computing will take
place. It is advisable to use it if you have many observations and or many
variables, otherwise it will slow down the process. The default is 1, meaning
that code is executed serially.
}
}

\details{
The kernel density estimate (KDE) provides a smooth estimate of the probability
density function of the sample data.

For a sample \eqn{y_1, \dots, y_n}, the KDE at a point \eqn{x} is defined as

\deqn{
\hat{f}(x) = \frac{1}{n h} \sum_{j=1}^n K\!\left(\frac{x - y_j}{h}\right),
}

where \eqn{K(z)} is the kernel function. In these functions, a Gaussian kernel
is used:

\deqn{
K(z) = \frac{1}{\sqrt{2\pi}} \exp\!\left(-\tfrac{1}{2} z^2\right).
}

The bandwidth \eqn{h} is chosen according to Silverman's rule of thumb:

\deqn{
h = 1.06 \cdot \hat{\sigma} \cdot n^{-1/5},
}

where \eqn{\hat{\sigma}} is the sample standard deviation.
}

\author{
Michail Tsagris and Christos Adam.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}
and Christos Adam \email{econp266@econ.soc.uoc.gr}.
}

\examples{
set.seed(123)
y <- rnorm(200)

dens   <- kde(y)

plot(y, dens, main = "KDE at sample points", ylab = "Density")
}
