\name{Kernel Density Estimation}
\alias{kde}
\title{
Kernel Density Estimation (KDE)
}

\description{
Kernel density estimation of a univariate sample using a Gaussian kernel.
}

\usage{
kde(y, h, ncores = 1)
}

\arguments{
\item{y}{
A numeric vector containing the sample data.
}
\item{h}{
A numeric value for boundwidth.
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
density function of the sample data. In this implementation, the KDE is
evaluated only at the observed sample points themselves.
}

\value{
Α numeric vector of density estimates corresponding to each observation in
\code{y}.
}

\author{
Michail Tsagris and Christos Adam.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}
and Christos Adam \email{econp266@econ.soc.uoc.gr}.
}

\examples{
set.seed(123)
y <- rnorm(200)

dens   <- kde(y, h=1)

plot(y, dens, main = "KDE at sample points", ylab = "Density")
}
