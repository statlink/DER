\name{The PaF income polarization index}
\alias{paf}
\alias{paf2}
\title{
The PaF income polarization index
}

\description{
The PaF income polarization index
}

\usage{
paf(y, a)
paf2(y, a)
}

\arguments{
\item{y}{
A numeric vector with income data.
}
\item{a}{
The value of \eqn{\alpha}. This can either be a number or a vector with many values. In any case, the \eqn{\alpha} may take values between 0.25 and 1.
}
}

\details{
The functions compute the PaF index of Duclos, Esteban and Ray (2004) for either a specific value, or for a range of values, of \eqn{\alpha}. The paf2() estimates the index using Eq. (8) and (9) in the paper, whereas paf() is faster as it uses Eq. (3) of the paper.
}

\value{
The paf() function, for a single value of \eqn{\alpha}, returns a vector with the PaF index, the alienation (twice the Gini index) and  identification components and 1 + the normalized covariance. If a range of values of \eqn{\alpha} are given, it will return a matrix with the same components, where each row corresponds to a specific value of \eqn{\alpha}.

The paf2() function returns only the PaF index for either one or more values of \eqn{\alpha}.
}

\references{
Duclos J. Y., Esteban, J. and Ray D. (2006). Polarization: concepts, measurement, estimation. In The Social Economics of Poverty (pp. 54--102). Routledge.

Duclos J. Y., Esteban, J. and Ray D. (2004). Polarization: concepts, measurement, estimation. Econometrica, 72(6): 1737--1772.
}

\author{
Michail Tsagris.

R implementation and documentation: Michail Tsagris \email{mtsagris@uoc.gr}.
}

%\note{
%%  ~~further notes~~
%}

\seealso{
\code{\link{paf.boot}
}
}

\examples{
y <- rgamma(100, 500, 0.03)
paf(y, 0.25)
paf( y, c(0.25, 0.5, 0.75, 1) )
}
