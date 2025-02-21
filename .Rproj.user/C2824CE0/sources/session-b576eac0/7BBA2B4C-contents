\name{Many PaF income polarization indices}
\alias{colpafs}
\title{
Many PaF income polarization indices
}

\description{
Many PaF income polarization indices
}

\usage{
colpafs(y, a)
}

\arguments{
\item{y}{
A numeric matrix with income data. The PaF index will be computed for each column sperately.
}
\item{a}{
The value of \eqn{\alpha}. This can either be a number or a vector with many values. In any case, the \eqn{\alpha} may take values between 0.25 and 1.
}
}

\details{
The function compute the PaF index of Duclos, Esteban and Ray (2004) for a specific value of \eqn{\alpha}, for each column of the matrix.
}

\value{
A matrix, where each row contains the PaF index, the alienation (twice the Gini index) and identification components and 1 + the normalized covariance.
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
\code{\link{paf}
}
}

\examples{
y <- matrix( abs( rnorm(100) ) + 1000, ncol = 10 )
colpafs(y, 0.25)
}
