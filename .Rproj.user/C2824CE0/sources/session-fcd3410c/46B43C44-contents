\name{Many decomposed PaF income polarization indices}
\alias{colpafs2}
\title{
Many decomposed PaF income polarization indices
}

\description{
Many decomposed PaF income polarization indices
}

\usage{
colpafs2(y, a)
}

\arguments{
\item{y}{
A numeric matrix with income data. The PaF index will be computed for each column sperately.
}
\item{a}{
The value of \eqn{\alpha}, a number between 0.25 and 1.
}
}

\details{
The function compute the decomposed PaF index of Duclos, Esteban and Ray (2004) for a specific value of \eqn{\alpha}, for each column of the matrix. The decomposition is with respect to the deprivation and surplus components as suggested by Araar (2008).
}

\value{
A matrix, where each row contains the PaF index, the deprivation and the surplus components.
}

\references{
Araar A. (2008). On the Decomposition of Polarization Indices: Illustrations with Chinese and Nigerian Household Surveys. CIRPEE Working Paper No. 08-06.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1136142

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
\code{\link{paf2}, \link{colpafs}
}
}

\examples{
y <- matrix( rgamma(100 * 10, 10, 0.01), ncol = 10 )
colpafs2(y, 0.25)
}
