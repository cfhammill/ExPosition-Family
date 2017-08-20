## test for canonical correlation analysies (CCA) and correspondence analysis (CA): the two most important techniques for the GSVD test.

library(SlimPosition)
#source('./R/gsvd.R')
#source('./R/tolerance.svd.R')
#source('./R/power.rebuild_matrix.R')
#source('./R/invert.rebuild_matrix.R')
#source('./R/isDiagonal.matrix.R')

  #authors data for CA
load('./data/authors.rda')


## let's just do this directly through the GSVD and do all the preprocessing here.
sum.data <- sum(authors$ca$data)
wi <- rowSums(authors$ca$data)/sum.data
wj <- colSums(authors$ca$data)/sum.data

dat <- (authors$ca$data/sum.data) - (wi %o% wj)

r_res <- gsvd(dat, 1/wi, 1/wj)
cpp_res <- SlimPosition:::gsvd_new(dat, 1/wi, 1/wj)

all.equal(r_res$tau, cpp_res$tau)

microbenchmark::microbenchmark(gsvd(dat, 1/wi, 1/wj)
                             , SlimPosition:::gsvd_new(dat, 1/wi, 1/wj))

  #wine data for CCA
load('./data/two.table.wine.rda')

X <- scale(wine$objective, center = T, scale = T)
Y <- scale(wine$subjective, center = T, scale = T)

#cca.res <- gsvd(t(X) %*% Y, crossprod(X), crossprod(Y))

dat <- t(X) %*% Y
XtX <- crossprod(X)
YtY <- crossprod(Y)

r_res <- gsvd(dat, XtX, YtY)
cpp_res <- SlimPosition:::gsvd_eig(dat, XtX, YtY, .Machine$double.eps
                                 , 5, 0)

all.equal(r_res$d, cpp_res$d)

microbenchmark::microbenchmark(gsvd(dat, XtX, YtY)
                             , SlimPosition:::gsvd_eig(dat, XtX, YtY, .Machine$double.eps
                                                     , 5, 0)
                               , SlimPosition:::gsvd_new(dat, XtX, YtY, .Machine$double.eps))


### The most important results that need to match with any other implementation of the SVD are:
  # $d (or $d.orig if using a reduced set of vectors) -- the singular values
  # $u the right singular vectors
  # $v the left singular vectors


# From there, it might be worthy to test if R or C++ are faster for the pre and post multiplications of the data and the pre multiplications of the vectors.



