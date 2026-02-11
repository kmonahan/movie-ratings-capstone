library("tidyverse")

load("rdas/final_holdout_test_clean.RData")
load("rdas/fit.RData")

# UTILITY FUNCTIONS
# Clamp function to enforce constraint that ratings are between 0.5 and 5
# From https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#user-effects
clamp <- function(x, lower = 0.5, upper = 5)
  pmax(pmin(x, upper), lower)

# Calculate RMSE from residuals
# From https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#sec-netflix-loss-function
rmse <- function(r)
  sqrt(mean(r^2))

mu <- fit$mu
b_u <- setNames(fit$b_u$a, fit$b_u$userId)
b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
b_g <- setNames(fit$b_g$c, fit$b_g$genres)
b_d <- setNames(fit$b_d$d, fit$b_d$decade)

pq <- rowSums(fit$p[as.character(final_holdout_test$userId), ] * fit$q[as.character(final_holdout_test$movieId), ])
final_holdout_test$pq <- pq
resid <- with(final_holdout_test, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)] + pq))
rmse(resid)