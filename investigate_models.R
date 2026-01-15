library(tidyverse)
library(data.table)

load("rdas/train_set_clean.RData")
load("rdas/test_set_clean.RData")

# UTILITY FUNCTIONS
# Clamp function to enforce constraint that ratings are between 0.5 and 5
# From https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#user-effects 
clamp <- function(x, lower = 0.5, upper = 5) pmax(pmin(x, upper), lower)

# Calculate RMSE from residuals
# From https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#sec-netflix-loss-function
rmse <- function(r) sqrt(mean(r^2))

# BASELINE

# Calculate the overall average rating
mu <- mean(train_set$rating, na.rm = TRUE)

# Let's start with a baseline, using the overall average, so we can see what
# improves accuracy.
baseline <- rmse(test_set$rating - mu)

# USER AND MOVIE EFFECTS
# fit_als function taken from https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#penalized-least-squares
fit_als <- function(data = train_set, lambda = 0.0001, tol = 1e-6, max_iter = 100) {
  # Convert to data table so we can use the `:=` functional form
  fit <- as.data.table(copy(data))
  N <- nrow(fit)
  mu <- mean(fit$rating)
  # Adds columns a and b to the fit table. Initial value for all rows is 0.
  fit[, `:=`(a = 0, b = 0)]
  
  for (iter in 1:max_iter) {
    # Stash a copy of the existing a and b columns, so we can compare them later
    prev_a <- copy(fit$a)
    prev_b <- copy(fit$b)
    
    # Estimate the user effect, given a movie effect
    fit[, a := sum(rating - mu - b)/(.N + N*lambda) , by = userId]
    # Now alternate and estimate the movie effect, given a user effect
    fit[, b := sum(rating - mu - a)/(.N + N*lambda), by = movieId]
    
    # Calculate the amount of the update
    delta <- max(c(abs(fit$a - prev_a), abs(fit$b - prev_b)))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    if (delta < tol) break
  }
  # Return the regularized effects
  return(with(fit, list(mu = mu, b_u = setNames(a, userId), b_i = setNames(b, movieId))))
}

# Find the best lambda
# From the first run, lambda is 0.040404040
# Commenting this out so we don't keep running the code.
# lambdas <- seq(0, 1, length.out = 100)
# rmses <- sapply(lambdas, function(l) { 
#   fit <- fit_als(lambda = l, tol = 1e-3)
#   resid <- with(test_set, rating - clamp(fit$mu + fit$b_u[userId] + fit$b_i[movieId]))
#   rmse(resid)
# })

# Now that we know the general area, try a much smaller range
# lambdas <- seq(0.03, 0.05, length.out = 100)
# rmses <- sapply(lambdas, function(l) { 
#   fit <- fit_als(lambda = l, tol = 1e-3)
#   resid <- with(test_set, rating - clamp(fit$mu + fit$b_u[userId] + fit$b_i[movieId]))
#   rmse(resid)
# })
# 
# plot(lambdas, rmses, type = "l", xlab = expression(lambda), ylab = "RMSE")
# 
# lambda <- lambdas[which.min(rmses)]
# Commenting out the code and hard-coding the result so we don't keep running
# a very slow loop
lambda <- 0.04454545454545454545

fit <- fit_als(train_set, lambda)
mu <- fit$mu
b_u <- fit$b_u
b_i <- fit$b_i

resid <- with(test_set, rating - clamp(mu + b_i[movieId] + b_u[userId]))
with_both_effect <- rmse(resid)

# GENRE EFFECT
# TODO: Genre calculated this way is less accurage that genre effect on its own
# Either we are doing ALS wrong and/or we need to calculate a better lambda
fit_als_genre <- function(data = train_set, lambda_1 = 0.0001, lambda_2 = 0.0001, tol = 1e-6, max_iter = 100) {
  fit <- as.data.table(copy(data))
  N <- nrow(fit)
  mu <- mean(fit$rating)
  fit[, `:=`(a = 0, b = 0, c = 0)]
  
  for (iter in 1:max_iter) {
    prev_a <- copy(fit$a)
    prev_b <- copy(fit$b)
    prev_c <- copy(fit$c)
    
    fit[, a := sum(rating - mu - b - c)/(.N + N*lambda_1) , by = userId]
    fit[, b := sum(rating - mu - a - c)/(.N + N*lambda_1), by = movieId]
    fit[, c := sum(rating - mu - a - b)/(.N + N*lambda_2), by = genres]
    
    delta <- max(c(abs(fit$a - prev_a), abs(fit$b - prev_b), abs(fit$c - prev_c)))
    if (delta < tol) break
  }
  return(with(fit, list(mu = mu, b_u = setNames(a, userId), b_i = setNames(b, movieId), b_g = setNames(c, genres))))
}

fit <- fit_als_genre(lambda_1 = lambda, lambda_2 = 0.0001)
mu <- fit$mu
b_u <- fit$b_u
b_i <- fit$b_i
b_g <- fit$b_g

resid_genre <- with(test_set, rating - clamp(mu + b_i[movieId] + b_u[userId] + b_g[genres]))
with_genre_effect <- rmse(resid_genre)
prev_resid <- 1.06017851959758

# Let's try with decade
fit_als_decade <- function(data = train_set, lambda_1 = 0.0001, lambda_2 = 0.001, tol = 1e-6, max_iter = 100) {
  fit <- as.data.table(copy(data))
  N <- nrow(fit)
  mu <- mean(fit$rating)
  fit[, `:=`(a = 0, b = 0, c = 0)]
  
  for (iter in 1:max_iter) {
    prev_a <- copy(fit$a)
    prev_b <- copy(fit$b)
    prev_c <- copy(fit$c)
    
    fit[, a := sum(rating - mu - b - c)/(.N + N*lambda) , by = userId]
    fit[, b := sum(rating - mu - a - c)/(.N + N*lambda), by = movieId]
    fit[, c := sum(rating - mu - a - b)/(.N + N*lambda), by = decade]
    
    delta <- max(c(abs(fit$a - prev_a), abs(fit$b - prev_b), abs(fit$c - prev_c)))
    if (delta < tol) break
  }
  return(with(fit, list(mu = mu, b_u = setNames(a, userId), b_i = setNames(b, movieId), b_d = setNames(c, decade))))
}
fit <- fit_als_decade(lambda_1 = lambda, lambda_2 = lambda)
mu <- fit$mu
b_u <- fit$b_u
b_i <- fit$b_i
b_d <- fit$b_d

# Decade also decreases the accuracy.
resid <- with(test_set, rating - clamp(mu + b_i[movieId] + b_u[userId] + b_d[decade]))
with_decade <- rmse(resid)

# What if we try with decade on its own?
fit_decades <- train_set |> group_by(decade) |> summarise(b_d = mean(rating - mu))
resid <- with(test_set, rating - clamp(mu + fit_decades$b_d[decade]))
with_decade_only <- rmse(resid)

# TODO: What if we treat each genre separately?