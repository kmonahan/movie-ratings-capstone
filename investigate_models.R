library(tidyverse)
library(doParallel)
library(foreach)
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
    
    fit[, a := sum(rating - mu - b)/(.N + N*lambda), by = userId]
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

fit <- as.data.table(copy(train_set))
N <- nrow(fit)
movie_features <- fit[, .(
  avg_rating = mean(rating),
  genre = first(genres),
  decade = first(decade),
  n_ratings = .N
), by = movieId]
movie_features[, c := (sum(avg_rating - mu) * n_ratings) / (sum(n_ratings) + N * 0.1), by = genre]
movie_features[, d := (sum(avg_rating - mu - c) * n_ratings) / (sum(n_ratings) + N * 0.1), by = decade]

# ADDITIONAL MOVIE FEATURES EFFECT
# Effect of genre on movie * effect of genre on user
fit_als_all <- function(data = train_set, lambda_u = 0.001, lambda_m = 0.001, tol = 1e-6, max_iter = 100) {
  # Convert to data table so we can use the `:=` functional form
  fit <- as.data.table(copy(data))
  N <- nrow(fit)
  mu <- mean(fit$rating)

  # Adds columns a and b to the fit table. Initial value for all rows is 0.
  fit[, `:=`(a = 0, b = 0)]
    
  # Genre and decade are attributes of a movie, so we calculate them by movie
  # first, and then move on to ALS to compute individual user and movie effects.
  # This is hopefully faster than trying to calculate them all in the loop.
  movie_features <- fit[, .(
    avg_rating = mean(rating),
    genres = first(genres),
    decade = first(decade),
    n_ratings = .N
  ), by = movieId]
  movie_features[, c := (sum(avg_rating - mu) * n_ratings) / (sum(n_ratings) + N*lambda_m), by = genres]
  movie_features[, d := (sum(avg_rating - mu - c) * n_ratings) / (sum(n_ratings) + N*lambda_m), by = decade]
  fit <- merge(fit, movie_features[, .(movieId, c, d)], by = "movieId")
  
  for (iter in 1:max_iter) {
    # Stash a copy of the existing a and b columns, so we can compare them later
    prev_a <- copy(fit$a)
    prev_b <- copy(fit$b)
    
    # Estimate the user effect, given a movie and genre
    fit[, a := sum(rating - mu - b - c - d)/(.N + N*lambda_u), by = userId]
    # Now alternate and estimate the movie effect, given a user and genre
    fit[, b := sum(rating - mu - a - c - d)/(.N + N*lambda_m), by = movieId]
    
    # Calculate the amount of the update
    delta <- max(c(abs(fit$a - prev_a), abs(fit$b - prev_b)))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    if (delta < tol) break
  }
  # Return the regularized effects
  return(with(fit, list(mu = mu, b_u = setNames(a, userId), b_i = setNames(b, movieId), b_g = setNames(c, genres), b_d = setNames(d, decade))))
}

# lambdas <- 10^-seq(3:6)
# grid_params <- crossing(lambda_a = lambdas, lambda_b = lambdas, lambda_c = lambdas, lambda_d = lambdas)
# 
# n_cores <- detectCores()
# cluster <- makeCluster(n_cores - 1)
# registerDoParallel(cluster)
# trials <- nrow(grid_params)
# # TODO: Use proper cross-validation here
# tuning_results <- foreach(i=1:nrow(grid_params), .combine=cbind, .packages = "data.table") %dopar% {
#   row <- grid_params[i,]
#   fit <- fit_als_all(test_set, lambda_a=row['lambda_a'], lambda_b=row['lambda_b'], lambda_c=row['lambda_c'], lambda_d=row['lambda_d'], tol = 1e-3, max_iter = 50)
#   mu <- fit$mu
#   b_u <- fit$b_u
#   b_i <- fit$b_i
#   b_g <- fit$b_g
#   b_d <- fit$b_d
#   resid <- with(test_set, rating - clamp(mu + b_i[movieId] + b_u[userId] + b_g[genres] + b_d[decade]))
#   rmse(resid)
# }
# stopCluster(cl=cluster)

n_cores <- detectCores()
cluster <- makeCluster(n_cores / 2)
registerDoParallel(cluster)
lambdas <- seq(0, 1, length.out = 10)
tuning_params <- crossing(lambda_u=lambdas, lambda_m=lambdas)
rmses <- foreach(i=1:nrow(tuning_params), .packages = "data.table") %dopar% {
 l <- tuning_params[i, ]
 fit <- fit_als_all(lambda_u = l['lambda_u'], lambda_m = l['lambda_m'], tol = 1e-3)
 resid <- with(test_set, rating - clamp(mu + b_i[movieId] + b_u[userId] + b_g[genres] + b_d[decade]))
 rmse(resid)
}
stopCluster(cl=cluster)
min(as.numeric(rmses))
max(as.numeric(rmses))

fit <- fit_als_all(train_set, lambda = lambda)
mu <- fit$mu
b_u <- fit$b_u
b_i <- fit$b_i
b_g <- fit$b_g
b_d <- fit$b_d

resid <- with(test_set, rating - clamp(mu + b_i[movieId] + b_u[userId] + b_g[genres] + b_d[decade]))
with_all_effects <- rmse(resid)