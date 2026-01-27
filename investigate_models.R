library(tidyverse)
library(doParallel)
library(foreach)
library(data.table)

load("rdas/train_set_clean.RData")
load("rdas/test_set_clean.RData")

# UTILITY FUNCTIONS
# Clamp function to enforce constraint that ratings are between 0.5 and 5
# From https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#user-effects
clamp <- function(x, lower = 0.5, upper = 5)
  pmax(pmin(x, upper), lower)

# Calculate RMSE from residuals
# From https://rafalab.dfci.harvard.edu/dsbook-part-2/highdim/regularization.html#sec-netflix-loss-function
rmse <- function(r)
  sqrt(mean(r^2))

# BASELINE

# Calculate the overall average rating
mu <- mean(train_set$rating, na.rm = TRUE)

# Let's start with a baseline, using the overall average, so we can see what
# improves accuracy.
baseline <- rmse(test_set$rating - mu)

# ADDITIONAL MOVIE FEATURES EFFECT
# Effect of genre on movie * effect of genre on user
fit_als_all <- function(data = train_set,
                        lambda_u = 5,
                        lambda_m = 10,
                        tol = 1e-6,
                        max_iter = 100) {
  # Convert to data table so we can use the `:=` functional form
  fit <- as.data.table(copy(data))
  N <- nrow(fit)
  mu <- mean(fit$rating)
  
  # Index by user and movie.
  # Add genre and decade to the movie index, because they are attributes of the
  # movie (vary per movie, not per user).
  user_index <- fit[, .(a = 0), by = "userId"]
  movie_index <- fit[, .(b = 0,
                         genres = first(genres),
                         decade = first(decade)), by = "movieId"]
  
  # Calculate genre effect
  # First get just what we need: ratings per genre, number of genres, and the name of the genre(s)
  genre_index <- fit[, .(resid = sum(rating - mu), n = .N), by = "genres"]
  genre_index[, c := resid / (n + lambda_m)]
  
  # Create a temporary working data frame to use to calculate the decade effect
  # TODO: Is creating the extra tables actually faster than modifying fit directly?
  # TODO: Should I go back to using rm() to free up memory?
  temp_fit <- merge(fit, genre_index[, .(genres, c)], by = "genres")
  decade_index <- temp_fit[, .(resid = sum(rating - mu - c), n = .N), by = "decade"]
  decade_index[, d := resid / (n + lambda_m)]
  
  # Now that we calculated the effects, add them to our movie_index
  movie_index <- merge(movie_index, genre_index[, .(genres, c)], by = "genres")
  movie_index <- merge(movie_index, decade_index[, .(decade, d)], by = "decade")
  
  # Now use ALS to calculate the movie and user effects
  for (iter in 1:max_iter) {
    # Stash a copy of the existing a and b columns, so we can compare them later
    prev_a <- copy(user_index$a)
    prev_b <- copy(movie_index$b)
    
    # Create a temporary working data frame to hold our calculations by merging
    # fit with our indexes
    temp_fit <- merge(fit, user_index, by = "userId")
    temp_fit <- merge(temp_fit, movie_index[, .(movieId, b, c, d)], by = "movieId")
    
    # Estimate the user effect, given a movie, genre, and decade effect, and update
    # our user index
    user_index <- temp_fit[, .(a = sum(rating - mu - b - c - d) / (.N + lambda_u)), by = "userId"]
    
    # Recreate the temporary working data frame, now that we have an updated user effect
    temp_fit <- merge(fit, user_index, by = "userId")
    temp_fit <- merge(temp_fit, movie_index[, .(movieId, b, c, d)], by = "movieId")
    
    # Now estimate the movie effect, given a user, genre, and decade effect,
    # and update our movie index
    fit_movies <- temp_fit[, .(b = sum(rating - mu - a - c - d) / (.N + lambda_m)), by = "movieId"]
    movie_index <- merge(movie_index[, .(movieId, genres, decade, c, d)], fit_movies, by = "movieId")
    
    
    # Check for convergence
    delta <- max(c(abs(user_index$a - prev_a), abs(movie_index$b - prev_b)))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    if (delta < tol)
      break
  }
  # Return the regularized effects
  list(
    mu = mu,
    b_u = setNames(user_index$a, user_index$userId),
    b_i = setNames(movie_index$b, movie_index$movieId),
    b_g = setNames(genre_index$c, genre_index$genres),
    b_d = setNames(decade_index$d, decade_index$decade)
  )
}

# TODO: Tune and select lambdas
fit <- fit_als_all(train_set)
mu <- fit$mu
b_u <- fit$b_u
b_i <- fit$b_i
b_g <- fit$b_g
b_d <- fit$b_d

resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
with_all_effects <- rmse(resid)



# ADDITIONAL MOVIE FEATURES EFFECT
# Effect of genre on movie * effect of genre on user
fit_als_with_latent <- function(data = train_set,
                        K = 2,        
                        lambda_u = 5,
                        lambda_m = 10,
                        lambda_pq = 0,
                        tol = 1e-6,
                        max_iter = 100) {
  # Convert to data table so we can use the `:=` functional form
  fit <- as.data.table(copy(data))
  N <- nrow(fit)
  mu <- mean(fit$rating)
  
  # Index by user and movie.
  # Add genre and decade to the movie index, because they are attributes of the
  # movie (vary per movie, not per user).
  user_index <- fit[, .(a = 0), by = "userId"]
  movie_index <- fit[, .(b = 0,
                         genres = first(genres),
                         decade = first(decade)), by = "movieId"]
  
  # Calculate genre effect
  # First get just what we need: ratings per genre, number of genres, and the name of the genre(s)
  genre_index <- fit[, .(resid = sum(rating - mu), n = .N), by = "genres"]
  genre_index[, c := resid / (n + lambda_m)]
  
  # Create a temporary working data frame to use to calculate the decade effect
  # TODO: Is creating the extra tables actually faster than modifying fit directly?
  # TODO: Should I go back to using rm() to free up memory?
  temp_fit <- merge(fit, genre_index[, .(genres, c)], by = "genres")
  decade_index <- temp_fit[, .(resid = sum(rating - mu - c), n = .N), by = "decade"]
  decade_index[, d := resid / (n + lambda_m)]
  
  # Now that we calculated the effects, add them to our movie_index
  movie_index <- merge(movie_index, genre_index[, .(genres, c)], by = "genres")
  movie_index <- merge(movie_index, decade_index[, .(decade, d)], by = "decade")
  
  # Next use singular value decomposition to find the latent user effects
  # Adapted from the source code for fit_recommender_model in dslabs
  # https://cran.r-project.org/web/packages/dslabs/index.html
  p <- svd(matrix(rnorm(K * nrow(user_index), 0, 0.1), nrow(user_index), K))$u
  rownames(p) <- user_index$userId
  q <- matrix(rep(0, K * nrow(movie_index)), nrow(movie_index), K)
  rownames(q) <- movie_index$movieId
  pq <- rep(0, nrow(fit))
  prev_obj <- 0
  
  # Now use ALS to calculate the movie and user effects and the latent effects
  for (iter in 1:max_iter) {
    # Stash a copy of the existing a and b columns, so we can compare them later
    prev_a <- copy(user_index$a)
    prev_b <- copy(movie_index$b)
    
    # Create a temporary working data frame to hold our calculations by merging
    # fit with our indexes
    temp_fit <- merge(fit, user_index, by = "userId")
    temp_fit <- merge(temp_fit, movie_index[, .(movieId, b, c, d)], by = "movieId")
    
    # Estimate the user effect, given a movie, genre, and decade effect, and update
    # our user index
    user_index <- temp_fit[, .(a = sum(rating - mu - b - c - d, na.rm = FALSE) / (.N + lambda_u)), by = "userId"]
    
    # Recreate the temporary working data frame, now that we have an updated user effect
    temp_fit <- merge(fit, user_index, by = "userId")
    temp_fit <- merge(temp_fit, movie_index[, .(movieId, b, c, d)], by = "movieId")
    
    # Now estimate the movie effect, given a user, genre, and decade effect,
    # and update our movie index
    fit_movies <- temp_fit[, .(b = sum(rating - mu - a - c - d, na.rm = FALSE) / (.N + lambda_m)), by = "movieId"]
    movie_index <- merge(movie_index[, .(movieId, genres, decade, c, d)], fit_movies, by = "movieId")
    
    # Now calculate the initial pq
    pq <- rowSums(p[as.character(user_index$userId), -1, drop = FALSE] * q[as.character(movie_index$movieId), -1, drop = FALSE])
    
    message(sprintf("Resid %d: Length %d; NAs %d", iter, length(resid), sum(is.na(resid))))
    
    # For K latent factors, calculate the effect using ALS again
    for (k in 1:K) {
      q[as.character(movie_index$movieId), k] <- sapply(movie_index$movieId, function(i) { 
        x <- replace_na(p[as.character(fit$userId)[i], k], -1)
        sum(x*resid[i])/(sum(x^2))
      })
      p[as.character(user_index$userId), k] <- sapply(user_index$userId, function(i) {
        x <- replace_na(q[as.character(fit$movieId)[i], k], -1)
        sum(x*resid[i])/(sum(x^2))
      })
      resid <- resid - p[as.character(fit$userId), k]*q[as.character(fit$movieId), k]
      message(sprintf("k iter %d, p NAs %d, p not NAs %d", k, sum(is.na(p)), sum(!is.na(p))))
      if (sum(is.na(p)) > 0) break
    }
    message(sprintf("NAs: p %d, q %d, pq %d", sum(is.na(p)), sum(is.na(q)), sum(is.na(pq))))
    # Update pq now that we've calculated the effects
    pq <- rowSums(p[as.character(fit$userId), ] * q[as.character(fit$movieId), ])
    # Update our residuals
    temp_fit$pq <- pq
    resid <- temp_fit |> mutate(resid = sum(rating - mu - a - b - c - d - pq)) |> pull(resid)
    
    # Check for convergence
    delta <- max(c(abs(user_index$a - prev_a), abs(movie_index$b - prev_b)))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    
    # TODO: Use other values when calculating delta
    if (delta < tol | sum(is.na(p)) > 0)
      break
    message(sprintf("Iteration %d: Delta = %.6f", 
                    iter, delta))
  }
  list(p = p, q = q)
  
  # TODO: Figure this out!
  ## orthogonalize factors via SVD of p %*% t(q[index_q,]) and rescale by sqrt(s$d)
  # Error in qr.default(p) : NA/NaN/Inf in foreign function call (arg 1)
  # Called from: qr.default(p)
  # QR_p <- qr(p)
  # QR_q <- qr(q[as.character(movie_index$movieId),,drop = FALSE])
  # s <- svd(qr.R(QR_p) %*% t(qr.R(QR_q)))
  # u <- qr.Q(QR_p) %*% s$u
  # v <- qr.Q(QR_q) %*% s$v

  # rownames(u) <- rownames(p)
  # rownames(v) <- rownames(q[as.character(movie_index$movieId),,drop = FALSE])
  # p <- sweep(u, 2, sqrt(s$d), FUN = "*")
  # q[as.character(movie_index$movieId),] <- sweep(v, 2, sqrt(s$d), FUN = "*")

  # Return the regularized effects
  # list(
  #  mu = mu,
  #  b_u = setNames(user_index$a, user_index$userId),
  #  b_i = setNames(movie_index$b, movie_index$movieId),
  #  b_g = setNames(genre_index$c, genre_index$genres),
  #  b_d = setNames(decade_index$d, decade_index$decade),
  #  p = p,
  #  q = q
  # )
}

# TODO: Tune and select lambdas
# TODO: Try with a more realistic K
fit <- fit_als_with_latent(train_set, max_iter = 5, tol = 1e-3)
mu <- fit$mu
b_u <- fit$b_u
b_i <- fit$b_i
b_g <- fit$b_g
b_d <- fit$b_d

pq <- rowSums(fit$p[as.character(test_set$userId), ] * q[as.character(test_set$movieId), ])

resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade) + pq]))
with_latent <- rmse(resid)