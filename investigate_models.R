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
                        K = 5,        
                        lambda_u = 300,
                        lambda_m = 300,
                        lambda_pq = 700,
                        tol = 1e-6,
                        max_iter = 100) {
  
  # Copy the data so we can mutate it at will
  fit <- as.data.table(copy(data))
  
  # Calculate some initial numbers
  N <- nrow(fit)
  mu <- mean(fit$rating)

  # Shorthand for easy reference
  user_ids <- as.character(fit$userId)
  movie_ids <- as.character(fit$movieId)
  
  # Index by user and movie.
  user_index <- split(1:N, user_ids)
  movie_index <- split(1:N, movie_ids)
  unique_users <- unique(user_ids)
  unique_movies <- unique(movie_ids)
  
  n_item <- sapply(movie_index, length)
  min_ratings_index <- which(n_item >= 20)
  movie_index_min <- movie_index[min_ratings_index]
  
  movie_ids_by_length <- n_item[movie_ids]
  min_data_index <- which(movie_ids_by_length >= 20)
  user_index_min <- split(min_data_index, user_ids[min_data_index])
  
  # Set initial user and movie effects of 0
  fit$a <- rep(0, N)
  fit$b <- rep(0, N)
  
  # Calculate genre effect
  # First get just what we need: ratings per genre, number of genres, and the name of the genre(s)
  fit_genre <- fit |> group_by(genres) |> summarize(resid = sum(rating - mu), n = n(), genres = first(genres))
  fit_genre <- fit_genre |> mutate(c = resid / (n + lambda_m)) |> select(genres, c)
  fit <- left_join(fit, fit_genre, by = "genres")
  rm(fit_genre)
  
  # Calculate decade effect
  fit_decade <- fit |> group_by(decade) |> summarize(resid = sum(rating - mu - c), n = n(), decade = first(decade))
  fit_decade <- fit_decade |> mutate(d = resid / (n + lambda_m)) |> select(decade, d)
  fit <- left_join(fit, fit_decade, by = "decade")
  rm(fit_decade)
  
  # Next use singular value decomposition to find the latent user effects
  # Adapted from the source code for fit_recommender_model in dslabs
  # https://cran.r-project.org/web/packages/dslabs/index.html
  I <- length(unique_users)
  J <- length(unique_movies)
  p <- svd(matrix(rnorm(K * I, 0, 0.1), I, K))$u
  rownames(p) <- unique_users
  q <- matrix(rep(0, K * J), J, K)
  rownames(q) <- unique_movies
  pq <- rep(0, N)
  prev_mse <- 0
  
  # Now use ALS to calculate the movie and user effects and the latent effects
  for (iter in 1:max_iter) {
    # Estimate the user effect, given a movie, genre, and decade effect, and update
    # our user index
    fit_users <- fit |> 
      group_by(userId) |> 
      summarize(a = sum(rating - mu - b - c - d) / (n() + lambda_u), userId = first(userId)) |> 
      select(userId, a)
    fit <- rows_update(fit, fit_users, by = "userId")
    rm(fit_users)
    
    # Now estimate the movie effect, given a user, genre, and decade effect,
    # and update our movie index
    fit_movies <- fit |> 
      group_by(movieId) |> 
      summarize(b = sum(rating - mu - a - c - d) / (n() + lambda_m), movieId = first(movieId)) |> 
      select(movieId, b)
    fit <- rows_update(fit, fit_movies, by = "movieId")
    rm(fit_movies)
    
    # Now calculate the initial pq
    pq <- rowSums(p[user_ids, -1, drop = FALSE] * q[movie_ids, -1, drop = FALSE])
    resid <- with(fit, rating - (mu + a + b + c + d + pq))
    
    # For K latent factors, calculate the effect using ALS again
    for (k in 1:K) {
      q[min_ratings_index, k] <- sapply(movie_index_min, function(i) {
        x <- p[user_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq)
      })
      
      p[, k] <- sapply(user_index_min, function(i) {
        x <- q[movie_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq)
      })
      
      resid <- resid - p[user_ids, k]*q[movie_ids, k]
    }
    # Update pq now that we've calculated the effects
    pq <- rowSums(p[user_ids, ] * q[movie_ids, ])
    # Update our residuals
    fit$pq <- pq
    resid <- with(fit, rating - (mu + a + b + c + d + pq))
    
    # Check for convergence
    # TODO: Should the lambdas be in here?
    mse <- mean(resid^2)
    delta <- abs((prev_mse - mse) / (prev_mse + tol))
    message(sprintf("Iteration %d: Delta = %.6f", 
                    iter, delta))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    if (delta < tol)
      break
    prev_mse <- mse
  }
  
  # Create canonical form of orthogonal factors, ordered by importance
  # "Orthogonal" = pointing in unrelated directions (at 90 degrees to one another)
  # This helps prevent factors from being redundant and/or overlapping
  
  # Taken from the source code for fit_recommender_model in dslabs
  # https://cran.r-project.org/web/packages/dslabs/index.html
  
  # Computes the QR decomposition of p
  QR_p <- qr(p)
  # Computes the QR decomposition of q, including only movies with at least 20 ratings
  QR_q <- qr(q[min_ratings_index,,drop = FALSE])
  # Computes the SVD of the product of the two R matrices
  # That way, we're multiplying KxK matrices instead of the much larger NxN matrices
  s <- svd(qr.R(QR_p) %*% t(qr.R(QR_q)))
  # Creates the new orthogonalized user factors
  u <- qr.Q(QR_p) %*% s$u
  # Same but for movies
  v <- qr.Q(QR_q) %*% s$v
  # Give our new factors the same rownames as the original p and q
  rownames(u) <- rownames(p)
  rownames(v) <- rownames(q[min_ratings_index,,drop = FALSE])
  # Multiply all columns in u by the square root of s$d, which represents the importance/strength of the factor
  # Using the square root keeps p and q on similar scales
  p <- sweep(u, 2, sqrt(s$d), FUN = "*")
  # Same multiplication for q
  q[min_ratings_index,] <- sweep(v, 2, sqrt(s$d), FUN = "*")
  # Now factors are ordered by importance -- the first column captures the most variance and so on
  
  b_u <- fit |> group_by(userId) |> summarize(a = first(a))
  b_i <- fit |> group_by(movieId) |> summarize(b = first(b))
  b_g <- fit |> group_by(genres) |> summarize(c = first(c))
  b_d <- fit |> group_by(decade) |> summarize(d = first(d))
  

  # Return the regularized effects
  list(
    mu = mu,
    b_u = b_u,
    b_i = b_i,
    b_g = b_g,
    b_d = b_d,
    p = p,
    q = q
  )
}

# TODO: Tune and select lambdas
fit <- fit_als_with_latent(train_set, max_iter = 50, tol = 1e-3)
mu <- fit$mu
b_u <- setNames(fit$b_u$a, fit$b_u$userId)
b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
b_g <- setNames(fit$b_g$c, fit$b_g$genres)
b_d <- setNames(fit$b_d$d, fit$b_d$decade)

pq <- rowSums(fit$p[as.character(test_set$userId), ] * fit$q[as.character(test_set$movieId), ])
test_set$pq <- pq
resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)] + pq))
with_latent <- rmse(resid)