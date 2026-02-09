library(tidyverse)
library(doParallel)
library(foreach)
library(data.table)
library(caret)

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
fit_als_with_latent <- function(data = train_set,
                        K = 5,        
                        lambda_u = 0.00005,
                        lambda_m = 0.00005,
                        lambda_pq = 0.001,
                        min_ratings = 40,
                        tol = 1e-6,
                        max_iter = 500) {
  
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
  min_ratings_index <- which(n_item >= min_ratings)
  movie_index_min <- movie_index[min_ratings_index]
  
  movie_ids_by_length <- n_item[movie_ids]
  min_data_index <- which(movie_ids_by_length >= min_ratings)
  user_index_min <- split(min_data_index, user_ids[min_data_index])
  
  # Set initial user and movie effects of 0
  fit$a <- rep(0, N)
  fit$b <- rep(0, N)
  fit$c <- rep(0, N)
  fit$d <- rep(0, N)

  
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
  resid <- with(fit, rating - mu)
  prev_loss <- mean(resid^2)
  message(sprintf("Iteration 0: Loss = %.6f", 
                  prev_loss))
  
  # Now use ALS to calculate the movie and user effects and the latent effects
  for (iter in 1:max_iter) {
    # Estimate the user effect, given a movie, genre, and decade effect, and update
    # our user index
    fit_users <- fit |> 
      group_by(userId) |> 
      summarize(a = sum(rating - mu - b - c - d) / (n() + lambda_u * N), userId = first(userId)) |> 
      select(userId, a)
    fit <- rows_update(fit, fit_users, by = "userId")
    rm(fit_users)
    
    # Now estimate the movie effect, given a user, genre, and decade effect,
    # and update our movie index
    fit_movies <- fit |> 
      group_by(movieId) |> 
      summarize(b = sum(rating - mu - a - c - d) / (n() + lambda_m * N), movieId = first(movieId)) |> 
      select(movieId, b)
    fit <- rows_update(fit, fit_movies, by = "movieId")
    rm(fit_movies)
    
    # Calculate genre effect
    fit_genre <- fit |> 
      group_by(genres) |> 
      summarize(c = sum(rating - mu - a - b - d) / n(), genres = first(genres)) |> 
      select(genres, c)
    fit <- rows_update(fit, fit_genre, by = "genres")
    rm(fit_genre)
    
    # Calculate decade effect
    fit_decade <- fit |> 
      group_by(decade) |> 
      summarize(d = sum(rating - mu - a - b - c) / n(), decade = first(decade)) |>
      select(decade, d)
    fit <- rows_update(fit, fit_decade, by = "decade")
    rm(fit_decade)
    
    # Now calculate the initial pq
    pq <- rowSums(p[user_ids, -1, drop = FALSE] * q[movie_ids, -1, drop = FALSE])
    resid <- with(fit, rating - (mu + a + b + c + d + pq))
    
    # For K latent factors, calculate the effect using ALS again
    for (k in 1:K) {
      q[min_ratings_index, k] <- sapply(movie_index_min, function(i) {
        x <- p[user_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq * N)
      })
      
      p[, k] <- sapply(user_index_min, function(i) {
        x <- q[movie_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq * N)
      })
      
      resid <- resid - p[user_ids, k]*q[movie_ids, k]
    }
    # Update pq now that we've calculated the effects
    pq <- rowSums(p[user_ids, ] * q[movie_ids, ])
    # Update our residuals
    fit$pq <- pq
    resid <- with(fit, rating - (mu + a + b + c + d + pq))
    
    # Check for convergence using Ridge regression/L2 regularization
    # Loss function is modified to include the regularization term,
    # so it's MSE + the sum of the co-efficient squared
    loss <- mean(resid^2) + (sum(fit$a^2) * lambda_u) + (sum(fit$b^2) * lambda_m) + (sum(fit$pq^2) * lambda_pq)
    
    delta <- abs(prev_loss - loss) / (prev_loss + 1e-8)
    message(sprintf("Iteration %d: Delta = %.6f, Loss = %.6f", 
                    iter, delta, loss))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    if (delta < tol)
      break
    prev_loss <- loss
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


ptm <- proc.time()
fit <- fit_als_with_latent(train_set)
mu <- fit$mu
b_u <- setNames(fit$b_u$a, fit$b_u$userId)
b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
b_g <- setNames(fit$b_g$c, fit$b_g$genres)
b_d <- setNames(fit$b_d$d, fit$b_d$decade)

pq <- rowSums(fit$p[as.character(test_set$userId), ] * fit$q[as.character(test_set$movieId), ])
test_set$pq <- pq
resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)] + pq))
with_latent <- rmse(resid)
proc.time() - ptm

# With tolerance of 1e-6 (since min-ratings doesn't affect it)
# 2290.98 elapsed
# 0.909580541096071 RMSE

# With tolerance of 1e-8
# 2523.61 elapsed
# No change in RMSE

# With 8 factors
# 2318.87 elapsed
# No change in RMSE