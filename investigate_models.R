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
fit_als_with_known_effects <- function(data = train_set,
                                       lambda_u = 50,
                                       lambda_m = 100,
                                       lambda_g = 10000,
                                tol = 1e-6,
                                max_iter = 500) {
  
  # Copy the data so we can mutate it at will
  dt <- as.data.table(copy(data))
  
  # Calculate some initial numbers
  N <- nrow(dt)
  mu <- mean(dt$rating)
  
  # Initialize the residuals and the various effects
  dt[, resid := rating - mu]
  b_i_dt <- b_u_dt <- b_g_dt <- b_d_dt <- NULL
  prev_loss <- rmse(dt$resid)
  
  # Now use ALS to calculate the move, user, genre, and decade effects
  for (iter in 1:max_iter) {
    # Estimate the movie effect
    b_i_dt <- dt[, .(b = sum(resid) / (.N + lambda_m)), by = movieId]
    dt[b_i_dt, on = "movieId", bi := i.b]
    dt[, resid := rating - mu - bi]
    
    # Estimate the movie effect, given a user effect,
    b_u_dt <- dt[, .(a = sum(resid) / (.N + lambda_u)), by = userId]
    dt[b_u_dt, on = "userId", bu := i.a]
    dt[, resid := rating - mu - bi - bu]
    
    # Estimate the genre effect, given a movie effect and user effect
    b_g_dt <- dt[, .(c = sum(resid) / (.N + lambda_g)), by = genres]
    dt[b_g_dt, on = "genres", bg := i.c]
    dt[, resid := rating - mu - bi - bu - bg]
    
    # Check for convergence
    loss <- rmse(dt$resid)
    delta <- abs(loss - prev_loss) / prev_loss
    
    message(sprintf("Iteration %d: Delta = %.8f, Loss = %.6f", 
                    iter, delta, loss))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    # The second check is to prevent bad values from growing so exponentially 
    # that we overload R
    if (delta < tol | delta > 1e07)
      break
    prev_loss <- loss
  }
  
  
  # Return the regularized effects
  list(
    mu = mu,
    b_i = as.data.frame(b_i_dt),
    b_u = as.data.frame(b_u_dt),
    b_g = as.data.frame(b_g_dt)
  )
}

# fit <- fit_als_with_known_effects(train_set)
# mu <- fit$mu
# b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
# b_u <- setNames(fit$b_u$a, fit$b_u$userId)
# b_g <- setNames(fit$b_g$c, fit$b_g$genres)
# 
# resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres]))
# with_known_effects <- rmse(resid)


# ADDITIONAL MOVIE FEATURES EFFECT
# Effect of genre on movie * effect of genre on user
fit_als_with_latent <- function(data = train_set,
                        K = 16,        
                        lambda_u = 5,
                        lambda_m = 5,
                        lambda_d = 10,
                        lambda_g = 10,
                        lambda_pq = 0.02,
                        min_ratings = 20,
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
  users_with_min_ratings <- names(user_index_min)
  
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
  fit$pq <- pq
  resid <- with(fit, rating - mu)
  prev_loss <- mean(resid^2)
  
  # Now use ALS to calculate the movie and user effects and the latent effects
  for (iter in 1:max_iter) {
    # Update pq from the current p and q
    fit[, pq := rowSums(p[user_ids, , drop = FALSE] * q[movie_ids, , drop = FALSE])]
    
    # Estimate the user effect, given a movie, genre, and decade effect, and update
    # our user index
    a_vals <- fit[, .(a = sum(rating - mu - b - c - d - pq) / (.N + lambda_u)), by = userId]
    fit[a_vals, a := i.a, on = "userId"]
    
    # Now estimate the movie effect, given a user, genre, and decade effect,
    # and update our movie index
    b_vals <- fit[, .(b = sum(rating - mu - a - c - d - pq) / (.N + lambda_m)), by = movieId]
    fit[b_vals, b := i.b, on = "movieId"]
    
    # Calculate genre effect
    c_vals <- fit[, .(c = sum(rating - mu - a - b - d - pq) / (.N + lambda_g)), by = genres]
    fit[c_vals, c := i.c, on = "genres"]
    
    # Calculate decade effect
    d_vals <- fit[, .(d = sum(rating - mu - a - b - c - pq) / (.N + lambda_d)), by = decade]
    fit[d_vals, d := i.d, on = "decade"]
    
    # Now calculate the initial residual
    resid <- with(fit, rating - (mu + a + b + c + d + pq))
    
    prev_p <- p
    prev_q <- q
    
    # For K latent factors, calculate the effect using ALS again
    for (k in 1:K) {
      resid <- resid + p[user_ids, k] * q[movie_ids, k]  # add back factor k
      q[min_ratings_index, k] <- sapply(movie_index_min, function(i) {
        x <- p[user_ids[i], k]
        sum(x*resid[i])/(sum(x^2) +  lambda_pq * length(i))
      })
      
      p[users_with_min_ratings, k] <- sapply(user_index_min, function(i) {
        x <- q[movie_ids[i], k]
        sum(x*resid[i])/(sum(x^2) +  lambda_pq * length(i))
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
    # so it's MSE + the sum of the co-efficient squared.
    # Check every 5 iterations to speed things a bit.
    if (iter %% 5 == 0 || iter <= 3) {
      b_u <- a_vals
      b_i <- b_vals
      b_g <- c_vals
      b_d <- d_vals
      loss <- mean(resid^2) + (sum(b_u$a^2) * lambda_u / N) + (sum(b_i$b^2) * lambda_m / N) + (sum(b_g$c^2) * lambda_g / N) + (sum(b_d$d^2) * lambda_d / N) + ((sum(p^2) + sum(q^2)) * lambda_pq / N)
      raw_rmse <- rmse(resid)
      delta <- abs(prev_loss - loss) / (prev_loss + 1e-8)
      message(sprintf("Iteration %d: Delta = %.8f, Loss = %.6f, Raw RMSE = %.8f", 
                      iter, delta, loss, raw_rmse))
      # If the update is less than what we set as our tolerance, we're done!
      # Otherwise, it'll keep going until we hit max_iter iterations.
      # The second check is to prevent bad values from growing so exponentially 
      # that we overload R
      if (delta < tol | delta > 1e07)
        break
      prev_loss <- loss
    }
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


# fit <- fit_als_with_latent(train_set)
# mu <- fit$mu
# b_u <- setNames(fit$b_u$a, fit$b_u$userId)
# b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
# b_g <- setNames(fit$b_g$c, fit$b_g$genres)
# b_d <- setNames(fit$b_d$d, fit$b_d$decade)
# 
# pq <- rowSums(fit$p[as.character(test_set$userId), ] * fit$q[as.character(test_set$movieId), ])
# test_set$pq <- pq
# resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)] + pq))
# with_latent <- rmse(resid)
# save(fit, file="rdas/fit.RData")

# 0.8855778232756