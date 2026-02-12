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
                                lambda_u = 0.00001,
                                lambda_m = 0.00001,
                                lambda_d = 0.01,
                                lambda_g = 0.001,
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
  
  # Set initial user and movie effects of 0
  fit$a <- rep(0, N)
  fit$b <- rep(0, N)
  fit$c <- rep(0, N)
  fit$d <- rep(0, N)
  
  resid <- with(fit, rating - mu)
  prev_loss <- mean(resid^2)
  
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
      summarize(c = sum(rating - mu - a - b - d) / (n() + lambda_g * N), genres = first(genres)) |> 
      select(genres, c)
    fit <- rows_update(fit, fit_genre, by = "genres")
    rm(fit_genre)
    
    # Calculate decade effect
    fit_decade <- fit |> 
      group_by(decade) |> 
      summarize(d = sum(rating - mu - a - b - c) / (n() + lambda_d * N), decade = first(decade)) |>
      select(decade, d)
    fit <- rows_update(fit, fit_decade, by = "decade")
    rm(fit_decade)

    resid <- with(fit, rating - (mu + a + b + c + d))
    
    # Check for convergence using Ridge regression/L2 regularization
    # Loss function is modified to include the regularization term,
    # so it's MSE + the sum of the co-efficient squared
    b_u <- fit |> group_by(userId) |> summarize(a = first(a))
    b_i <- fit |> group_by(movieId) |> summarize(b = first(b))
    b_g <- fit |> group_by(genres) |> summarize(c = first(c))
    b_d <- fit |> group_by(decade) |> summarize(d = first(d))
    loss <- mean(resid^2) + (sum(b_u$a^2) * lambda_u) + (sum(b_i$b^2) * lambda_m) + (sum(b_g$c^2) * lambda_g) + (sum(b_d$d^2) * lambda_d)
    raw_mse <- mean(resid^2)
    delta <- abs(prev_loss - loss) / (prev_loss + 1e-8)
    message(sprintf("Iteration %d: Delta = %.8f, Loss = %.6f, Raw MSE = %.8f", 
                    iter, delta, loss, raw_mse))
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
    b_u = b_u,
    b_i = b_i,
    b_g = b_g,
    b_d = b_d
  )
}

fit <- fit_als_with_known_effects(train_set)
mu <- fit$mu
b_u <- setNames(fit$b_u$a, fit$b_u$userId)
b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
b_g <- setNames(fit$b_g$c, fit$b_g$genres)
b_d <- setNames(fit$b_d$d, fit$b_d$decade)

resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
with_known_effects <- rmse(resid)


# ADDITIONAL MOVIE FEATURES EFFECT
# Effect of genre on movie * effect of genre on user
fit_als_with_latent <- function(data = train_set,
                        K = 5,        
                        lambda_u = 0.00001,
                        lambda_m = 0.00001,
                        lambda_d = 0.01,
                        lambda_g = 0.001,
                        lambda_pq = 10,
                        min_ratings = 40,
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
  fit$pq <- pq
  resid <- with(fit, rating - mu)
  prev_loss <- mean(resid^2)
  
  # Now use ALS to calculate the movie and user effects and the latent effects
  for (iter in 1:max_iter) {
    # Estimate the user effect, given a movie, genre, and decade effect, and update
    # our user index
    fit_users <- fit |> 
      group_by(userId) |> 
      summarize(a = sum(rating - mu - b - c - d - pq) / (n() + lambda_u * N), userId = first(userId)) |> 
      select(userId, a)
    fit <- rows_update(fit, fit_users, by = "userId")
    rm(fit_users)
    
    # Now estimate the movie effect, given a user, genre, and decade effect,
    # and update our movie index
    fit_movies <- fit |> 
      group_by(movieId) |> 
      summarize(b = sum(rating - mu - a - c - d - pq) / (n() + lambda_m * N), movieId = first(movieId)) |> 
      select(movieId, b)
    fit <- rows_update(fit, fit_movies, by = "movieId")
    rm(fit_movies)
    
    # Calculate genre effect
    fit_genre <- fit |> 
      group_by(genres) |> 
      summarize(c = sum(rating - mu - a - b - d - pq) / (n() + lambda_g * N), genres = first(genres)) |> 
      select(genres, c)
    fit <- rows_update(fit, fit_genre, by = "genres")
    rm(fit_genre)
    
    # Calculate decade effect
    fit_decade <- fit |> 
     group_by(decade) |> 
     summarize(d = sum(rating - mu - a - b - c - pq) / (n() + lambda_d * N), decade = first(decade)) |>
     select(decade, d)
    fit <- rows_update(fit, fit_decade, by = "decade")
    rm(fit_decade)
    
    # Now calculate the initial pq
    pq <- rowSums(p[user_ids, -1, drop = FALSE] * q[movie_ids, -1, drop = FALSE])
    resid <- with(fit, rating - (mu + a + b + c + d + pq))
    
    prev_p <- p
    prev_q <- q
    
    # For K latent factors, calculate the effect using ALS again
    # numerator: -0.40927661, denominator: 25.27217519, value: -0.01619475
    # numerator: -0.04970590, denominator: 25.17030991, value: -0.00197478
    # numerator: -0.00519023, denominator: 25.02657342, value: -0.00020739
    # numerator: -0.01971479, denominator: 25.00082604, value: -0.00078857
    # numerator: 0.04349910, denominator: 25.00069453, value: 0.00173992
    # numerator: 0.03398125, denominator: 25.01074474, value: 0.00135867
    # numerator: 0.05868307, denominator: 25.00944546, value: 0.00234644
    for (k in 1:K) {
      q[min_ratings_index, k] <- sapply(movie_index_min, function(i) {
        x <- p[user_ids[i], k]
        #message(sprintf("numerator: %.8f, denominator: %.8f, value: %.8f",  sum(x*resid[i]), (sum(x^2) + lambda_pq), sum(x*resid[i])/(sum(x^2) + lambda_pq)))
        sum(x*resid[i])/(sum(x^2) + lambda_pq)
      })
      # Damping to prevent too much oscillation
      q[, k] <- 0.7 * q[, k] + 0.3 * prev_q[, k]
      
      p[, k] <- sapply(user_index_min, function(i) {
        x <- q[movie_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq)
      })
      p[, k] <- 0.7 * p[, k] + 0.3 * prev_p[, k]
      
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
    b_u <- fit |> group_by(userId) |> summarize(a = first(a))
    b_i <- fit |> group_by(movieId) |> summarize(b = first(b))
    b_g <- fit |> group_by(genres) |> summarize(c = first(c))
    b_d <- fit |> group_by(decade) |> summarize(d = first(d))
    loss <- mean(resid^2) + (sum(b_u$a^2) * lambda_u) + (sum(b_i$b^2) * lambda_m) + (sum(b_g$c^2) * lambda_g) + (sum(b_d$d^2) * lambda_d)
    raw_mse <- mean(resid^2)
    delta <- abs(prev_loss - loss) / (prev_loss + 1e-8)
    message(sprintf("Iteration %d: Delta = %.8f, Loss = %.6f, Raw MSE = %.8f", 
                    iter, delta, loss, raw_mse))
    # If the update is less than what we set as our tolerance, we're done!
    # Otherwise, it'll keep going until we hit max_iter iterations.
    # The second check is to prevent bad values from growing so exponentially 
    # that we overload R
    if (delta < tol | delta > 1e07)
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

# Numbers of doom
# Iteration 1: Delta = 0.27261865, Loss = 0.818034, Raw MSE = 0.77592902
# Iteration 2: Delta = 0.02167575, Loss = 0.800303, Raw MSE = 0.76088064
# Iteration 3: Delta = 0.00108364, Loss = 0.799436, Raw MSE = 0.75999412
# Iteration 4: Delta = 0.00057847, Loss = 0.798973, Raw MSE = 0.75975871
# Iteration 5: Delta = 0.00040811, Loss = 0.798647, Raw MSE = 0.75962817
# Iteration 6: Delta = 0.00030076, Loss = 0.798407, Raw MSE = 0.75953410
# Iteration 7: Delta = 0.00023032, Loss = 0.798223, Raw MSE = 0.75946043
# Iteration 8: Delta = 0.00018193, Loss = 0.798078, Raw MSE = 0.75940063
# Iteration 9: Delta = 0.00014716, Loss = 0.797960, Raw MSE = 0.75935100
# Iteration 10: Delta = 0.00012119, Loss = 0.797864, Raw MSE = 0.75930910
# Iteration 11: Delta = 0.00010117, Loss = 0.797783, Raw MSE = 0.75927324
# Iteration 12: Delta = 0.00008535, Loss = 0.797715, Raw MSE = 0.75924223
# Iteration 13: Delta = 0.00007259, Loss = 0.797657, Raw MSE = 0.75921517
# Iteration 14: Delta = 0.00006215, Loss = 0.797607, Raw MSE = 0.75919139
# Iteration 15: Delta = 0.00005350, Loss = 0.797565, Raw MSE = 0.75917038
# Iteration 16: Delta = 0.00004626, Loss = 0.797528, Raw MSE = 0.75915171
# Iteration 17: Delta = 0.00004015, Loss = 0.797496, Raw MSE = 0.75913507
# Iteration 18: Delta = 0.00003497, Loss = 0.797468, Raw MSE = 0.75912018
# Iteration 19: Delta = 0.00003054, Loss = 0.797443, Raw MSE = 0.75910683
# Iteration 20: Delta = 0.00002674, Loss = 0.797422, Raw MSE = 0.75909481
# Iteration 21: Delta = 0.00002347, Loss = 0.797403, Raw MSE = 0.75908399
# Iteration 22: Delta = 0.00002064, Loss = 0.797387, Raw MSE = 0.75907422
# Iteration 23: Delta = 0.00001819, Loss = 0.797372, Raw MSE = 0.75906538
# Iteration 24: Delta = 0.00001605, Loss = 0.797360, Raw MSE = 0.75905737
# Iteration 25: Delta = 0.00001418, Loss = 0.797348, Raw MSE = 0.75905011
# Iteration 26: Delta = 0.00001255, Loss = 0.797338, Raw MSE = 0.75904351
# Iteration 27: Delta = 0.00001112, Loss = 0.797329, Raw MSE = 0.75903751
# Iteration 28: Delta = 0.00000986, Loss = 0.797322, Raw MSE = 0.75903205
# Iteration 29: Delta = 0.00000875, Loss = 0.797315, Raw MSE = 0.75902707
# Iteration 30: Delta = 0.00000777, Loss = 0.797308, Raw MSE = 0.75902253
# Iteration 31: Delta = 0.00000691, Loss = 0.797303, Raw MSE = 0.75901838
# Iteration 32: Delta = 0.00000614, Loss = 0.797298, Raw MSE = 0.75901459
# Iteration 33: Delta = 0.00000547, Loss = 0.797294, Raw MSE = 0.75901113
# Iteration 34: Delta = 0.00000487, Loss = 0.797290, Raw MSE = 0.75900796
# Iteration 35: Delta = 0.00000433, Loss = 0.797286, Raw MSE = 0.75900505
# Iteration 36: Delta = 0.00000386, Loss = 0.797283, Raw MSE = 0.75900238
# Iteration 37: Delta = 0.00000344, Loss = 0.797281, Raw MSE = 0.75899994
# Iteration 38: Delta = 0.00000306, Loss = 0.797278, Raw MSE = 0.75899769
# Iteration 39: Delta = 0.00000273, Loss = 0.797276, Raw MSE = 0.75899563
# Iteration 40: Delta = 0.00000243, Loss = 0.797274, Raw MSE = 0.75899373
# Iteration 41: Delta = 0.00000217, Loss = 0.797272, Raw MSE = 0.75899199
# Iteration 42: Delta = 0.00000193, Loss = 0.797271, Raw MSE = 0.75899038
# Iteration 43: Delta = 0.00000172, Loss = 0.797269, Raw MSE = 0.75898891
# Iteration 44: Delta = 0.00000154, Loss = 0.797268, Raw MSE = 0.75898754
# Iteration 45: Delta = 0.00000137, Loss = 0.797267, Raw MSE = 0.75898629
# Iteration 46: Delta = 0.00000122, Loss = 0.797266, Raw MSE = 0.75898513
# Iteration 47: Delta = 0.00000109, Loss = 0.797265, Raw MSE = 0.75898406
# Iteration 48: Delta = 0.00000097, Loss = 0.797264, Raw MSE = 0.75898308

save(fit, file="rdas/fit.RData")

# 0.8855778232756