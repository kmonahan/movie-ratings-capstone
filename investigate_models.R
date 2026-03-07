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
  
  # Now use ALS to calculate the move, user, and genre effects
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
                        K = 24,        
                        lambda_u = 50,
                        lambda_m = 100,
                        lambda_g = 10000,
                        lambda_pq = 5,
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
  
  # Now use ALS to calculate the move, user, and genre effects
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
  
  # Now calculate the latent effects
  prev_loss <- rmse(dt$resid)

  # Index by user and movie.
  unique_users <- unique(dt$userId)
  unique_movies <- unique(dt$movieId)
  n_users  <- length(unique_users)
  n_movies <- length(unique_movies)
  
  # Index by position in the unique_users/unique_movies lists
  user_int  <- match(dt$userId, unique_users)
  movie_int <- match(dt$movieId, unique_movies)
  resids <- dt$resid
  
  # Pre-sort the data so maybe the code doesn't take 2 weeks to run this time.
  # Sort by users for p (user latent factors)
  order_by_user <- order(user_int)
  users_by_userId <- user_int[order_by_user]
  movies_by_userId <- movie_int[order_by_user]
  resids_by_userId <- resids[order_by_user]
  
  # Sort by movies for q (movie latent factors)
  order_by_movie <- order(movie_int)
  users_by_movieId <- user_int[order_by_movie]
  movies_by_movieId <- movie_int[order_by_movie]
  resids_by_movieId <- resids[order_by_movie]
  
  # And now, a trick Claude Code told me about: use rle() to compress the indices
  # into lengths and values, so we can easily find the starting and ending point
  # for a user or movie's ratings. This again is aimed at speeding things up so
  # the code does not take 2 weeks to run because omg I am not doing that again.
  u_rle <- rle(users_by_userId)
  user_endpoints <- cumsum(u_rle$lengths)
  user_starting_points <- c(1L, user_endpoints[-length(user_endpoints)] + 1L)
  m_rle <- rle(movies_by_movieId)
  movie_endpoints <- cumsum(m_rle$lengths)
  movie_starting_points <- c(1L, movie_endpoints[-length(movie_endpoints)] + 1L)
  
  # Initialize p and q
  p <- matrix(rnorm(n_users  * K, mean = 0, sd = 0.01), nrow = n_users,  ncol = K)
  q <- matrix(rnorm(n_movies * K, mean = 0, sd = 0.01), nrow = n_movies, ncol = K)
  
  # Calculate the diagonal and regularize it
  lambda_pq_diag <- lambda_pq * diag(K)
  
  for (iter in 1:max_iter) {
    # Do this in two loops, like we did above. First we calculate our p values,
    # holding q constant.
    for (i in 1:n_users) {
      current_user <- user_starting_points[i]:user_endpoints[i]
      q_i <- q[movies_by_userId[current_user], , drop = FALSE]
      resids_i <- resids_by_userId[current_user]
      # k x k matrix of inner products of movie factors for each movie the user has rated
      # Same as t(q_i) %*% q_i + lambda_pq_diag, but hopefully faster
      qi_by_qi <- crossprod(q_i) + lambda_pq_diag
      # correlation between latent dimension and user's residuals
      # Same as t(q_i) %*% resids_i
      x <- crossprod(q_i, resids_i)
      # Update the results
      p[u_rle$values[i], ] <- solve(qi_by_qi, x)
    }
    for (j in 1:n_movies) {
      current_movie <- movie_starting_points[j]:movie_endpoints[j]
      p_j <- p[users_by_movieId[current_movie], , drop = FALSE]
      resids_j <- resids_by_movieId[current_movie]
      pj_by_pj <- crossprod(p_j) + lambda_pq_diag
      x <- crossprod(p_j, resids_j)
      q[m_rle$values[j], ] <- solve(pj_by_pj, x)
    }
    pq <- rowSums(p[user_int, , drop = FALSE] * q[movie_int, , drop = FALSE])
    
    # Check for convergence
    loss <- rmse(resids - pq)
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
  rownames(p) <- as.character(unique_users)
  rownames(q) <- as.character(unique_movies)
  list(
    mu = mu,
    b_i = as.data.frame(b_i_dt),
    b_u = as.data.frame(b_u_dt),
    b_g = as.data.frame(b_g_dt),
    p = p,
    q = q
  )
}


fit <- fit_als_with_latent(train_set)
mu <- fit$mu
b_u <- setNames(fit$b_u$a, fit$b_u$userId)
b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
b_g <- setNames(fit$b_g$c, fit$b_g$genres)
# 
pq <- rowSums(fit$p[as.character(test_set$userId), ] * fit$q[as.character(test_set$movieId), ])
test_set$pq <- pq
resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + pq))
with_latent <- rmse(resid)
save(fit, file="rdas/fit.RData")

# 0.8855778232756