##########################################################
# Required starting code
# Create edx and final_holdout_test sets 
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# End of required starting code
##########################################################
library(tidyverse)
library(doParallel)
library(foreach)
library(data.table)


##########################################################
# Step 1: Split edx data into training and test sets
##########################################################
edx_filtered <-  as.data.table(edx)

# Create our test and training sets by assigning 20% of the ratings made by
# each user to our test set
indexes <- split(1:nrow(edx_filtered), edx_filtered$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
  unlist(use.names = TRUE) |> sort()
temp <- edx_filtered[test_ind,]
train_set <- edx_filtered[-test_ind,]

# Make sure movies and users in the test set are also in the training set
test_set <- temp |> 
  semi_join(train_set, by = "movieId") |>
  semi_join(train_set, by = "userId")
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(temp)

########################################################
# Step 2: Do some data wrangling
########################################################
# Split year into a separate column and convert everything to the right format.
# We may not need this, but hey, it's good practice.
prep_data_columns <- function(dirty_data) {
  dirty_data |>
    extract(title, c('title', 'year'), '(.*) \\((\\d{4})\\)') |>
    mutate(
      year = as.numeric(year),
      timestamp = as_datetime(timestamp),
      genres = as.factor(genres),
      decade = year - year %% 10
    )
}

train_set <- prep_data_columns(train_set)
test_set <- prep_data_columns(test_set)

######################################################
# Step 3: Explore data and review possible effects
######################################################
# USER EFFECT
# Do users vary in how they rate movies?
fit_users <- train_set |> 
  group_by(userId) |> 
  summarize(b_u = mean(rating))
qplot(fit_users$b_u, bins = 10, color = I("black"))

# MOVIE EFFECT
# Are some movies generally rated higher than others?
fit_movies <- train_set |> 
  group_by(movieId) |> 
  summarise(b_i = mean(rating))
qplot(fit_movies$b_i, bins = 10, color = I("black"))

# GENRE EFFECT
# Are some genres rated differently than others?
fit_genres <- train_set |> 
  group_by(genres) |> 
  summarise(b_g = mean(rating))
qplot(fit_genres$b_g, bins = 10, color = I("black"))

# DECADE EFFECT
# Does the decade matter? Are movies made in the 1980s considered generally
# better or worse than movies made in the 1940s?
fit_decades <- train_set |>
  group_by(decade) |>
  summarise(b_d = mean(rating))
qplot(fit_decades$b_d, bins = 10, color = I("black"))

# It seems like all of these factors vary across ratings and might be useful
# for our predictions.


#######################################################
# Step 4: Set up utility functions and baseline
#######################################################
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

# At time of testing, baseline is 1.05972777160527

#######################################################
# Step 5: Model using only non-latent effects
#######################################################

# This went through many variations before I finally ended up with a working
# value. I tried both using constant values of lambdas and lambdas as a percentage
# of N. I went with the constant values as it was easier to understand the value
# and scale, whereas my percentages kept being orders of magnitude off.

# You'll also notice I dropped decade here. I originally included it but found
# in practice that it had a small effect on the overall accuracy and so I could
# simplify a bit.

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

fit <- fit_als_with_known_effects(train_set)
mu <- fit$mu
b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
b_u <- setNames(fit$b_u$a, fit$b_u$userId)
b_g <- setNames(fit$b_g$c, fit$b_g$genres)
 
resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres]))
with_known_effects <- rmse(resid)
# At time of testing, with_known_effects was 0.962266925225248

# For tuning, I calculated the mean and median number of ratings so I knew in
# what general neighborhood the lambdas needed to be.
b_u <- train_set |> group_by(userId) |> summarize(n=n())
mean(b_u$n) # 102.6
median(b_u$n) # 49
b_i <- train_set |> group_by(movieId) |> summarize(n=n())
mean(b_i$n) # 672
median(b_i$n) # 97
b_g <- train_set |> group_by(genres) |> summarize(n=n())
mean(b_g$n) # 8999
median(b_g$n) # 1168

# Commenting this out so the script can be run without the tuning, which takes
# a while.
# tuning_grid <- crossing(
#   lambda_u = c(50, 100, 150),
#   lambda_m = c(100, 500, 1000),
#   lambda_g = c(1000, 3000, 7000, 10000)
# )
# n <- nrow(tuning_grid)
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# results <- foreach(
#   i = 1:n,
#   .packages = c("data.table"),
#   .combine = c
# ) %dopar% {
#   tuning_grid_row <- tuning_grid[i, ]
#   fit <- fit_als_with_known_effects(train_set, lambda_u = tuning_grid_row$lambda_u, lambda_m = tuning_grid_row$lambda_m, lambda_g = tuning_grid_row$lambda_g)
#   mu <- fit$mu
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   #
#   resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres]))
#   rmse(resid)
# }
# stopImplicitCluster()
# tuning_grid[which.min(results),]

# That gave me the values I used above: 50, 100, and 10000

#######################################################
# Step 6: Model with latent effects + tuning
#######################################################
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
  # This is all the exact same as the previous function.
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
      # Get the latent factors (q) for every movie that the user has rated.
      # For example, how good are the costumes for each?
      q_i <- q[movies_by_userId[current_user], , drop = FALSE]
      resids_i <- resids_by_userId[current_user]
      # K x K matrix of movie factors, with the regularization penalty.
      # Need this in order to figure out how movies are related to one another.
      # If 18th-century novel adaptations always have good costumes, don't want
      # to end up always counting the residual for both.
      # Same as t(q_i) %*% q_i + lambda_pq_diag, but supposedly faster
      qi_by_qi <- crossprod(q_i) + lambda_pq_diag
      # Calculate how much each K factor correlates with the user's residuals
      x <- crossprod(q_i, resids_i)
      # Calculate and save the effect of the fixed movie effects on the user,
      # for example, does the user rate higher movies in the "good costume" group?
      p[u_rle$values[i], ] <- solve(qi_by_qi, x)
    }
    # Same as above, just for q values (movies).
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
# At time of testing, 0.841691104747076

# Instead of running multiple tuning attempts in parallel, I ran them sequentially,
# testing values in order: 0.5, 0.25, 1, 2.5
# Had I know it was going to take 4 attempts, I should have done them in parallel,
# but c'est la vie. 

########################################################
# Step 7: Clean and test the final holdout data
########################################################
final_holdout_test <- prep_data_columns(final_holdout_test)
pq <- rowSums(fit$p[as.character(final_holdout_test$userId), ] * fit$q[as.character(final_holdout_test$movieId), ])
final_holdout_test$pq <- pq
resid <- with(final_holdout_test, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + pq))
rmse(resid) # 0.842095