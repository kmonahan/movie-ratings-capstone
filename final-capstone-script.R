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

# I originally tested this and then deleted the function
# as I focused on the model with latent effects, so what's
# here is a reconstruction of the function based on what
# I used for latent effects. Hence we're jumping straight
# to the lambda values I ended up using.

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
    message(sprintf("Iteration %d: Delta = %.6f, Loss = %.6f, Raw MSE = %.6f", 
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

# At time of testing, with_known_effects is 0.883318173989364

#######################################################
# Step 6: Model with latent effects + tuning
#######################################################

# I know not just from the course textbook/videos but also from general domain
# knowledge that there is more that goes into how people rate movies than is
# captured in the table columns. My two sisters judge movies based on costuming
# and whether the cinematography  is "pretty". I'll bail if the world-building
# doesn't make sense. Looking for latent factors is the best way I have at getting
# at that.

fit_als_with_latent <- function(data = train_set,
                                K = 5,        
                                lambda_u = 0.00001,
                                lambda_m = 0.00001,
                                lambda_d = 0.01,
                                lambda_g = 0.001,
                                lambda_pq = 1e-6,
                                min_ratings = 40,
                                tol = 1e-8,
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
  fit$pq <- pq
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
    for (k in 1:K) {
      q[min_ratings_index, k] <- sapply(movie_index_min, function(i) {
        x <- p[user_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq * N)
      })
      # Damping to prevent too much oscillation
      q[, k] <- 0.7 * q[, k] + 0.3 * prev_q[, k]
      
      p[, k] <- sapply(user_index_min, function(i) {
        x <- q[movie_ids[i], k]
        sum(x*resid[i])/(sum(x^2) + lambda_pq * N)
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
    loss <- mean(resid^2) + (sum(b_u$a^2) * lambda_u) + (sum(b_i$b^2) * lambda_m) + (sum(b_g$c^2) * lambda_g) + (sum(b_d$d^2) * lambda_d) + ((sum(p^2) + sum(q^2)) * lambda_pq)
    raw_mse <- mean(resid^2)
    delta <- abs(prev_loss - loss) / (prev_loss + 1e-8)
    message(sprintf("Iteration %d: Delta = %.6f, Loss = %.6f, Raw MSE = %.6f", 
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

# I did a lot of tuning and trial-and-error here, changing one of the params
# and testing again with my test_set. The commit history at 
# https://github.com/kmonahan/movie-ratings-capstone covers a lot of the journey.

# Some of my tuning code. Leaving this commented out because you do NOT want to
# run this. It takes way too long.

# # Movie penalty (lambda_m)
# folds <- createFolds(train_set$rating,
#                      k = 10,
#                      list = TRUE,
#                      returnTrain = TRUE)
# sets <- lapply(folds, function(fold) {
#   train_set[-fold, ]
# })
# lambdas <- 10^-(3:6)
# cores <- min(detectCores() - 1, 10)
# registerDoParallel(cores)
# results <- foreach(lambda = lambdas) %do% {
#   validations <- foreach(
#     set = sets,
#     .packages = c("caret", "data.table", "tidyverse"),
#     .verbose = TRUE,
#     .combine = c
#   ) %dopar% {
#     set_index <- split(1:nrow(set), set$userId)
#     # Assign 10% of each user's rating to the test set
#     test_index <- sapply(set_index, function(ind)
#       sample(ind, floor(length(ind) * .1))) |>
#       unlist(use.names = TRUE) |> sort()
#     mini_test_set <- set[test_index, ]
#     mini_train_set <- set[-test_index, ]
#     # Remove any movies that are not in BOTH the test and training sets
#     mini_test_set <- mini_test_set |>
#       semi_join(mini_train_set, by = "movieId")
#     mini_train_set <- mini_train_set |>
#       semi_join(mini_test_set, by = "movieId")
#     
#     fit <- fit_als_with_latent(
#       mini_train_set,
#       max_iter = 50,
#       tol = 1e-4,
#       lambda_m = lambda,
#       min_ratings = 0
#     )
#     mu <- fit$mu
#     b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#     b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#     b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#     b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#     pq <- rowSums(fit$p[as.character(mini_test_set$userId), ] * fit$q[as.character(mini_test_set$movieId), ])
#     mini_test_set$pq <- pq
#     resid <- with(mini_test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[as.character(genres)] + b_d[as.character(decade)] + pq))
#     rmse(resid)
#   }
#   mean(validations)
# }
# stopImplicitCluster()
# lambda_m <- 1e-05
# 
# # User penalty (lambda_u)
# folds <- createFolds(train_set$rating,
#                      k = 10,
#                      list = TRUE,
#                      returnTrain = TRUE)
# sets <- lapply(folds, function(fold) {
#   train_set[-fold, ]
# })
# lambdas <- 10^-(3:6)
# cores <- min(detectCores() - 1, 10)
# registerDoParallel(cores)
# results <- foreach(lambda = lambdas) %do% {
#   validations <- foreach(
#     set = sets,
#     .packages = c("caret", "data.table", "tidyverse"),
#     .verbose = TRUE,
#     .combine = c
#   ) %dopar% {
#     set_index <- split(1:nrow(set), set$userId)
#     # Assign 10% of each user's rating to the test set
#     test_index <- sapply(set_index, function(ind)
#       sample(ind, floor(length(ind) * .1))) |>
#       unlist(use.names = TRUE) |> sort()
#     mini_test_set <- set[test_index, ]
#     mini_train_set <- set[-test_index, ]
#     # Remove any movies that are not in BOTH the test and training sets
#     mini_test_set <- mini_test_set |>
#       semi_join(mini_train_set, by = "movieId")
#     mini_train_set <- mini_train_set |>
#       semi_join(mini_test_set, by = "movieId")
#     
#     fit <- fit_als_with_latent(
#       mini_train_set,
#       max_iter = 50,
#       tol = 1e-4,
#       lambda_m = lambda_m,
#       lambda_u = lambda,
#       min_ratings = 0
#     )
#     mu <- fit$mu
#     b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#     b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#     b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#     b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#     pq <- rowSums(fit$p[as.character(mini_test_set$userId), ] * fit$q[as.character(mini_test_set$movieId), ])
#     mini_test_set$pq <- pq
#     resid <- with(mini_test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[as.character(genres)] + b_d[as.character(decade)] + pq))
#     rmse(resid)
#   }
#   mean(validations)
# }
# stopImplicitCluster()
# lambda_u <- 1e-05
# 
# 
# # Latent factors penalty (lambda_pq)
# folds <- createFolds(train_set$rating,
#                      k = 10,
#                      list = TRUE,
#                      returnTrain = TRUE)
# sets <- lapply(folds, function(fold) {
#   train_set[-fold, ]
# })
# lambdas <- 10^-(2:5)
# cores <- min(detectCores() - 1, 10)
# registerDoParallel(cores)
# results <- foreach(lambda = lambdas, .combine = c) %do% {
#   validations <- foreach(
#     set = sets,
#     .packages = c("caret", "data.table", "tidyverse"),
#     .verbose = TRUE,
#     .combine = c
#   ) %dopar% {
#     set_index <- split(1:nrow(set), set$userId)
#     # Assign 10% of each user's rating to the test set
#     test_index <- sapply(set_index, function(ind)
#       sample(ind, floor(length(ind) * .1))) |>
#       unlist(use.names = TRUE) |> sort()
#     mini_test_set <- set[test_index, ]
#     mini_train_set <- set[-test_index, ]
#     # Remove any movies that are not in BOTH the test and training sets
#     mini_test_set <- mini_test_set |>
#       semi_join(mini_train_set, by = "movieId")
#     mini_train_set <- mini_train_set |>
#       semi_join(mini_test_set, by = "movieId")
#     
#     fit <- fit_als_with_latent(
#       mini_train_set,
#       max_iter = 50,
#       tol = 1e-4,
#       lambda_m = lambda_m,
#       lambda_u = lambda_u,
#       lambda_pq = lambda,
#       K = 8,
#       min_ratings = 0
#     )
#     mu <- fit$mu
#     b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#     b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#     b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#     b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#     pq <- rowSums(fit$p[as.character(mini_test_set$userId), ] * fit$q[as.character(mini_test_set$movieId), ])
#     mini_test_set$pq <- pq
#     resid <- with(mini_test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[as.character(genres)] + b_d[as.character(decade)] + pq))
#     rmse(resid)
#   }
#   mean(validations)
# }
# stopImplicitCluster()
# lambda_pq <- 0.001
# 
# # Min Ratings
# # We need a larger data set here, so let's see how it does if we treat the
# # train_set as the entire set
# min_ratings <- c(5, 10, 20, 40)
# set_index <- split(1:nrow(train_set), train_set$userId)
# # Assign 10% of each user's rating to the test set
# test_index <- sapply(set_index, function(ind)
#   sample(ind, floor(length(ind) * .1))) |>
#   unlist(use.names = TRUE) |> sort()
# mini_test_set <- train_set[test_index, ]
# mini_train_set <- train_set[-test_index, ]
# # Remove any movies that are not in BOTH the test and training sets
# mini_test_set <- mini_test_set |>
#   semi_join(mini_train_set, by = "movieId")
# mini_train_set <- mini_train_set |>
#   semi_join(mini_test_set, by = "movieId")
# 
# cores <- min(detectCores() - 1, 10)
# registerDoParallel(cores)
# results <- foreach(
#   min_rating = min_ratings,
#   .combine = c,
#   .packages = c("caret", "data.table", "tidyverse"),
#   .verbose = TRUE
# ) %dopar% {
#   fit <- fit_als_with_latent(
#     mini_train_set,
#     max_iter = 50,
#     tol = 1e-4,
#     min_ratings = min_rating
#   )
#   mu <- fit$mu
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#   pq <- rowSums(fit$p[as.character(mini_test_set$userId), ] * fit$q[as.character(mini_test_set$movieId), ])
#   mini_test_set$pq <- pq
#   resid <- with(mini_test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[as.character(genres)] + b_d[as.character(decade)] + pq))
#   rmse(resid)
# }
# stopImplicitCluster()
# # Same result regardless of min ratings??
# # Maybe we don't need a minimum then.
# # Will use the larger value for now, since maybe that'll speed things up.
# # It's also possible that with this large of a data set, there are no movies
# # with few enough ratings to fall below the min anyway.
# min_rating <- 40
# 
# # Now that we've got the general range for our values, let's do a final grid search.
# # Trying to use the full data set with 5-fold validation took so long that I called quits
# # after 33 hours. So let's get a smaller data sample.
# 
# lambda_m_vals <- c(1e-05, 3e-05, 5e-05)
# lambda_u_vals <- c(1e-05, 3e-05, 5e-05)
# lambda_pq_vals <- c(1e-03, 1e-04, 1e-05)
# lambda_d_vals <- c(0.01, 0.001, 0.0001)
# lambda_g_vals <- c(0.001, 0.0001)
# k_vals <- c(5, 7, 9)
# 
# # Start with users who have rated at least 100 movies and movies rated by those
# # users at least 5 times
# filtered_train_set <- train_set |> 
#   group_by(userId) |>
#   filter(n() >= 100) |>
#   ungroup() |>
#   group_by(movieId) |> 
#   filter(n() >= 5) |>
#   ungroup()
# 
# # Randomly choose 100,000 ratings from that group
# one_hundred_k_set <- sample_n(filtered_train_set, 100000)
# 
# samples <- createResample(
#   one_hundred_k_set$rating,
#   times = 5,
#   list = TRUE
# )
# sets <- lapply(samples, function(sample) {
#   one_hundred_k_set[-sample, ]
# })
# 
# # Measuring with proc.time, the smaller sample takes 239 seconds to test one
# # row. So even if running tests in parallel doesn't speed anything up (though it
# # should), testing all 81 rows would be 19359 seconds or 5-6 hours. Which is a
# # long time, but better than me giving up after 33. So let's give it a go!
# 
# # Much better! Under 2 hours!
# 
# tuning_grid <- crossing(
#   lambda_m = lambda_m_vals,
#   lambda_pq = lambda_pq_vals,
#   lambda_d = lambda_d_vals,
#   K = k_vals
# )
# n <- nrow(tuning_grid)
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# 
# 
# results <- foreach(
#   i = 1:n,
#   .packages = c("caret", "data.table", "tidyverse", "foreach"),
#   .combine = c
# ) %dopar% {
#   tuning_grid_row <- tuning_grid[i, ]
#   validations <- foreach(set = sets,
#                          .verbose = TRUE,
#                          .combine = c) %do% {
#                            set_index <- split(1:nrow(set), set$userId)
#                            # Assign 20% of each user's rating to the test set
#                            test_index <- sapply(set_index, function(ind)
#                              sample(ind, floor(length(ind) * .2))) |>
#                              unlist(use.names = TRUE) |> sort()
#                            mini_test_set <- set[test_index, ]
#                            mini_train_set <- set[-test_index, ]
#                            # Remove any movies that are not in BOTH the test and training sets
#                            mini_test_set <- mini_test_set |>
#                              semi_join(mini_train_set, by = "movieId")
#                            mini_train_set <- mini_train_set |>
#                              semi_join(mini_test_set, by = "movieId")
#                            # Remove any users that are now only in the test set
#                            mini_test_set <- mini_test_set |>
#                              semi_join(mini_train_set, by = "userId")
#                            
#                            fit <- fit_als_with_latent(
#                              mini_train_set,
#                              max_iter = 25,
#                              tol = 1e-4,
#                              lambda_m = tuning_grid_row$lambda_m,
#                              lambda_u = tuning_grid_row$lambda_m,
#                              lambda_pq = tuning_grid_row$lambda_pq,
#                              lambda_d = tuning_grid_row$lambda_d,
#                              lambda_g = tuning_grid_row$lambda_d,
#                              K = tuning_grid_row$K,
#                              min_ratings = 0
#                            )
#                            mu <- fit$mu
#                            b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#                            b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#                            b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#                            b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#                            pq <- rowSums(fit$p[as.character(mini_test_set$userId), ] * fit$q[as.character(mini_test_set$movieId), ])
#                            mini_test_set$pq <- pq
#                            resid <- with(mini_test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[as.character(genres)] + b_d[as.character(decade)] + pq))
#                            rmse(resid)
#                          }
#   mean(validations)
# }
# stopImplicitCluster()
# 
# tuning_grid[which.min(results),]

# When I plugged the tuning_grid results into my function, I found that they
# weren't quite right and I had to go back to some of the results from my previous
# tests. (And the lambda_pq ended up being WAY off, which I didn't realize until
# I compared the calculations with my non-latent-effects model). I suspect this
# is because I reduced my sample size too much in an attempt to not have the code
# run for multiple days. Given a much smaller N, the smaller lambda values likely
# resulted in no regularization at all.

# At this point, I did a lot of iterations of re-running the same code with a
# value tweaked, often stopping after 8 or so iterations if it was clear we were
# heading in the wrong direction. (It didn't help that there were bugs in my
# code that I found along the way). Eventually, I landed on lambda values that
# worked decently when tested on my test set, using the un-commented out code
# above.

########################################################
# Step 7: Clean and test the final holdout data
########################################################
final_holdout_test <- prep_data_columns(final_holdout_test)