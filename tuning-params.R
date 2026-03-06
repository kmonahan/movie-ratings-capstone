source("investigate_models.R")

# 10-fold cross-validation for tuning

# Movie penalty (lambda_m)
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
# min_rating <- 0
# 
# # Now that we've got the general range for our values, let's do a final grid search.
# # Trying to use the full data set with 5-fold validation took so long that I called quits
# # after 33 hours. So let's get a smaller data sample.
# 
# 
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
# 
# # Final answers:
# # Except not final, because these are wrong.
# lambda_m <- 0.00005
# lambda_u <- 0.00005
# lambda_pq <- 0.001
# K <- 5
# 
# # Randomly choose 625,000 ratings
# five_hundred_k_set <- sample_n(train_set, 625000)
# five_hundred_k_set <-  as.data.table(five_hundred_k_set)
# 
# # Create our test and training sets by assigning 20% of the ratings made by
# # each user to our test set
# indexes <- split(1:nrow(five_hundred_k_set), five_hundred_k_set$userId)
# test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
#   unlist(use.names = TRUE) |> sort()
# temp <- five_hundred_k_set[test_ind,]
# mini_train_set <- five_hundred_k_set[-test_ind,]
# 
# # Make sure movies and users in the test set are also in the training set
# mini_test_set <- temp |> 
#   semi_join(mini_train_set, by = "movieId") |>
#   semi_join(mini_train_set, by = "userId")
# removed <- anti_join(temp, mini_test_set)
# mini_train_set <- rbind(mini_train_set, removed)
# rm(temp)
# 
# lambda_pq_vals <- c(5, 10, 15, 20)
# results <- foreach(lambda_pq = lambda_pq_vals, .combine=c) %do% {
#   fit <- fit_als_with_latent(
#     mini_train_set,
#     lambda_pq = lambda_pq,
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
# 
# # And again
# five_hundred_k_set <- sample_n(train_set, 625000)
# five_hundred_k_set <-  as.data.table(five_hundred_k_set)
# 
# indexes <- split(1:nrow(five_hundred_k_set), five_hundred_k_set$userId)
# test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
#   unlist(use.names = TRUE) |> sort()
# temp <- five_hundred_k_set[test_ind,]
# mini_train_set <- five_hundred_k_set[-test_ind,]
# 
# # Make sure movies and users in the test set are also in the training set
# mini_test_set <- temp |> 
#   semi_join(mini_train_set, by = "movieId") |>
#   semi_join(mini_train_set, by = "userId")
# removed <- anti_join(temp, mini_test_set)
# mini_train_set <- rbind(mini_train_set, removed)
# rm(temp)
# 
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# lambda_pq_vals <- c(5, 7, 10)
# results <- foreach(lambda_pq = lambda_pq_vals, .combine=c, .packages = c("caret", "data.table", "tidyverse", "foreach")) %dopar% {
#   fit <- fit_als_with_latent(
#     mini_train_set,
#     lambda_pq = lambda_pq,
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
# 
# k_vals <- c(5, 7, 9)
# five_hundred_k_set <- sample_n(train_set, 625000)
# five_hundred_k_set <-  as.data.table(five_hundred_k_set)
# 
# indexes <- split(1:nrow(five_hundred_k_set), five_hundred_k_set$userId)
# test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
#   unlist(use.names = TRUE) |> sort()
# temp <- five_hundred_k_set[test_ind,]
# mini_train_set <- five_hundred_k_set[-test_ind,]
# 
# # Make sure movies and users in the test set are also in the training set
# mini_test_set <- temp |> 
#   semi_join(mini_train_set, by = "movieId") |>
#   semi_join(mini_train_set, by = "userId")
# removed <- anti_join(temp, mini_test_set)
# mini_train_set <- rbind(mini_train_set, removed)
# rm(temp)
# 
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# results <- foreach(K = k_vals, .combine=c, .packages = c("caret", "data.table", "tidyverse", "foreach")) %dopar% {
#   fit <- fit_als_with_latent(
#     mini_train_set,
#     K = K,
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
# 
# lambda_vals <- c(5, 10, 15, 25, 40)
# tuning_grid <- crossing(
#   lambda_u = lambda_vals,
#   lambda_m = lambda_vals,
#   lambda_d = lambda_vals,
#   lambda_g = lambda_vals
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
#   fit <- fit_als_with_known_effects(train_set, lambda_u = tuning_grid_row$lambda_u, lambda_m = tuning_grid_row$lambda_m, lambda_d = tuning_grid_row$lambda_d, lambda_g = tuning_grid_row$lambda_d)
#   mu <- fit$mu
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#   # 
#   resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
#   rmse(resid)
# }
# stopImplicitCluster()

b_u <- train_set |> group_by(userId) |> summarize(n=n())
# 102.6
mean(b_u$n)
# 49
median(b_u$n)
# lambda_u_vals <- c(25, 50, 100, 150, 250, 400)
# n <- length(lambda_u_vals)
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# results <- foreach(
#   i = 1:n,
#   .packages = c("data.table"),
#   .combine = c
# ) %dopar% {
#   fit <- fit_als_with_known_effects(train_set, lambda_u = lambda_u_vals[i])
#   mu <- fit$mu
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#   # 
#   resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
#   rmse(resid)
# }
# stopImplicitCluster()

# b_i <- train_set |> group_by(movieId) |> summarize(n=n())
# # 672
# mean(b_i$n)
# # 97
# median(b_i$n)
# lambda_m_vals <- c(100, 250, 500, 1000, 2000)
# n <- length(lambda_m_vals)
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# results <- foreach(
#   i = 1:n,
#   .packages = c("data.table"),
#   .combine = c
# ) %dopar% {
#   fit <- fit_als_with_known_effects(train_set, lambda_m = lambda_m_vals[i])
#   mu <- fit$mu
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#   #
#   resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
#   rmse(resid)
# }
# stopImplicitCluster()
# 
# b_g <- train_set |> group_by(genres) |> summarize(n=n())
# # 8999
# mean(b_g$n)
# # 1168
# median(b_g$n)
# lambda_g_vals <- c(1000, 3000, 7000, 15000, 30000)
# n <- length(lambda_g_vals)
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# results <- foreach(
#   i = 1:n,
#   .packages = c("data.table"),
#   .combine = c
# ) %dopar% {
#   fit <- fit_als_with_known_effects(train_set, lambda_g = lambda_g_vals[i])
#   mu <- fit$mu
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#   #
#   resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
#   rmse(resid)
# }
# stopImplicitCluster()
# 
b_d <- train_set |> group_by(decade) |> summarize(n=n())
# 717230.2
mean(b_d$n)
# 212120.5
median(b_d$n)
# lambda_d_vals <- c(10000, 50000, 150000, 400000)
# n <- length(lambda_d_vals)
# cores <- min(detectCores() - 1, 15)
# registerDoParallel(cores)
# results <- foreach(
#   i = 1:n,
#   .packages = c("data.table"),
#   .combine = c
# ) %dopar% {
#   fit <- fit_als_with_known_effects(train_set, lambda_u = 50, lambda_m = 2000, lambda_g = 1000, lambda_d = lambda_d_vals[i])
#   mu <- fit$mu
#   b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
#   b_u <- setNames(fit$b_u$a, fit$b_u$userId)
#   b_g <- setNames(fit$b_g$c, fit$b_g$genres)
#   b_d <- setNames(fit$b_d$d, fit$b_d$decade)
#   # 
#   resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres] + b_d[as.character(decade)]))
#   rmse(resid)
# }
# stopImplicitCluster()

tuning_grid <- crossing(
  lambda_u = c(50, 100, 150),
  lambda_m = c(100, 500, 1000),
  lambda_g = c(1000, 3000, 7000, 10000)
)
n <- nrow(tuning_grid)
cores <- min(detectCores() - 1, 15)
registerDoParallel(cores)
results <- foreach(
  i = 1:n,
  .packages = c("data.table"),
  .combine = c
) %dopar% {
  tuning_grid_row <- tuning_grid[i, ]
  fit <- fit_als_with_known_effects(train_set, lambda_u = tuning_grid_row$lambda_u, lambda_m = tuning_grid_row$lambda_m, lambda_g = tuning_grid_row$lambda_g)
  mu <- fit$mu
  b_i <- setNames(fit$b_i$b, fit$b_i$movieId)
  b_u <- setNames(fit$b_u$a, fit$b_u$userId)
  b_g <- setNames(fit$b_g$c, fit$b_g$genres)
  #
  resid <- with(test_set, rating - clamp(mu + b_i[as.character(movieId)] + b_u[as.character(userId)] + b_g[genres]))
  rmse(resid)
}
stopImplicitCluster()