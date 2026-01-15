library(tidyverse)

load("rdas/train_set_wide.RData")
load("rdas/test_set_wide.RData")
load("rdas/movie_map.RData")

# First, calculate the average of all ratings
mu <- mean(train_set_wide, na.rm = TRUE)

# Next, calculate the movie effect for all movies. The movie effect (b_i) is the 
# difference between the average rating for the movie (Y_i) and the overall average (mu)
b_i <- colMeans(train_set_wide - mu, na.rm = TRUE)
movies_with_effect <- data.frame(movieId = as.integer(colnames(train_set_wide)), 
                         mu = mu, b_i = b_i)
# We
# # First, find the lambda using cross-validation
# lambdas <- seq(0, 10, 0.1)
# 
# 
# sums <- colSums(y - mu, na.rm = TRUE)
# rmses <- sapply(lambdas, function(lambda){
#   b_i <-  sums / (n + lambda)
#   fit_movies$b_i <- b_i
#   left_join(test_set, fit_movies, by = "movieId") |> mutate(pred = mu + b_i) |> 
#     summarize(rmse = RMSE(rating, pred)) |>
#     pull(rmse)
# })