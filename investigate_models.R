library(tidyverse)

load("rdas/train_set_clean.RData")
load("rdas/test_set_clean.RData")

# Calculate the overall average rating
mu <- mean(train_set$rating, na.rm = TRUE)

# Create a function to calcuate RMSE, which is how Netflix decided on the winner
calc_rmse <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

# USER EFFECT
# Do users vary in how they rate movies?
fit_users <- train_set |> group_by(userId) |> summarize(b_u = mean(rating))
qplot(fit_users$b_u, bins = 10, color = I("black"))

# MOVIE EFFECT
# Are some movies generally rated higher than others?
fit_movies <- train_set |> group_by(movieId) |> summarise(b_i = mean(rating))
qplot(fit_movies$b_i, bins = 10, color = I("black"))

# GENRE EFFECT
# Are some genres rated differently than others?
fit_genres <- train_set |> group_by(genres) |> summarise(b_g = mean(rating))
qplot(fit_genres$b_g, bins = 10, color = I("black"))

# YEAR EFFECT
# Does the decade matter?
fit_decades <- train_set |>
  mutate(decade = year - year %% 10) |>
  group_by(decade) |>
  summarise(b_d = mean(rating))
qplot(fit_decades$b_d, bins = 10, color = I("black"))

# It seems like all of these factors vary across ratings and might be useful
# for our predictions.

# Let's start with a baseline, using the overall average, so we can see what
# improves accuracy.
baseline <- calc_rmse(test_set$rating, mu)

# Now, let's calculate predictions using the user effects. However, we want to
# regularize the values so it's not thrown off by a user who gave something 5
# stars and then never rated anything else.
lambdas <- 10^-(3:6)

# Partially adapted from the source code for fit_recommender_model in dslabs
# https://cran.r-project.org/web/packages/dslabs/index.html
total_num_ratings <- length(train_set$rating)
user_index <- split(1:total_num_ratings, train_set$userId)
rating <- train_set$rating
rmses <- sapply(lambdas, function(lambda){
  a <- sapply(user_index, function(i) sum(rating[i] - mu)/(length(i) + lambda*total_num_ratings))
  test_set |> mutate(pred=mu + a[userId]) |> summarise(rmse = calc_rmse(rating, pred)) |> pull(rmse)
})

lambda_u <- lambdas[which.min(rmses)]
b_u <- sapply(user_index, function(i) sum(rating[i] - mu)/(length(i) + lambda_u*total_num_ratings))

# Next: MOVIE EFFECT

# However, there are probably other factors beyond what is listed here. Some
# people like particular actors. Some people like movies with good costumes.
# As we've seen here, some decades and genres are more popular than others.

# So, we need a model that accounts for variances in user and movie and various
# factors that group movies together, which include but are not limited to
# decade and genre.
