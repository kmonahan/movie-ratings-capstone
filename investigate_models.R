library(tidyverse)
library(dslabs)
library(caret)

load("rdas/train_set_clean.RData")
load("rdas/test_set_clean.RData")

# Calculate the overall average rating
mu <- mean(train_set$rating, na.rm = TRUE)

# MOVIE EFFECT
# Are some movies generally rated higher than others?
fit_movies <- train_set |> group_by(movieId) |> summarise(b_i = mean(rating))
qplot(fit_movies$b_i, bins = 10, color = I("black"))

# USER EFFECT
# Do users vary in how they rate movies?
fit_users <- train_set |> group_by(userId) |> summarize(b_u = mean(rating))
qplot(fit_users$b_u, bins = 10, color = I("black"))

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

# It seems like all of these factors are relevant to the ratings.
# However, there are probably other factors beyond what is listed here. Some
# people like particular actors. Some people like movies with good costumes.
# As we've seen here, some decades and genres are more popular than others.

# So, we need a model that accounts for variances in user and movie and various
# factors that group movies together, which include but are not limited to
# decade and genre.

# The dslabs package comes with a fit_recommender_model function for
# just such an occasion!

# But, for best results, we need to tune it. Since our data set is large and takes
# a loooooooong time to run, let's use a sample to tune our parameters before we
# fit the model on the full training set.
movies_with_enough_ratings <- train_set |> group_by(movieId) |> filter(n() >= 5) |> ungroup()
index <- sample(nrow(movies_with_enough_ratings), 100000)
train_set_sample <- train_set[index,]
index <- sample(nrow(test_set), 10000)
test_set_sample <- test_set[index,]

# We also need our RMSE function, so we can judge what parameters are most accurate.
rmse <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# And since fit_recommender_model doesn't give us a predict() function like caret
# does, let's write one.
predict_frm <- function(fit, data) {
  fit$mu + fit$a[data$userId] + fit$b[data$movieId] + rowSums(fit$p[data$userId,]*fit$q[data$movieId,])
}

tuning_params <- expand.grid(lambda_1 = 10^-(3:6), lambda_2 = 10^-c(2:5), K=c(2,4,8,16,32))
results <- apply(tuning_params, 1, function(r) {
  fit <- fit_recommender_model(train_set_sample$rating, train_set_sample$userId, train_set_sample$movieId, K = r[['K']], lambda_1 = r[['lambda_1']], lambda_2 = r[['lambda_2']], min_ratings=5)
  predict_frm(fit, test_set_sample)
})