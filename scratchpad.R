# USER EFFECTS
# It's not clear if we need this part, so let's leave it for now.

# Now, let's calculate predictions using the user effects. However, we want to
# regularize the values so it's not thrown off by a user who gave something 5
# stars and then never rated anything else.
lambdas <- 10^-(3:6)

# Partially adapted from the source code for fit_recommender_model in dslabs
# https://cran.r-project.org/web/packages/dslabs/index.html

# Aliases for less typing
rating <- train_set$rating
user_id <- as.character(train_set$userId)
movie_id <- as.character(train_set$movieId)
genre <- train_set$genres

total_num_ratings <- length(rating)

user_index <- split(1:total_num_ratings, user_id)

# Data tuning to get a lambda value that we'll use for regularization
rmses_u <- sapply(lambdas, function(lambda){
  a <- sapply(user_index, function(i) sum(rating[i] - mu)/(length(i) + lambda*total_num_ratings))
  train_set |> mutate(pred=mu + a[userId]) |> summarise(rmse = calc_rmse(rating, pred)) |> pull(rmse)
})
lambda_u <- lambdas[which.min(rmses_u)]

# Calculate the user effects
b_u <- sapply(user_index, function(i) {
  sum(rating[i] - mu)/(length(i) + lambda_u*total_num_ratings)
})

# Has our accuracy improved over the baseline?
with_user_effect <- test_set |> mutate(pred=mu + b_u[userId]) |> summarize(rmse = calc_rmse(rating, pred)) |> pull(rmse)
baseline - with_user_effect

# Next, let's add the movie effect, combined with the user effect. Once again,
# we want to regularize the data first.

# TODO: Re-tune lambda
# TODO: Accuracy is going down. Why? Is it the lack of tuning?
# That does seem to be part of it, since adjusting the lambda does have some
# impact on RSME. The smaller lambda used for user effect above decreases
# accuracy compared to 0.1, so might want to revisit those lambda values.
movie_index <- split(1:total_num_ratings, movie_id)
rmses_i <- sapply(lambdas, function(lambda){
  b <- sapply(user_index, function(i) sum(residuals_after_users[i]) / (length(i) + lambda * total_num_ratings))
  train_set |> mutate(pred=mu + b_u[userId] + b[movieId]) |> summarise(rmse = calc_rmse(rating, pred)) |> pull(rmse)
})
lambda_i <- lambdas[which.min(rmses_u)]
b_i <- sapply(movie_index, function(i) {
  sum(residuals_after_users[i]) / (length(i) + lambda_i * total_num_ratings)
})
with_movie_effect <- test_set |> mutate(pred=mu + b_u[userId] + b_i[movieId]) |> summarize(rmse = calc_rmse(rating, pred)) |> pull(rmse)
with_user_effect - with_movie_effect
residuals_after_movies <- rating - mu - (b_u[user_id] + b_i[movie_id])

# Same idea for genres
genre_index <- split(1:total_num_ratings, genre)
rmses <- sapply(lambdas, function(lambda){
  c <- sapply(genre_index, function(i) sum(residuals_after_movies[i])/(length(i) + lambda*total_num_ratings))
  test_set |> mutate(pred=mu + c[genres]) |> summarise(rmse = calc_rmse(rating, pred)) |> pull(rmse)
})
lambda_g <- lambdas[which.min(rmses)]
b_g <- sapply(movie_index, function(i) {
  sum(residuals_after_movies[i], na.rm=TRUE) / (length(i) + lambda_g * total_num_ratings)
})
residuals_after_genres <- replace_na(rating - mu - (b_u[user_id] + b_i[movie_id] + b_g[genre]), 0)

# Looking at the residual calculations, it seems that in some cases, the residuals
# increased after adding genre. So let's compare prediction accuracy with and
# without genres.
without_genre <- test_set |> mutate(pred=mu + b_u[userId] + b_i[movieId]) |> summarize(rmse = calc_rmse(rating, pred)) |> pull(rmse)
with_genre <- test_set |> mutate(pred=mu + b_u[userId] + b_i[movieId] + b_g[genres]) |> summarize(rmse = calc_rmse(rating, pred)) |> pull(rmse)

# Correct, we're heading in the wrong direction. The only place where we've improved
# accuracy so far is by adding users.

# However, there are probably other factors beyond what is listed here. Some
# people like particular actors. Some people like movies with good costumes.
# As we've seen here, some decades and genres are more popular than others.

# So, we need a model that accounts for variances in user and movie and various
# factors that group movies together, which include but are not limited to
# decade and genre.