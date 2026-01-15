library(tidyverse)
library(data.table)

load("rdas/train_set_clean.RData")
load("rdas/test_set_clean.RData")

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
  group_by(decade) |>
  summarise(b_d = mean(rating))
qplot(fit_decades$b_d, bins = 10, color = I("black"))

# It seems like all of these factors vary across ratings and might be useful
# for our predictions.