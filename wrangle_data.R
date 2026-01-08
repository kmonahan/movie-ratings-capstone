library(tidyverse)

load("rdas/train_set.RData")
load("rdas/test_set.RData")

# Split year into a separate column and convert everything to the right format.
# We may not need this, but hey, it's good practice.
prep_data_columns <- function(dirty_data) {
  dirty_data |>
    extract(title, c('title', 'year'), '(.*) \\((\\d{4})\\)') |>
    mutate(
      year = as.numeric(year),
      timestamp = as_datetime(timestamp),
      genres = as.factor(genres)
    )
}

train_set <- prep_data_columns(train_set)
test_set <- prep_data_columns(test_set)

save(test_set, file = "rdas/test_set_clean.RData")
save(train_set, file = "rdas/train_set_clean.RData")

# Create a wider table, with users as our rows and movies as our columns.
# We could also do it the other way around, but n_distinct indicates there are
# twice as many users as movies, so we'll use movies as our columns. This does
# give us 10,119 columns, which is a little alarming. So let's see if we can do that.
create_wider_matrix <- function(tidy_data) {
  y <- select(tidy_data, movieId, userId, rating) |>
    pivot_wider(names_from = movieId, values_from = rating)
  rnames <- y$userId
  y <- as.matrix(y[, -1])
  rownames(y) <- rnames
  y
}

# Since we're taking most of the movie info out of our matrix, create a map for the rest of the data.
create_movie_map <- function(tidy_data) {
  tidy_data |> select(movieId, title, year, genres) |> distinct(movieId, .keep_all = TRUE)
}

train_set_wide <- create_wider_matrix(train_set)
test_set_wide <- create_wider_matrix(test_set)
# Only need one map here because we already filtered out movies that aren't in both sets.
movie_map <- create_movie_map(train_set)

save(train_set_wide, file = "rdas/train_set_wide.RData")
save(test_set_wide, file = "rdas/test_set_wide.RData")
save(movie_map, file = "rdas/movie_map.RData")