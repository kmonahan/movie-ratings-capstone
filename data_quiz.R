library(tidyverse)
load("rdas/edx.RData")

dim(edx)

sum(edx$rating == 0)
sum(edx$rating == 3)
n_distinct(edx$movieId)
n_distinct(edx$userId)

genres <- c('Drama', 'Comedy', 'Thriller', 'Romance')
ratings_count <- sapply(genres, function(genre) {
  edx |> filter(str_detect(genres, genre)) |> nrow()
})
t(ratings_count)

edx |> group_by(movieId) |> summarize(title = first(title), ratings = n()) |> top_n(20)

edx |> group_by(rating) |> summarize(times_given = n()) |> arrange(desc(times_given))