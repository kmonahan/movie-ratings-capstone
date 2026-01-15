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
      genres = as.factor(genres),
      decade = year - year %% 10
    )
}

train_set <- prep_data_columns(train_set)
test_set <- prep_data_columns(test_set)

save(test_set, file = "rdas/test_set_clean.RData")
save(train_set, file = "rdas/train_set_clean.RData")