library(tidyverse)
library(data.table)

load("rdas/edx.RData")


# Keep only users who have rated at least 100 movies, so we have data to 
# draw on.
# edx_filtered <- as.data.table(edx)[, if (.N >= 100) .SD, by = userId]
edx_filtered <-  as.data.table(edx)

# Create our test and training sets by assigning 20% of the ratings made by
# each user to our test set
indexes <- split(1:nrow(edx_filtered), edx_filtered$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
  unlist(use.names = TRUE) |> sort()
test_set <- edx_filtered[test_ind,]
train_set <- edx_filtered[-test_ind,]

# Remove any movies that are not in BOTH the test and training sets
test_set <- test_set |> 
  semi_join(train_set, by = "movieId")
train_set <- train_set |> 
  semi_join(test_set, by = "movieId")

save(test_set, file="rdas/test_set.RData")
save(train_set, file="rdas/train_set.RData")