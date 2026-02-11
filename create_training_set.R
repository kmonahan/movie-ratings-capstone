library(tidyverse)
library(data.table)

load("rdas/edx.RData")


edx_filtered <-  as.data.table(edx)

# Create our test and training sets by assigning 20% of the ratings made by
# each user to our test set
indexes <- split(1:nrow(edx_filtered), edx_filtered$userId)
test_ind <- sapply(indexes, function(ind) sample(ind, ceiling(length(ind)*.2))) |>
  unlist(use.names = TRUE) |> sort()
temp <- edx_filtered[test_ind,]
train_set <- edx_filtered[-test_ind,]

# Make sure movies and users in the test set are also in the training set
test_set <- temp |> 
  semi_join(train_set, by = "movieId") |>
  semi_join(train_set, by = "userId")
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)
rm(temp)

save(test_set, file="rdas/test_set.RData")
save(train_set, file="rdas/train_set.RData")