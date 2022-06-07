"""Augment data based on one-to-many relationship by means of aggregate and join"""

# Template for augmenting data based on one-to-many relationship. For example,
# dataset X representing customers and dataset Y containing their orders.
# Then recipe will:
#  1. aggregate Y by X's primary key and then
#  2. left join X with Y and augment X with aggregated Y's data.
#
# Specification:
# Inputs:
#   X: datatable - primary header level dataset
#   Y_name: string - Y dataset location to aggregate and augment X
# Parameters:
#   primary_key_cols: list of strings - column name(s) representing X primary key
# Output:
#   dataset containing all rows from both datasets

# find location of the dataset file by going to DETAILS where it's displayed
# on top under dataset name
from datatable import f, g, by, join

Y_name = "./tmp/h2oai/f91dc63e-4eda-11eb-a831-0242ac110002/imdb_episode_ratings.1609798797.6816177.bin"
primary_key_cols = ["tconst"]

new_dataset_name = "imdb_ratings_with_episodes_stats"

# read Y frame containing data to aggregate
Y = dt.fread(Y_name)

# define aggregate computations/features
aggs = {'episode_count': dt.count(),
        'episode_rating_mean': dt.mean(dt.f['episodeAverageRating']),
        'episode_rating_stddev': dt.sd(dt.f['episodeAverageRating']),
        'episode_rating_min': dt.min(dt.f['episodeAverageRating']),
        'episode_rating_max': dt.max(dt.f['episodeAverageRating'])}

# run aggregates
Y_aggs = Y[:, aggs, by(*primary_key_cols)]

# join (augment) X with aggregated features
X.key = primary_key_cols
result = Y_aggs[:, :, join(X)]

# augment Y_aggs with the keys that are found in 
# (until datatable supports only left outer join with existing unique key on the joined table we need this workaround)
Y_aggs_keys = dt.unique(Y_aggs[:, primary_key_cols])
Y_aggs_keys.key = primary_key_cols
X_missing_in_Y_keys = X[g[-1] == None, f[:], join(Y_aggs_keys)]
result.rbind(X_missing_in_Y_keys, force=True)

# assign key for validation purpose only
result.key = primary_key_cols

# reorder columns
result = result[:, list(X.names) + list(aggs.keys())]

return {new_dataset_name: result}
