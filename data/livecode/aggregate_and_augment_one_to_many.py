# Template for augmenting data based on one-to-many relationship between datasets X (e.g. header level) 
# and Y (e.g. line item). The recipe performs:
#  1. aggregates Y by X's primary key and then 
#  2. augments X by joining aggregated Y's data.
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
from datatable import f, by, join

Y_name = "./tmp/h2oai/f91dc63e-4eda-11eb-a831-0242ac110002/imdb_episode_ratings.1609798797.6816177.bin"
primary_key_cols = ["tconst"]

new_dataset_name = "imdb_ratings_with_episodes_stats"

Y = dt.fread(Y_name)

aggs = {'episode_count': dt.count(), 
        'episode_rating_mean': dt.mean(dt.f['episodeAverageRating']),
        'episode_rating_stddev': dt.sd(dt.f['episodeAverageRating']),
        'episode_rating_min': dt.min(dt.f['episodeAverageRating']),
        'episode_rating_max': dt.max(dt.f['episodeAverageRating'])}
Y_aggs = Y[:, aggs, by(*primary_key_cols)]

X.key = primary_key_cols
result = Y_aggs[:, :, join(X)]

# reorder columns
result = result[:, list(X.names) + list(aggs.keys())]

return {new_dataset_name: result}
