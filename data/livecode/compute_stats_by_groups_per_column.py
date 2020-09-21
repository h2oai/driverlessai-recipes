# Compute per-column expressions (signed distance from the mean in this example) 
# for all numeric (int, float) columns with stats computed by groups and
# new column added for each original numeric feature.
# see: https://stackoverflow.com/questions/62974899/updating-or-adding-multiple-columns-with-pydatatable-in-style-of-r-datables-sd
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   group_by_cols: list of column names - group columns to compute stats by
# Output:
#   dataset augmented with computed statistics

from datatable import f, by, sort, update, shift, isna, mean

group_by_cols = ["user_id"]

new_dataset_name = "new_dataset_name_with_stats"

aggs = {f"{col}_dist_from_mean": mean(dt.f[col]) - f[col]
        for col in X[:, f[int].extend(f[float])].names}

X[:, update(**aggs), by(*group_by_cols)]

return {new_dataset_name: X}
