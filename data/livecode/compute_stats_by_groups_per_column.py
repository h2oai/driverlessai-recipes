# Compute per-column expressions (signed distance from the mean in this example) 
# for all numeric (int, float) columns with stats computed by groups and
# new column added for each original numeric feature.
# see: https://stackoverflow.com/questions/62974899/updating-or-adding-multiple-columns-with-pydatatable-in-style-of-r-datables-sd

from datatable import f, mean, update, by

group_by_column = "user_id"

aggs = {f"{col}_dist_from_mean" : mean(dt.f[col]) - f[col]
        for col in X[:, f[int].extend(f[float])].names}

X[:, update(**aggs), by(f[group_by_column])]

return X
