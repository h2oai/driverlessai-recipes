"""Group by and aggregate dataset"""

# Compute aggregates with per-column expressions (means and sums in this example)
# for numeric (int, float) columns by groups.
# New frame contains computed aggregates of the columns and group by columns.
# see: compute_stats_by_groups_per_column.py and
#      https://stackoverflow.com/questions/62974899/updating-or-adding-multiple-columns-with-pydatatable-in-style-of-r-datables-sd
#
# Specification:
# Inputs:
#   X: datatable - primary data set
#   mean_columns: list of str - columns to compute means on (change to the aggregates and columns of your choice)
#   sum_columns: list of str - columns to compute sums on (change to the aggregates and columns of your choice)
# Parameters:
#   group_by_cols: list of column names - group columns to aggregate by
# Output:
#   dataset with computed aggregates and groups

from datatable import f, by, sort

group_by_cols = ["user_id"]
mean_columns = ['k_value', 'a_value']
sum_columns = ['s_value']

new_dataset_name = "new_dataset_name_with_aggregates"

# change/define aggregate functions: datatable aggregate functions to use: count, sum, min, max, mean, sd, median,
# and also first, last when used with sort.
aggs_mean = [dt.mean(dt.f[col]) for col in mean_columns]
aggs_sum = [dt.sum(dt.f[col]) for col in sum_columns]
aggs = aggs_mean + aggs_sum

result = X[:, aggs, by(*group_by_cols)]

return {new_dataset_name: result}
