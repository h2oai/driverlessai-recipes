"""Remove duplicate rows"""

# Remove duplicate rows by grouping the same rows,
# sorting them and then selecting first (1) or last (-1)
# row from each group
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   sort_cols: date/time/int/str - column(s) to order rows within each group
#   key_cols: list of column names - group columns
# Output:
#   dataset after removing dups

from datatable import by, sort

# column(s) that define duplicate rows:
key_cols = ['county', 'state', 'fips']
# column(s) (e.g. time) to sort rows within each group
sort_cols = ['date']

new_dataset_name = "new_dataset_name_after_dropping_dups"

# select last row from each group
# use 1 instead of -1 to select first row from each group
return {new_dataset_name: X[-1, :, by(*key_cols), sort(*sort_cols)]}
