"""Add unique row id to a dataset"""

# Add (insert) new column containing unique row
# identifier. New dataset will be identical to its source
# plus inserted first column containing unique ids from 0 to N-1
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   column_name: string - new column name to store id values
# Output:
#   dataset augmented with id column

# name of column to store unique id
column_name = "rowid"

new_dataset_name = "new_dataset_name_with_id"

key_column_arg = {column_name: range(1, X.nrows + 1)}
X[:, dt.update(**key_column_arg)]

return {new_dataset_name: X}
