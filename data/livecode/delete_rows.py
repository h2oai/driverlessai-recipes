# Delete rows based on certain condition.
# In this case delete rows where certain column contains null values.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   col_name: str - column name
# Output:
#   dataset containing only rows with non-null values in designated column

new_dataset_name = "new_dataset_name_after_deleting_rows"

col_name = 'name'

del X[dt.isna(dt.f[col_name]), :]

return {new_dataset_name: X}