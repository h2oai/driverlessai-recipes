"""Delete columns based on regex pattern"""

# Delete columns with the names matching regular expression pattern.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   col_name_regex: str - regular expression pattern
# Output:
#   dataset containing only column names that do not match the pattern

col_name_regex = '\d+'

import re

new_dataset_name = "new_dataset_name_after_deleting_columns"

regex = re.compile(col_name_regex)
cols_to_delete = list(filter(regex.match, X.names))

del X[:, cols_to_delete]

return {new_dataset_name: X}
