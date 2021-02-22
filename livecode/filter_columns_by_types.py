# Filter only columns of certain types. Beware that column order
# changes after filtering. For more details see f-expressions in 
# datatable docs: 
# https://datatable.readthedocs.io/en/latest/manual/f-expressions.html#f-expressions
# E.g. below all integer and floating-point columns are retained 
# while the others are dropped. Because int type is followed by
# float type columns are re-shuffled so all integer columns 
# placed first and then float ones.
# For reference various data type filters are listed.
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   None explicitly, filtering columns by types inside.
# Output:
#   dataset with columns filtered by data types

from datatable import f

# character filter
column_str_filter = f[str]

new_dataset_name = "new_dataset_name_after_filtering_columns"

# integer and floating-point filter
column_int_float_filter = f[int].extend(f[float])

# numeric filter
column_numeric_filter = column_int_float_filter.extend(f[bool])

return {new_dataset_name: X[:, column_int_float_filter]}
