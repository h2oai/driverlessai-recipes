from datatable import f
# Filter only columns of certain types. Beware that column order
# changes after filtering. For more details see f-expressions in 
# datatable docs: 
# https://datatable.readthedocs.io/en/latest/manual/f-expressions.html#f-expressions
# E.g. below all integer and floating-point columns are retained 
# while the others are dropped. Because int type is followed by
# float type columns are re-shuffled so all integer columns 
# placed first and then float ones.
# For reference various data type filters are listed.

# character filter
column_str_filter = f[str]

# integer and floating-point filter
column_int_float_filter = f[int].extend(f[float])

# numeric filter
column_numeric_filter = column_int_float_filter.extend(f[bool])

return X[:, column_int_float_filter]
