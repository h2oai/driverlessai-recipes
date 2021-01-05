# Template to parse and split a character column using pandas str.split, 
# then assign the transposed results to individual columns. 
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   col_name: str - column containing string to split
#   sep: str - separator to use when splitting
#   transposed_col_names: list of str - column names with transposed values
# Output:
#   dataset containing new columns with transposed values

col_name = "genres"
sep = ","
transposed_col_names = ["genre1", "genre2", "genre3"]

new_dataset_name = "new_dataset_with_split_and_transposed_values_to_columns"

pdf = X[col_name].to_pandas()
transposed_values = dt.Frame(pdf[col_name].str.split(sep, expand=True))
transposed_values.names = transposed_col_names

X.cbind(transposed_values)

return {new_dataset_name: X}
