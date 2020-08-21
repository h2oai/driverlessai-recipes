# Parse and convert string column to date
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   col_name: str - column containing string to parse as date/time
#   date_col_name: str - new column to store parsed date/time
# Output:
#   dataset with parsed date

col_name = "dtstr"
date_col_name = "ts"

new_dataset_name = "new_dataset_name_with_parsed_date"

# remove rows with non-numierc values
del X[~X[col_name].to_pandas()[col_name].str.isnumeric(), :]

# remove rows with
del X[X[col_name].to_pandas()[col_name].str.strip().str.len() != 6, :]

# convert to pandas for string processing
X_pd = X[col_name].to_pandas()

# strip of spaces
X[col_name] = X_pd[col_name].str.strip()

# parse to date
X[date_col_name] = X_pd[col_name].str[0:4] + "-" + X_pd[col_name].str[4:] + "-01"

return {new_dataset_name: X}