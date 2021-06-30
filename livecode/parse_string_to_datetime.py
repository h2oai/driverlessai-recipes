"""Parse string column and convert to date time type"""

# Parse and convert string column to date.
# This example converts string in the format `MMMMYY` to `MMMM-YY-DD`.
# Please adjust code to the format you expect to find in your data.
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

# remove rows with less or greater than 6 characters long (not 'MMMMYY' format)
del X[X[col_name].to_pandas()[col_name].str.strip().str.len() != 6, :]

# convert to pandas for string processing
X_pd = X[col_name].to_pandas()

# strip of both leading and trailing spaces
X_pd[col_name] = X_pd[col_name].str.strip()

# parse to date
X[date_col_name] = X_pd[col_name].str[0:4] + "-" + X_pd[col_name].str[4:] + "-01"

return {new_dataset_name: X}
