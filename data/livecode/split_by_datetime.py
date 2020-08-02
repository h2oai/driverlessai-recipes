# Data is called X and is a DataTable object
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   date_col: string - name of temporal column
#   split_date: date/time - temporal value to split dataset on

date_col = "Date"  # date column name
split_date = "2013-01-01"  # before this is train, starting with this is test

new_dataset_name = "new_dataset_name_after_split"

# Change date column from DataTable to Pandas
df = X[date_col].to_pandas()
train = X[df[date_col] < split_date, :]
test = X[df[date_col] >= split_date, :]

return {"my_train_data": train, "my_test_data": test}
