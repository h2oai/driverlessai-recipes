# Split dataset into two partitions by time given
# date/time value.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   date_col: string - name of temporal column
#   split_date: date/time - temporal value to split dataset on

date_col = "Date"  # date column name
split_date = "2013-01-01"  # before this is train, starting with this is test

new_partition_before_name = "new_dataset_before_split_time"
new_partition_after_name = "new_dataset_after_split_time"

# Change date column from DataTable to Pandas
df = X[date_col].to_pandas()
train = X[df[date_col] < split_date, :]
test = X[df[date_col] >= split_date, :]

return {new_partition_before_name: train, new_partition_after_name: test}
