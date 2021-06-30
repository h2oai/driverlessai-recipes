"""Split dataset into 2 partitions based on time horizon of test set"""

# Split dataset into two partitions by time given
# time horizon (length) of last partition. With this
# approach we simply count number of unique values in temporal
# column and take the N-th from the end to be the border value.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   date_col: string - name of temporal column
#   forecast_len: integer - length of last partition measured in temporal units used in X

date_col = "Date"  # date column name
forecast_len = 7  # length in temporal units (e.g. days)

new_partition_before_name = "new_dataset_before_split_time"
new_partition_after_name = "new_dataset_after_split_time"

# determine threshold to split train and test based on forecast horizon length
dates = dt.unique(X[:, date_col])
split_date = dates[-(forecast_len + 1):, :, dt.sort(date_col)][0, 0]
test_date = dates[-1, :, dt.sort(date_col)][0, 0]

# split data to honor forecast horizon in test set
train = X[dt.f[date_col] <= split_date, :]
test = X[dt.f[date_col] > split_date, :]

return {new_partition_before_name: train, new_partition_after_name: test}
