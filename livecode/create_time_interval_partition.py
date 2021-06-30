"""Create dataset partition based on time interval"""

# Extract single partition based on time interval
# Data is called X and is a DataTable object
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   date_col: date/time/int - time column to order rows
#   split_date_min: lower bound of partition interval
#   split_date_max: upper bound of partition interval
# Output:
#   dataset containing partition interval

date_col = "intake_date"  # date column name
split_date_min = "2020-02-01"  # partition start date
split_date_max = "2020-02-29"  # partition end date

new_dataset_name = "new_dataset_name_partition_interval"

# Change date column from DataTable to Pandas
df = X[date_col].to_pandas()
partition = X[(df[date_col] >= split_date_min) & (df[date_col] <= split_date_max), :]

return {new_dataset_name: partition}
