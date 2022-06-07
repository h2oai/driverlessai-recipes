"""Split dataset into 2 partitions based on date"""

# Split dataset into two partitions by time given
# date/time value.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   date_col: string - name of temporal column
#   split_date: date/time - temporal value to split dataset on
#   date_format: string - date format to parse date in pandas, if None then no parsing takes place
import pandas as pd

date_col = "Date"  # date column name
split_date = "2013-01-01"  # before this is train, starting with this is test
date_format = "%d-%m-%Y"  # date format to use to parse date in pandas

new_partition_before_name = "new_dataset_before_split_time"
new_partition_after_name = "new_dataset_after_split_time"

# Change date column from DataTable to Pandas
df = X[date_col].to_pandas()

if date_format:
    df[date_col] = pd.to_datetime(df[date_col], format=date_format)

train = X[df[date_col] < split_date, :]
test = X[df[date_col] >= split_date, :]

return {new_partition_before_name: train, new_partition_after_name: test}
