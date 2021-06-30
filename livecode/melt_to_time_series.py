"""Melt (unpivot) time series in wide format to H2O standard long time series format"""

# Melt time series in wide format (single row) into long format supported
# by DAI: a row represents a point in time (lag) so a column represents
# a time series values.
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   id_cols: list of columns - columns to use as identifier variables
#   time_series_col_name: string - name of time series columns
#   time_series_new_name: string - name to use in melted data
#   timestamp_col_name: string - column name for time values
# Output:
#   dataset with melted time series

import re
import pandas as pd

# for testing
# X = dt.fread("~/Downloads/benchmark-bond-trade-price-challenge/train.csv")

id_cols = ["id", "bond_id", "weight", "current_coupon", "time_to_maturity", "is_callable", "reporting_delay"]
time_series_col_name = 'trade_price_last'
time_series_new_name = 'trade_price'
timestamp_col_name = 'ts'

new_dataset_name = "new_dataset_name_with_melted_time_series"

# create regex to match all columns containing time series values
# assuming that they have the same prefix and count as
# suffix corresponding to the lag
time_series_col_regex = time_series_col_name + '\d+'
time_series_regex = re.compile(time_series_col_regex)
value_cols = list(filter(time_series_regex.match, X.names))

# metl using pandas
X_pd = X.to_pandas()
X_melt = pd.melt(X_pd, id_vars=id_cols, value_vars=value_cols, var_name=timestamp_col_name,
                 value_name=time_series_new_name)

# replace times series lag names with integers as time
X_melt[timestamp_col_name] = X_melt[timestamp_col_name].str.replace(time_series_col_name, '')
X_melt[timestamp_col_name] = pd.to_numeric(X_melt[timestamp_col_name])

return {new_dataset_name: X_melt}
