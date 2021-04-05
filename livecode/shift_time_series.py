# Shift all time series values up by the N-th percentile value, where percentiles are
# computed per tgc values. This transformation usually applies in case when
# 0s are present in time series.
#
# Specification:
# Inputs:
#   X: datatable - dataset containing time series data
# Parameters:
#   tgc: list of columns that define time series groups (tgc)
#   value_col: column that contains time series value
#   quantile: quantile (from 0 to 1) corresponding to the values (computed per tgc) to shift time series by
#   qunatile_col_name: column name to use to store quantile values computed per tgc
# Output:
#   dataset containing time series shifted by the quantile values

import pandas as pd

value_col = 'C4'
tgc = ['C1', 'C3']
quantile = 0.1
quantile_col_name = "shift_val"

new_dataset_name = "new_dataset_with_shifted_columns"

# X = dt.fread("~/Projects/Pepsico/data/beverage_mix_forecast/latam-beverage-item-demand-time.csv")

Xpd = X[:, tgc + [value_col]].to_pandas()
Xq = Xpd.groupby(tgc).quantile(q = quantile)
Xq.rename(columns = {value_col: quantile_col_name}, inplace = True)
Xq.reset_index(inplace=True)
Xqpd = dt.Frame(Xq)
Xqpd.key = tgc

X_with_q = X[:, :, dt.join(Xqpd)]

X_with_q[:, dt.update(**{value_col+"_shifted": dt.f[value_col] + dt.f[quantile_col_name]})]

return {new_dataset_name: X_with_q}