"""Compute shift differences between consecutive rows"""

# Compute per-column difference between current and previous (shift)
# values for each time series - both by time groups (multiple time
# series) and across covariates (multivariate time series).
# Multiple time series identified by group columns while 
# covariates are explicitly assigned in `shift_cols`.
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   time_col: date/time/int - time column to order rows before the shift op
#   group_by_cols: list of column names - group columns
#   shift_cols: list of column names - columns to shift
# Output:
#   dataset augmented with shifted columns

from datatable import f, by, sort, update, shift, isna

time_col = "date"
group_by_cols = ["state"]
shift_cols = ["cases", "deaths"]

new_dataset_name = "new_dataset_name_with_shift"

# produce lag of 1 unit and add as new feature for each shift column
aggs = {f"{col}_yesterday": shift(f[col]) for col in shift_cols}
X[:, update(**aggs), sort(time_col), by(*group_by_cols)]

# update NA lags
aggs = {f"{col}_yesterday": 0 for col in shift_cols}
X[isna(f[f"{shift_cols[0]}_yesterday"]), update(**aggs)]

aggs = {f"{col}_daily": f[col] - f[f"{col}_yesterday"] for col in shift_cols}
X[:, update(**aggs), sort(time_col), by(group_by_cols)]

for col in shift_cols:
    del X[:, f[f"{col}_yesterday"]]

return {new_dataset_name: X}
