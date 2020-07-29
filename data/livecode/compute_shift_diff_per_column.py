# Compute per-column difference between current and previous (shift)
# values for each time series - both by time groups (multiple time
# series) and across covariates (multivariate time series).
# Multiple time series identified by group columns while 
# covariates are explicitly assigned in `shift_cols`.

from datatable import f, by, sort, update, shift, isna

time_col = "date"
shift_cols = ["cases", "deaths"]
group_by_cols = ["state"]
us_states = dt.fread("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv")

# produce lag of 1 unit and add as new feature for each shift column
aggs = {f"{col}_yesterday" : shift(f[col]) for col in shift_cols}
X[:, update(**aggs), sort(time_col), by(group_by_cols)]

# update NA lags
aggs = {f"{col}_yesterday" : 0 for col in shift_cols}
X[isna(f[f"{shift_cols[0]}_yesterday"]), update(**aggs)]

aggs = {f"{col}_daily" : f[col] - f[f"{col}_yesterday"] for col in shift_cols}
X[:, update(**aggs), sort(time_col), by(group_by_cols)]

for col in shift_cols:
    del X[:, f[f"{col}_yesterday"]]

return X
