# Add any missing Group by Date records and fill with a default value -
# additional columns will be null for the default values
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   ts_col: date/time - temporal column
#   group_by_cols: list of columns - column(s) to define groups of rows
#   target_col: list of column names - group columns
#   default_missing_value: - value to fill when missing found
# Output:
#   dataset augmented with missing data

# Column names in our dataset
ts_col = "Date"
group_by_cols = ["Store", "Dept"]
target_col = "Weekly_Sales"
default_missing_value = 0

new_dataset_name = "new_dataset_name_after_filling_ts"

# check the datatype of user-defined input variables
if not isinstance(ts_col, str):
    raise ValueError("Variable: 'ts_col' should be <str>")
if not isinstance(group_by_cols, list):
    raise ValueError("Column: 'group_by_cols' should be <list>")
if not isinstance(target_col, str):
    raise ValueError("Column: 'target_col' should be <str>")
# don't check datatype of default_missing_value because it depends on the column

# check if user-defined inputs exist in the dataset
features = list(X.names)
if ts_col not in features:
    raise ValueError("Column: '" + ts_col + "' is not present in the data set")
for _ in group_by_cols:
    if _ not in features:
        raise ValueError("Group by Column: '" + str(_) + "' is not present in the dataset")
if target_col not in features:
    raise ValueError("Column: '" + target_col + "' is not present in the data set")

# convert to pandas
df = X.to_pandas()

# order by group(s) and time
df = df.sort_values(group_by_cols + [ts_col])

# cross join of dates and groups
unique_dates = pd.DataFrame(df[ts_col].unique(), columns=[ts_col])
unique_dates['key'] = 0
unique_groups = df[group_by_cols].drop_duplicates()
unique_groups['key'] = 0
all_vals = pd.merge(unique_dates, unique_groups, how="outer").drop("key", axis=1)

# join back to the original dataset
df_filled = pd.merge(df, all_vals, how="outer")

# fill all nulls with default value - this is appropriate for TS experiments, even if there were existing nulls
df_filled[target_col] = df_filled[target_col].fillna(0)

return {new_dataset_name: df_filled}