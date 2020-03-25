"""Add any missing Group by Date records and fill with a default value - additional columns will be null for the default values"""

# Column names in our dataset
ts_column 		 	 	= "Date"
group_by_columns 	 	= ["Store", "Dept"]
target_column		 	= "Weekly_Sales"
default_missing_value 	= 0

# check the datatype of user-defined input variables
if not isinstance(ts_column, str):
	raise ValueError("Variable: 'ts_column' should be <str>")
if not isinstance(group_by_columns, list):
	raise ValueError("Column: 'group_by_columns' should be <list>")
if not isinstance(target_column, str):
	raise ValueError("Column: 'target_column' should be <str>")
# don't check datatype of default_missing_value because it depends on the column

# check if user-defined inputs exist in the dataset
features = list(X.names)
if ts_column not in features:
	raise ValueError("Column: '" + ts_column + "' is not present in the data set")
for _ in group_by_columns:
	if _ not in features:
		raise ValueError("Group by Column: '" + str(_) + "' is not present in the dataset")
if target_column not in features:
	raise ValueError("Column: '" + target_column + "' is not present in the data set")

# convert to pandas
df = X.to_pandas()

# order by group(s) and time
df = df.sort_values(group_by_columns + [ts_column])

# cross join of dates and groups
unique_dates = pd.DataFrame(df[ts_column].unique(), columns=[ts_column])
unique_dates['key'] = 0
unique_groups = df[group_by_columns].drop_duplicates()
unique_groups['key'] = 0
all_vals = pd.merge(unique_dates, unique_groups, how="outer").drop("key", axis=1)

# join back to the original dataset
df_filled = pd.merge(df, all_vals, how="outer")

# fill all nulls with default value - this is appropriate for TS experiments, even if there were existing nulls
df_filled[target_column] = df_filled[target_column].fillna(0)

return df_filled
