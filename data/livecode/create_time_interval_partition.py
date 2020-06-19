# Extract single partition based on time interval
# Data is called X and is a DataTable object

date_col = "intake_date"  # date column name
split_date_min = "2020-02-01" # partition start date
split_date_max = "2020-02-29" # partition end date

# Change date column from DataTable to Pandas
df = X[date_col].to_pandas()
partition = X[(df[date_col] >= split_date_min) & (df[date_col] <= split_date_max), :]

return {"dallas_shelter_survdata_Feb_2020": partition}
