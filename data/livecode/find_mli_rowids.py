# Get interesting RowIDs to search for in MLI
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   target_col: list of column names - group columns
# Output:
#   dataset with selected rows and ids

target_col = "Known_Fraud"

df = X.to_pandas()
df = df.reset_index()  # create Index as a column

df = df[df[target_col]] # in this case I want IDs where the row is Known_Fraud, but this can be any boolean

return df.head()  # use PREVIEW to see and write down some interesting rows