# Get interesting RowIDs to search for in MLI

df = X.to_pandas()
df = df.reset_index() # create Index as a column

df = df[df["Known_Fraud"]] # in this case I want IDs where the row is Known_Fraud, but this can be any boolean

return df.head() # use PREVIEW to see and write down some interesting rows
