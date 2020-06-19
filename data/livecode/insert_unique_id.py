# Livecode to add (insert) new column containing unique row
# identifier. New dataset will be identical to its source
# plus inserted first column containing unique ids from 0 to N-1

# name of column to store unique id
column_name = "rowid"

X_id = dt.Frame(list(range(X.shape[0])), names=[column_name])
X_id.cbind(X)

return X_id
