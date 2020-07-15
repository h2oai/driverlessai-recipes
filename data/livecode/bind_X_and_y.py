# Livecode for binding 2 datasets with the same number of rows, e.g.
# one dataset with features and another dataset has target.
# Recipe won't perform any joins/mapping but rather stitch
# 2 datasets' rows together.

# find location of the dataset file by going to DETAILS where it's displayed
# on top under dataset name
y_file_name = "./tmp/877bbf3a-6557-11ea-a946-0242ac110002/class_f50k_X_y_train.csv.1584123720.3251045.bin"
y = dt.fread(y_file_name)
if y.shape[0] != X.shape[0]:
    raise ValueError("Datasets must have equal number of rows")

# target column name
target_col = "y"
if not target_col in y.names:
    raise ValueError("Target column not found in y dataset")

X.cbind(y[target_col])

return X
