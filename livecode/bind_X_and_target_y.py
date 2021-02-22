# Template for binding dataset and target from another dataset with the same number of rows,
# e.g. one dataset has features and another contains target.
# Recipe won't perform any joins/mapping but rather stitch 2 datasets together into wider dataset with
# the same number of rows and columns from 1st dataset plus target from another.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
#   y_name: string - location of the dataset containing target value
# Parameters:
#   target_col: string - target name
# Output:
#   dataset containing all rows from both datasets

# find location of the dataset file by going to DETAILS where it's displayed
# on top under dataset name
y_name = "./tmp/877bbf3a-6557-11ea-a946-0242ac110002/class_f50k_X_y_train.csv.1584123720.3251045.bin"
# target column name
target_col = "y"

new_dataset_name = "new_dataset_name_after_cbind_y"

y = dt.fread(y_name)
if y.shape[0] != X.shape[0]:
    raise ValueError("Datasets must have equal number of rows.")

if not target_col in y.names:
    raise ValueError(f"Target column '{target_col}' not found in y dataset.")

if target_col in X.names:
    raise ValueError(f"Target column '{target_col}' found in X dataset.")

X.cbind(y[target_col])

return {new_dataset_name: X}
