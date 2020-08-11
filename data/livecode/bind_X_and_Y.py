# Template for binding columns from 2 datasets with the same number of rows.
# Recipe won't perform any joins/mapping but rather stitch 2 datasets together into wider dataset with
# the same number of rows and columns from both.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
#   Y_name: datatable - dataset to bind with
# Parameters:
#   None
# Output:
#   dataset containing all rows from both datasets

# find location of the dataset file by going to DETAILS where it's displayed
# on top under dataset name
Y_name = "./tmp/877bbf3a-6557-11ea-a946-0242ac110002/class_f50k_X_y_train.csv.1584123720.3251045.bin"

new_dataset_name = "new_dataset_name_after_cbind"

Y = dt.fread(Y_name)
if Y.shape[0] != X.shape[0]:
    raise ValueError("Datasets must have equal number of rows")

X.cbind(Y)

return {new_dataset_name: X}