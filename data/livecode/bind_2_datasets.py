# Livecode for binding 2 datasets' rows (rbind). Datasets should have the same
# columnar structure, e.g. train dataset and test dataset (with target present).
# For more details see docs on datatable's Frame.rbind() here:
# https://datatable.readthedocs.io/en/latest/api/frame.html#datatable.Frame.rbind
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
#   X2_name: datatable - dataset to bind with
# Parameters:
#   None
# Output:
#   dataset containing all rows from both datasets

# find location of the dataset file by going to DETAILS where it displayed
# on top under dataset name
X2_name = "./tmp/877bbf3a-6557-11ea-a946-0242ac110002/class_f50k_X_y_train.csv.1584123720.3251045.bin"

new_dataset_name = "new_dataset_name_after_rbind"

# read and validate 2d dataset
X2 = dt.fread(X2_name)
if X2.shape[1] != X.shape[1]:
    raise ValueError("Datasets must have equal number of columns")

# bind datasets
X.rbind(X2)

return {new_dataset_name: X}
