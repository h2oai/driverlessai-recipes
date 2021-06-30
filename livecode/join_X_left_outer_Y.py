"""Join two datasets"""

# Template for joining 2 datasets, e.g.
# one dataset with transactions and another dataset has extended set of features.
# find location of the dataset file by going to DETAILS where it's displayed
# on top under dataset name
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
#   Y_name: datatable - dataset to bind with
# Parameters:
#   join_key: string - column name to use as a key in join
# Output:
#   dataset containing all rows from both datasets

Y_name = "./tmp/gregory/136100a2-baec-11ea-b568-0ea86ce99368/DataPreviewRecipe_bf1b2__generated.bin.1593533368.9107203.bin"
join_key = "Product_Code"

new_dataset_name = "new_dataset_name_after_join"

Y = dt.fread(Y_name)

Y.key = join_key

return {new_dataset_name: X[:, :, dt.join(Y)]}