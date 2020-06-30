# Livecode for joining 2 datasets, e.g.
# one dataset with transactions and another dataset has extended set of features.

# find location of the dataset file by going to DETAILS where it's displayed
# on top under dataset name
Y_file_name = "./tmp/gregory/136100a2-baec-11ea-b568-0ea86ce99368/DataPreviewRecipe_bf1b2__generated.bin.1593533368.9107203.bin"
Y = dt.fread(Y_file_name)

key = ["Product_Code"]
Y.key = key

return X[:, :, dt.join(Y)]
