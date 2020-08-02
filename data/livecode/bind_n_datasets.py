# Livecode for binding multiple datasets' rows (rbind). Datasets should have the same
# columnar structure, e.g. each file contains one month of train data.
# For more details see docs on datatable's Frame.rbind() here:
# https://datatable.readthedocs.io/en/latest/api/frame.html#datatable.Frame.rbind
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
#   files_to_bind: list of datatables - datasets to bind with
# Parameters:
#   None
# Output:
#   dataset containing all rows from primary and the list datasets
new_dataset_name = "new_dataset_name_after_rbind"

# find locations of the dataset files by going to DETAILS and look on top under dataset name
files_to_bind = ["./tmp/ff318940-c7b0-11ea-bec7-0242ac110002/e_jitter_2.csv.zip.1594937390.6086378.bin",
                 "./tmp/82b3e622-c7b7-11ea-bec7-0242ac110002/e_jitter_2.csv.gz.1594940189.8891091.bin",
                 "./tmp/c53b54e2-c7b4-11ea-bec7-0242ac110002/e_jitter_4.csv.gz.1594939013.8837.bin",
                 "./tmp/7a6ab89e-c7b0-11ea-bec7-0242ac110002/e_jitter_5.csv.zip.1594937167.4584439.bin",
                 "./tmp/7a3a14ce-c7b4-11ea-bec7-0242ac110002/e_jitter_6.csv.zip.1594938885.805977.bin"]

for file_name in files_to_bind:
    X2 = dt.fread(file_name)
    if X2.shape[1] != X.shape[1]:
        raise ValueError("Datasets must have equal number of columns")
    X.rbind(X2)

return {new_dataset_name: X}
