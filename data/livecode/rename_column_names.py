# Rename column name(s) in the dataset
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   column_rename_map: dictionary - dictionary mapping old column names to new ones

column_rename_map = {"oldname1": "newname1", "oldname2": "newname2"}

new_dataset_name = "new_dataset_name_after_col_rename"

X.names = column_rename_map

return {new_dataset_name: X}
