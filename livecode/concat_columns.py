# Consolidate multiple columns into single text column by concatenating
# them and adding column name as a prefix.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   col_names - list of text column names to consolidate
#   txt_col_name - column name containing consolidated text
# Output:
#   dataset containing original and consolidated columns
from datatable import f, FExpr, update
import functools

col_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
txt_col_name = "toxic_consolidated"

new_dataset_name = "new_dataset_with_concat_txt_column"

concat_cols_fexpr = functools.reduce(FExpr.__add__, (col + ": " + f[col] + " " for col in col_names))
X[:, update(**{txt_col_name: concat_cols_fexpr})]

return {new_dataset_name: X}