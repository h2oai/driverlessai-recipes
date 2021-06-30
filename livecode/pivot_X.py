"""Pivot dataset"""

# Change dataset format from long to wide using pivot function.
# Identify id columns and value columns to use Pandas pivot 
# function
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   id_cols: list of columns - column to use to make new frame’s index
#   var_name: string - name to use for the 'variable' columns
#   value_name: string - name to use for the 'value' column
# Output:
#   dataset containing all rows from both datasets
import pandas as pd

# id column(s)
id_cols = "Product_Code"

# Name to use for the ‘variable’ column. 
var_name = 'variable'

# Name to use for the column with values
value_name = 'value'

new_dataset_name = "new_dataset_name_after_melt"

X_pd = X.to_pandas()
X_unmelted = X_pd.pivot(index=id_cols, columns=var_name)
X_unmelted = X_unmelted[value_name].reset_index()
X_unmelted.columns.name = None

return {new_dataset_name: X_unmelted}
