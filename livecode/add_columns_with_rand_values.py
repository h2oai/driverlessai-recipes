"""Augment dataset with columns containing random values"""

# Add one or more columns containing random integer values
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   col_count: int - number of random columns to add
#   random_column_names: List[str] - names of the columns
#   min_max_values: tuple - the reange (minimum and maximum) to pick random integers from
# Output:
#   dataset containing all rows from both datasets
import numpy as np

col_count = 1
random_column_names = ["random_columm_int"]
value_range = (0, 100)

new_dataset_name = "new_dataset_with_random_column"

if col_count != len(random_column_names):
    raise ValueError("Number of column names must be equal to number of columns.")

rcol = dt.Frame(np.random.randint(value_range[0], value_range[1], size=(X.shape[0], col_count)))
rcol.names = random_column_names
X.cbind(rcol)

return {new_dataset_name: X}
