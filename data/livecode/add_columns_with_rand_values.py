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
min_max_values = (0, 100)

new_dataset_name = "new_dataset_with_random_column"

rcol = dt.Frame(np.random.randint(min_max_values[0], min_max_values[1], size=(X.shape[0], 1)))
rcol.names = random_column_names
X.cbind(rcol)

return {new_dataset_name: X}
