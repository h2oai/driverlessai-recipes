# Cast columns with mostly numeric values to new numeric columns.
# Certain percentage (threshold) of non-numeric values per column allowed before casting.
# Non-numeric values are ignored and become nulls inside new columns.
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   columns: list - columns to cast to numeric. If None (default) then all character columns
#   threshold: numeric - threshold for percentage of allowed non-numeric values per column before conversion
#   name_suffix: string - suffix to add to the name of new numeric column converted from the original
# Output:
#   dataset with added numeric columns derived from character columns

import pandas as pd
import numpy as np

# percent of values allowed to dismiss as non-numeric (missing) after conversion
columns = None
threshold = 0.01
name_suffix = '_num'

new_dataset_name = "new_dataset_name_after_casting_to_num"

# select columns to cast and convert to pandas frame
if columns is None:
    columns = X[:, dt.f[str]].names
df = X[:, columns].to_pandas()

# cast to numeric
nrows = df.shape[0]
if nrows == 0:
    return None

# check if percentage of non-numerics is low and then cast
for c in columns:
    # special care taken of negative values
    percent_of_non_numeric = sum(df[c].apply(lambda x: not (x[1:].isnumeric() if x.startswith("-") else x.isnumeric()))) / nrows
    if percent_of_non_numeric <= threshold:
        X[c + name_suffix] = pd.to_numeric(df[c], errors='coerce')

return {new_dataset_name: X}