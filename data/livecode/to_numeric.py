# Make almost numeric fields numeric by ignoring strings and representing them as missing
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   threshold: numeric - threshold for percantage of allowed non-numeric values per column before conversion
#   name_suffix: string - suffix to add to the name of new numeric column converted from the original
# Output:
#   dataset with added numeric columns derived from character columns

import pandas as pd
import numpy as np

# percent of values allowed to dismiss as non-numeric (missing) after conversion
threshold = 0.01
name_suffix = '_to_numeric'

df = X[:, dt.f[str]].to_pandas()

columns = list(df)
for c in columns:
    fi = df[c].apply(lambda x: x.isnumeric())
    if sum(fi==False)/len(fi) <= threshold:
        X[c + name_suffix] = pd.to_numeric(df[c], errors='coerce')

return X