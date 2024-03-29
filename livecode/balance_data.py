"""Downsample majority class in imbalanced dataset"""

# Create a sampled dataset for imbalanced use cases - probably not for modeling but
# can be nice to better see trends in MLI PDP plots
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   target_col: str - usually target column to use when balancing data
#   times: int - how much to downsample majority class: in number of times size of minority class
#   random_seed: int - random seed to control for reproducibility
# Output:
#   dataset with downsampled majority class

from sklearn.utils import resample
import pandas as pd
from datatable import f, count, sort, by, min, max, rbind

# parameters
target_col = "Known_Fraud"
times = 5
random_seed = 123

new_dataset_name = "new_dataset_name_with_downsampled_majority"

# counts by target groups
g = X[:, {"count": count()}, by(target_col)]
if not g.shape[1] == 2:
    raise ValueError("Not a binary target - target column must contain exactly 2 values.")

# find sizes and target values for minority and majority class partitions
n_minority = g[:, min(f.count)][0, 0]
n_majority = g[:, max(f.count)][0, 0]
target_minority = g[f.count == n_minority, target_col][0, 0]
target_majority = g[f.count == n_majority, target_col][0, 0]

# validate that times indeed downsamples majority class
if times * n_minority >= n_majority:
    raise ValueError(
        "Downsampling coefficient `times` is too large: downsampled dataset results in inflated majority class.")

# downsample with pandas frame
df_majority = X[f[target_col] == target_majority, :].to_pandas()
df_majority_downsampled = resample(df_majority,
                                   replace=False,
                                   n_samples=n_minority * times,
                                   random_state=random_seed)

return {new_dataset_name:
            rbind(X[f[target_col] == target_minority, :],
                  dt.Frame(df_majority_downsampled))}
