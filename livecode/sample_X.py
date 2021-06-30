""" Randomly sample rows from dataset"""

# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   fraction: float - fraction of rows to sample from 'X' (must be between 0 and 1)
#   random_seed: int - random seed to control for reproducibility

import random

fraction = 0.1
random_seed = 0.7030

new_dataset_name = "new_dataset_name_after_sampling"

N = X.shape[0]
sample_size = int(N * fraction)

random.seed(random_seed)
return {new_dataset_name: X[random.sample(range(N), sample_size), :]}
