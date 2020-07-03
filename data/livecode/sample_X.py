# Random sample of rows from X

import random

# fraction of rows to sample from X
fraction = 0.1

N = X.shape[0]
sample_size = int(N * fraction)

random.seed(0.7030)
return X[random.sample(range(N), sample_size), :]
