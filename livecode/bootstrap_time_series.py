"""Bootstrap time series data (bagging and adding) - experimental"""

# Bootstrap time series (or time series bagging) by the means of randomly picking
# time series pairs and add them up, then adding resulting time series
# to the original dataset.
#
# Specification:
# Inputs:
#   X: datatable - dataset containing time series data
# Parameters:
#   tgc: list of columns that define time series groups (tgc) (IMPORTANT: only single feature tgc supported)
#   time_col: column that contains time series temporal component
#   value_col: column that contains time series value
#   size_multiplier: coefficient to increase dataset size by (1 - 100%, 2 - 200%, etc.)
# Output:
#   dataset containing time series shifted by the quantile values

import random

tgc = 'C1'  # IMPORTANT: only single feature tgc supported
time_col = 'C2'
value_col = 'C4'
size_multiplier = 1  # 1 will increase dataset size by 100%, 2 by 200%, etc.

new_dataset_name = "new_dataset_with_shifted_columns"

# generate time series unique ids using tgc
tgc_info = X[:, dt.count(), dt.by(tgc)]

X_result = X
# pick random pairs of time series ids, the add them up and add to the dataset.
for i in range(size_multiplier * tgc_info.shape[0]):
    # pick 2 random time series
    idxs = random.sample(range(tgc_info.shape[0]), 2)
    drink1 = tgc_info[idxs[0], :]
    drink2 = tgc_info[idxs[1], :]

    # we want to select smaller time series into 2d time series in case
    # the time series are not equal so left join will take care of missing data automatically
    if drink1[0, 'count'] > drink2[0, 'count']:
        drink1_name = drink1[0, tgc]
        drink2_name = drink2[0, tgc]
    else:
        drink1_name = drink2[0, tgc]
        drink2_name = drink1[0, tgc]

    t1 = X[dt.f[tgc] == drink1_name, [time_col, value_col]]
    t1.key = [time_col]
    t2 = X[dt.f[tgc] == drink2_name, [time_col, value_col]]
    tt = t2[:, :, dt.join(t1)]
    tt[:, dt.update(**{tgc: drink1_name + '+' + drink2_name,
                       'C3': 'bootstrap',  # custom dataset field - remove when re-using recipe
                       value_col: dt.f[1] + dt.f[2]})]
    # tt[:, dt.update(C1 = drink1_name + '+' + drink2_name, C3 = 'bootstrap', C4 = dt.f[1] + dt.f[2]),]

    X_result = dt.rbind(X_result, tt[:, [tgc, time_col, 'C3', value_col]])

return {new_dataset_name: X_result}
