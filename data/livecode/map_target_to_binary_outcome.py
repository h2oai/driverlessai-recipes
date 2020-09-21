# Maps multi-nominal target (outcome) to binomial target column by
# binding new column to a dataset (new dataset will be created).
# For example, use when working with multi-nominal classifier and want 
# to see if binomial model may be preferred or compliment use case.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   target_name: string - target column name
#   new_target_name: string - new target column name
#   value_to_map_to_true: value - target values that maps to binary positive (true) outcome
#   binary_outcomes: tuple - pair of binary outcomes to sue for new target
#   drop_old_target: bool - if true then drop old target column
# Output:
#   dataset containing all rows from both datasets

target_name = 'Score'
new_target_name = 'isScorePos'
value_to_map_to_true = 3
binary_outcomes = (False, True)
drop_old_target = False

# update with new target
X[:, dt.update(isScorePos=dt.ifelse(dt.f[target_name] >= value_to_map_to_true,
                                    binary_outcomes[1],
                                    binary_outcomes[0]))]

# drop old target
if drop_old_target:
    del X[:, target_name]

return X  # return dt.Frame, pd.DataFrame, np.ndarray or a list of those
