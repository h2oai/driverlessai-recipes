# Maps multi-nominal target (outcome) to binomial target column by
# binding new column to a dataset (new dataset will be created).
# For example, use when working with multi-nominal classifier and want 
# to see if binomial model may be preferred or compliment use case.

old_target_name = 'outcome_type'
new_target_name = 'is_adoption_target'
value_to_map_to_true = 'ADOPTION'
binary_outcomes = (False, True)
drop_old_target = False

# setup filter
binary_outcome_filter = (dt.f[old_target_name] == value_to_map_to_true)

# create new target
X[binary_outcome_filter, new_target_name] = binary_outcomes[1]
X[~binary_outcome_filter, new_target_name] = binary_outcomes[0]

# drop old target
if drop_old_target:
  del X[:, old_target_name]

return X  # return dt.Frame, pd.DataFrame, np.ndarray or a list of those
