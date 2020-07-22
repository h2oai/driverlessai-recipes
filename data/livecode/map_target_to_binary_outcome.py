# Maps multi-nominal target (outcome) to binomial target column by
# binding new column to a dataset (new dataset will be created).
# For example, use when working with multi-nominal classifier and want 
# to see if binomial model may be preferred or compliment use case.

old_target_name = 'Score'
new_target_name = 'isScorePos'
value_to_map_to_true = 3
binary_outcomes = (False, True)
drop_old_target = False

# update with new target
X[:, dt.update(isScorePos = dt.ifelse(dt.f[old_target_name] >= value_to_map_to_true, 
                                      binary_outcomes[1], 
                                      binary_outcomes[0]))]


# drop old target
if drop_old_target:
  del X[:, old_target_name]

return X  # return dt.Frame, pd.DataFrame, np.ndarray or a list of those
