"""Impute missing values"""

# Imputing all missing values in a dataset
# If you don't want certain data type to be filled just 
# change its filler's value to None
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   fill_int: integer - filler for missing integer values
#   fill_float: numeric - filler for missing float values
#   fill_char:  string - filler for missing string values
#   fill_bool: bool - filler for missing logical values
# Output:
#   dataset with filled values

# integer filler
fill_int = 0
# numeric filler
fill_float = 0.0
# character filler
fill_char = ""
# boolean filler
fill_bool = False

new_dataset_name = "new_dataset_name_after_filling_missing"

replacements = [fill_value for fill_value in [fill_int, fill_float, fill_char, fill_bool] if fill_value is not None]
X.replace(None, replacements)

return {new_dataset_name: X}
