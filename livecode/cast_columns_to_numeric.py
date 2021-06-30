"""Cast columns to numeric values. Columns may have small fraction of non-numeric values"""

# Cast columns with mostly numeric values to new numeric columns.
# Fraction of non-numeric values (per column) equal or below a threshold defined is allowed before casting.
# Non-numeric values are ignored and mapped to null value.
#
# Specification:
# Inputs:
#   X: datatable - primary data set
# Parameters:
#   columns: list - columns to cast to numeric. If None (default) then all character columns
#   threshold_fraction_non_numeric: numeric - threshold for percentage of allowed non-numeric values per column before conversion
#   in_place: bool - indicates if casting to numeric takes place in place or by adding new column
#   name_suffix: string - suffix to add to the name of new numeric column converted from the original
# Output:
#   dataset with added numeric columns derived from character columns

columns = None
threshold_fraction_non_numeric = 0.01
in_place = False
name_suffix = '_num'

new_dataset_name = "new_dataset_name_after_casting_to_num"

# check for empty frame
nrows = X.nrows
if nrows == 0:
    return None

# select columns to cast
if columns is None:
    columns = X[:, dt.f[str]].names
temp = X[:, columns]

for c in columns:
    # count nulls before casting
    null_count = temp[dt.isna(dt.f[c]), :].nrows
    # cast
    temp[c] = float
    # compute fraction of non-numeric values
    fraction_non_numeric = (temp[dt.isna(dt.f[c]), :].nrows - null_count) / temp.nrows
    # check if fraction is below pre-defined threshold
    if fraction_non_numeric <= threshold_fraction_non_numeric:
        assert temp[c].ltypes[0] == dt.ltype.real, f"wrong type: {temp[c].ltypes[0]}"
        # update or augment dataset with numeric values
        if in_place:
            X[c] = temp[c]
        else:
            X[c + name_suffix] = temp[c]

return {new_dataset_name: X}
