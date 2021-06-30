"""Extract header (top level) data and drop the rest"""

# Extract header data from detailed (line item level) dataset by filtering top level
# columns and then removing duplicate rows.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   cols_keep: list of column names - columns to keep for header dataset
#   cols_regex_keep: str - regular expression pattern for columns to keep
#   cols_remove: list of column names - columns to drop from header dataset
#   cols_regex_remove: str - regular expression pattern for column to remove
#   add_rowcount: bool - when true a column with line item row count added to the header data
# Output:
#   dataset containing only columns to keep (without removed ones) and no duplicates

cols_keep = []
cols_regex_keep = None
cols_remove = ['Measure', 'Value']
cols_regex_remove = None
add_rowcount = False

from datatable import f, count, by

new_dataset_name = "dataset_header_collapsed"

cols = list(X.names)
if cols_keep and len(cols_keep) > 0:
  cols = cols_keep
elif cols_regex_keep:
  regex = re.compile(cols_regex_keep)
  cols = list(filter(regex.match, cols))
elif cols_remove and len(cols_remove) > 0:
  [cols.remove(el) for el in cols_remove]
elif cols_regex_remove:
  regex = re.compile(cols_regex_remove)
  cols_remove = list(filter(regex.match, cols))
  [cols.remove(el) for el in cols_remove]

X = X[:, cols]

X = X[:, dt.count(), dt.by(cols)]

if not add_rowcount:
  del X[:, f[-1]]
  
return {new_dataset_name: X}
