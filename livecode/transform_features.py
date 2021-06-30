"""Transform dataset features"""

# Map and create new features by adding new columns or with in-place update.
# For example, use for mapping multi-valued key to single column or
# any other types of map (row by row) transformations.
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   transformations: map - map with datatable transformation in the form of key: value pairs where key
#                    is new / existing column name and value is datatable expression for this column.
# Output:
#   dataset containing original and transformed features
from datatable import f, isna, ifelse

transformations = {'title_with_type': f['primaryTitle'] + '-' + f['titleType'], # concatentate 2 columns
                   'startYear': ifelse(f['startYear']=='\\N', None, f['startYear']), # override empty value with NULL
                   'endYear': ifelse(f['endYear']=='\\N', None, f['endYear']), # override empty value with NULL in another column
                   'spanYears': ifelse((f['startYear']=='\\N') | (f['endYear']=='\\N'), 
                                       0, dt.int32(f['endYear']) - dt.int32(f['startYear'])) # compute the different between two columns
                  } 

X[:, dt.update(**transformations)]

return {"temp_to_delte": X}