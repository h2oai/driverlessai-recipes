"""Split dataset by partition column - will result in as many partitions as there are values in the partition column"""

# Split dataset by partition id (column): results in as many partitions (datasets)
# as there are values in parition column
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   partition_col_name: string - column name identifying which partition row belongs to
#   dataset_name_prefix: string - prefix to use in the names for new datasets
#   MAX_PARTITIONS: int - maximum number of partition datasets to create
import datatable as dt

# maximum number of partition to split allowed
MAX_PARTITIONS = 10
# partition column name
partition_col_name = 'quality'
# partitioned datasets name prefix
dataset_name_prefix = "mydata_"

values = dt.unique(X[partition_col_name]).to_list()[0]
if len(values) > MAX_PARTITIONS:
    raise ValueError("Too many partitions to split")

result = {}
for val in values:
    partition = X[dt.f[partition_col_name] == val, :]
    result.update({dataset_name_prefix + str(val): partition})

return result