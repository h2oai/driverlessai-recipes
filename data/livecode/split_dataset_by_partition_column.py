# Split dataset by partition id (column): results in as many partitions (datasets)
# as there are values in parition column
import datatable as dt

# maximum number of partition to split allowed
MAX_PARTITIONS = 10
# partition column name
partition_col_name = 'quality'

values = dt.unique(X[partition_col_name]).to_list()[0]
if len(values) > MAX_PARTITIONS:
    raise ValueError("Too many partitions to split")

result = {}
for val in values:
    partition = X[dt.f[partition_col_name] == val, :]
    result.update({"mydata_" + str(val): partition})

return result
