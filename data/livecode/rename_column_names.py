# Rename column name(s) in the dataset
import datatable as dt

column_name_map = {"oldname1": "newname1", "oldname2": "newname2"}
X.names = column_name_map

return {"new_renamed_dataset": X}
