# Live code recipe for imputing all missing values
# in a dataset
# If you don't want certain data type to be filled just 
# change its filler's value to None

# numeric filler
fill_numeric = 0

# character filler
fill_char = ""

# boolean filler
fill_bool = False

for col in X.names:
    if fill_numeric is not None and \
       X[col].stype in [dt.int8, dt.int16, dt.int32, dt.int64, 
                          dt.float32, dt.float64]:
        X[dt.isna(dt.f[col]), col] = fill_numeric
    elif fill_char is not None and \
         X[col].stype in [dt.str32, dt.str64]:
        X[dt.isna(dt.f[col]), col] = fill_char
    elif fill_bool is not None and \
         X[col].stype == dt.bool8:
        X[dt.isna(dt.f[col]), col] = fill_bool
        
return X
