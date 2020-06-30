# Change dataset format from long to wide using pivot function
# Identify id columns and value columns to use Pandas pivot 
# function
import pandas as pd

# id column(s)
id_vars = "Product_Code"

# Name to use for the ‘variable’ column. 
var_name = 'variable'

# Name to use for the column with values
value_name = 'value'

X_pd = X.to_pandas()
X_unmelted = X_pd.pivot(index=id_vars, columns=var_name)
X_unmelted = X_unmelted[value_name].reset_index()
X_unmelted.columns.name = None
return X_unmelted
