# Change dataset format from wide to long using melt function
# Identify id columns and value columns to use Pandas melt 
# function
import pandas as pd
import re

# id column(s)
id_vars = ["Product_Code"]
# Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.
var_name = None

# value column(s) with option of using regular expression to match
# multiple columns with similar pattern
reg_expr = re.compile("W\d\d?")
if reg_expr is not None:
  value_cols = list(filter(reg_expr.match, X.names))
else:
  value_cols = ["W0", "W1", "W2", "W3"]

# Name to use for the ‘value’ column. If None it uses default ‘value’
value_name = 'value'

X_pd = X.to_pandas()
return pd.melt(X_pd, id_vars=id_vars, value_vars=value_cols, 
               var_name=var_name, value_name=value_name)
