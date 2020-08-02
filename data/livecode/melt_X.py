# Change dataset format from wide to long using melt function
# Identify id columns and value columns to use Pandas melt 
# function
#
# Specification:
# Inputs:
#   X: datatable - primary dataset
# Parameters:
#   id_cols: list of columns - columns to use as identifier variables
#   value_col_regex: string - regular expression pattern to select value columns
#   value_cols: list of columns - columns to unpivot (melt) to use (if regex 'value_col_regex' is None)
#   var_name: string - name to use for the 'variable' columns
#   value_name: string - name to use for the 'value' column
# Output:
#   dataset containing all rows from both datasets
import pandas as pd
import re

# id column(s)
id_cols = ["Product_Code"]
# value column(s) with option of using regular expression to match
# multiple columns with similar pattern
value_col_regex = "W\d\d?"
value_cols = None
# Name to use for the ‘variable’ column. If None it uses frame.columns.name or ‘variable’.
var_name = None
# Name to use for the ‘value’ column. If None it uses default ‘value’
value_name = 'value'

if value_col_regex is not None:
    value_col_regex = re.compile(value_col_regex)
    value_cols = list(filter(value_col_regex.match, X.names))

X_pd = X.to_pandas()
return pd.melt(X_pd, id_cols=id_cols, value_vars=value_cols,
               var_name=var_name, value_name=value_name)
