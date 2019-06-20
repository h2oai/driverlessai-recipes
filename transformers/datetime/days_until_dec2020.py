#days_until_dec2020.py

# Transformer creates an "Days Until" feature based on date columns. "Days Until"
# being the no. of days before 2020-12-31. You can change that date by modifying code below
#
# We can also substitute the fixed data with "today's date", but the function
# will become quite mutable and will provide unpredictable results each day.
# Just pick a target date, change the name of the function after 2020 and use it as a transformer
#

 
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
import dateparser

_global_modules_needed_by_name = ['regex==2018.1.10', 'dateparser==0.7.1']

def convert_to_age(ts):
    if (type(ts) == "date"):
        time1 = dateparser.parse(ts)
        time2 = dateparser.parse("2020-12-31 11:59:59")
        #print(str(time1), str(time2), str((time2 - time1).days))
        return (time2 - time1).days
    else:
        return(-1)

class DaysUntilDec2020(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="date", min_cols=1, max_cols=1, relative_importance=1)


    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        if (X.nrows == 0):
            return X.to_pandas().iloc[:,0]
        else:           
            return X.to_pandas().apply(lambda row: convert_to_age(row[0]), axis=1)


