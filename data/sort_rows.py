"""Sort all rows of the dataset, this helps making experiments reproducible for datasets with different row order"""

# Author: Martin Barus, martin.barus@h2o.ai
# Reviewed by: 
# Created: Oct 6th, 2020
# Version: 1.0
#
# Invoke from Details in Data Sets Page
#
# Sort all rows 

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
from datatable import join, by, f, min, max, mean, sd
import numpy as np
import pandas as pd
from itertools import permutations, combinations
from random import sample


class GroupAgg(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[str, List[str],
                                                 dt.Frame, List[dt.Frame],
                                                 np.ndarray, List[np.ndarray],
                                                 pd.DataFrame, List[pd.DataFrame]]:
        if X is None:
            return []
        return X.sort(X.keys())
