"""Augments dataset by computing rowwise aggregates by column groups"""

from typing import Union, List, Dict
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
from datatable import f, update, rowsum, rowmean, rowsd, rowmax, rowmin, rowfirst, rowlast, rowcount

# Compute row-wise aggregates, usually applicable when dataset contains time
# series data as enumerated columns, e.g. "PAY_1, PAY_2, ..., PAY_6" in
# the Kaggle Credit Card Default dataset.
# Adopted from the livecode recipe here:
# https://github.com/h2oai/driverlessai-recipes/blob/master/livecode/compute_rowwise_stats_by_column_groups.py
#
# Specification:
# Inputs:
#   X: datatable - primary data set with one or more groups of time series columns
# Parameters:
#   columns: list of column name prefixes - each column prefix defines a column group to compute rowwise aggregates:
#            sum, mean, standard deviation, max, min, rowfirst, rowlast. E.g Kaggle Credit Card Default dataset column
#            names are "PAY_AMT", "BILL_AMT", and "PAY_"
#   ranges: pairs of values that defines range 'from' and 'to' corresponding to each column group,
#           e.g. Kaggle Credit Card Default dataset ranges could be (1, 6), (1, 6), (2, 6)
#           Currently, only integer enumerated column groups are supported.
# Output:
#   dataset augmented with computed rowwise statistics for each column group

class AggregateRowwiseByColumnGroups(CustomData):
    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[
        str, List[str],
        dt.Frame, List[dt.Frame],
        np.ndarray, List[np.ndarray],
        pd.DataFrame, List[pd.DataFrame],
        Dict[str, str],  # {data set names : paths}
        Dict[str, dt.Frame],  # {data set names : dt frames}
        Dict[str, np.ndarray],  # {data set names : np arrays}
        Dict[str, pd.DataFrame],  # {data set names : pd frames}
    ]:
        columns = ["PAY_AMT", "BILL_AMT", "PAY_"]
        ranges = [(1, 6), (1, 6), (2, 6)]

        #ts_col_lists = [[(col + "%d") % (week) for week in range(desde, hasta)] for (col, (desde, hasta)) in
        #                zip(columns, ranges)]
        ts_col_slices = [((col + "%d") % (desde), (col + "%d") % (hasta)) for (col, (desde, hasta)) in
                         zip(columns, ranges)]

        for c, r, s in zip(columns, ranges, ts_col_slices):
            update_map = {c + "_sum": rowsum(f[s[0]:s[1]]),
                          c + "_mean": rowmean(f[s[0]:s[1]]),
                          c + "_sd": rowsd(f[s[0]:s[1]]),
                          c + "_max": rowmax(f[s[0]:s[1]]),
                          c + "_min": rowmin(f[s[0]:s[1]]),
                          c + "_range": rowmax(f[s[0]:s[1]]) - rowmin(f[s[0]:s[1]]),
                          c + "_first": rowfirst(f[s[0]:s[1]]),
                          c + "_last": rowlast(f[s[0]:s[1]]),
                          c + "_missing": (r[1] - r[0] + 1) - rowcount(f[s[0]:s[1]])
                          }
            X[:, update(**update_map)]

        return X

