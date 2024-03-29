"""Calculates the distance in miles between two latitude/longitude points in space"""
# Using the Geodesic calculation in the Geopy Library
# Use sample taxi_small.csv file at https://h2o-datasets.s3.amazonaws.com/taxi_small.csv

import datatable as dt
import numpy as np
import geopy
from geopy.distance import geodesic
from h2oaicore.transformer_utils import CustomTransformer

_global_modules_needed_by_name = ['geopy==1.19.0']


class Geodesic(CustomTransformer):
    _unsupervised = True

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1, num_default_instances=1)

    @staticmethod
    def do_acceptance_test():
        return False

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        if len(X.names) >= 4:
            # modify for your dataset
            col_names_to_pick = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        else:
            return np.zeros(X.shape[0])
        try:
            x = X[:, col_names_to_pick].to_pandas()  # original line
        except KeyError:
            return np.zeros(X.shape[0])
        return x.apply(lambda x: geodesic((x[col_names_to_pick[0]], x[col_names_to_pick[1]]),
                                          (x[col_names_to_pick[2]], x[col_names_to_pick[3]])).miles,
                       axis=1)
