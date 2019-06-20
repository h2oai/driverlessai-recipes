# Created by Github User: Tomott12345 and kguruswamy
# This Custom Transformer calculates the distance in miles between to latitude/longitude points in space
# Using the Geodesic caclulation in the Geopy Library
# Use sample taxi_small.csv file in /data

import datatable as dt
import numpy as np
import geopy
from geopy.distance import geodesic
from h2oaicore.transformer_utils import CustomTransformer

_global_modules_needed_by_name = ['geopy==1.19.0']


class Geodesic(CustomTransformer):

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1, num_default_instances=1)

    @staticmethod
    def do_acceptance_test():
        return False

    # Train

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    # Validate

    def transform(self, X: dt.Frame):

        col_names = X.names

        if len(col_names) > 3:
            col_names_to_pick = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        else:
            # col_names_to_pick:  = 'lat1', 'long1', 'lat2', 'long2' #these strings maybe modified to your column names
            pass

        X = X[:, col_names_to_pick]

        x = X.to_pandas()  # original line

        return x.apply(lambda x: geodesic((x[col_names_to_pick[0]], x[col_names_to_pick[1]]),
                                          (x[col_names_to_pick[2]], x[col_names_to_pick[3]])).miles,
                       axis=1)
