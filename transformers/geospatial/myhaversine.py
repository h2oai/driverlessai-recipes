"""Computes miles between first two *_latitude and *_longitude named columns in the data set"""
#
# Custom transformer: MyHaversine
#
# Computes miles between first two lat, long columns in the data set. Column names should have
# strings 'latitude' and 'longitude' in it
# Example:
# pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude
#
# Author(s: Karthik Guruswamy, Principal SE, H2O.ai
#           Tom Ott, Principal SE, H2O.ai

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
from datatable import f
import numpy as np
import math


def distance(lat1, lon1, lat2, lon2):
    # radius = 6371 # km
    radius = 3959  # miles
    # 3959 * 5280 # radius in feet
    # 6371 * 1000 # radius in meters

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c
    return d


class MyHaversine(CustomTransformer):
    _unsupervised = True

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols="all", max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        col_names = X.names
        print(col_names)
        lat = []
        long = []
        for col in col_names:
            if col.find("latitude") > -1:
                lat.append(col)
            if (col.find("longitude") > -1):
                long.append(col)

        if (len(lat) == 2 and len(long) == 2):
            return X.to_pandas().apply(lambda row: \
                                           distance(row[lat[0]], \
                                                    row[long[0]], \
                                                    row[lat[1]], \
                                                    row[long[1]]), \
                                       axis=1)
        else:
            return X.to_pandas().iloc[:, 0]
