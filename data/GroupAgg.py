"""Aggregation features on numeric columns across multiple categorical columns"""

# Author: Karthik Guruswamy, karthik.guruswamy@h2o.ai
# Reviewed by: Olivier Grellier, olivier.grellier@h2o.ai
# Created: Nov 4th, 2019
# Version: 1.0
#
# Invoke from Details in Data Sets Page
#
# Given X cats, Y numeric columns this data prep recipe creates Aggregation Features like
# GroupBy_X1_X2_<agg>_Y1
# GroupBY_X1_<agg>_Y2
# Combinations of XC2 (one to two X at a time) with an aggregation of mean, min, max, sd on Y
# These aggregations are added to each original row to enrich the data
# You can control the combinations using the variable 'max_features_in_groups' and the aggregation
# list using the variable 'aggs' below.
#
# Not always useful, but in cases of highly imbalanced data, it does come handy to create additional
# features. You really have to try :)


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
        import os
        from h2oaicore.systemutils import config

        # TUNING SECTION
        # These two variables controls the total # of extra GROUP BY columns generated and memory used
        # If the input table has a lot of columns, there could be a combinatorial explosion based on the GROUP BYs below
        # ******* DO NOT GO OVER 2 below
        
        max_features_in_groups = 1
        
        # Datatable aggregation functions. 
        aggs = [('mean', dt.mean), ('max', dt.max), ('min', dt.min), ('sd', dt.sd)]

        #####

        # Be aware that int features may also be categorical
        categorical_features = [
            X.names[i] for i, t in enumerate(X.ltypes)
            if (t in [dt.ltype.str, dt.ltype.int]) & (X[X.names[i]].nunique1()/X.shape[0] <= 0.1)
        ]

        numerical_features = [
            X.names[i] for i, t in enumerate(X.ltypes)
            if (t in [dt.ltype.real, dt.ltype.int]) & (X.names[i] not in categorical_features)
        ]

        # Check nuniques
        categorical_features = [_f for _f in categorical_features if X[_f].nunique1() < X.shape[0]]

        # Create features combinations for groupbys
        # Combinations have single features and up to max_features_in_groups tuples
        groupby_features = []
        for i in range(max_features_in_groups):
            groupby_features += list(combinations(categorical_features, i + 1))

        # Add the aggregations
        # Go through numerical features
        for num_feat in numerical_features:  # Walk through Numeric cols 1 by 1
            for group in groupby_features:  # Walk through Aggregates 1 by 1
                for agg, func in aggs:  
                    # Aggregate
                    agg_y_colname = '_'.join(group)
                    agg_feature_name = "GroupBy_" + agg_y_colname.upper() + "_" + str(agg) + "_of_" + num_feat.upper()
                    aggregated = X[:, list(group) + [num_feat]][:, {
                        agg_feature_name: func(dt.f[num_feat])
                    }, dt.by(*list(group))]
                    # Merge back
                    aggregated.key = group
                    X = X[:, :, dt.join(aggregated)]

        return X
