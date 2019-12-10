# Author: Michelle Tanco - michelle.tanco@h2o.ai
# Last Updated: December 4th, 2019
# Purpose:  For non-time series experiments that have a time element
#           Uses hard-coded group(s), time, and target values
#           Counts number and percent of positive events in the previous days(s) by group
#           If the previous row does not exist, that is the group is new, return the mode target event
# DAI Version: 1.8.0


from h2oaicore.transformer_utils import CustomTransformer
# from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
import datatable as dt
import numpy as np
import pandas as pd
from datetime import timedelta

# Currently only for binary classification use cases
class PCTDaysTransformer(CustomTransformer):
    _regression = False
    _binary = True
    _multiclass = False
    _numeric_output = True
    _is_reproducible = True
    _included_model_classes = None
    _excluded_model_classes = None
    _allow_transform_to_modify_output_feature_names = True


    # Make usable by default
    @staticmethod
    def is_enabled():
        return True

    # Cannot use DAI testing as we have hard coded column names
    @staticmethod
    def do_acceptance_test():
        return False

    # Requires all columns in the data set
    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    # Set the hard coded values about our data set
    # TODO: rewrite this as config arguments so code does not need to be hard coded
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.col_date = "event_ts"
        self.col_group = ["customer_id"]
        self.target_positive = "Yes"

        # the number of ROWS we want to look back
        self.timespans = [30, 60, 90]

    # This function runs during training
    # TODO: don't assume rows are ordered correctly
    def fit_transform(self, X: dt.Frame, y: np.array = None):

        # values we need to accessible in other parts of DAI
        features = []
        self._output_feature_names = []
        self._feature_desc = []

        # group by and time
        cols_agg = self.col_group.copy()
        cols_agg.append(self.col_date)

        # group by, time, and y
        cols_agg_y = cols_agg.copy()
        cols_agg_y.append('col_target_bin')

        # Move to pandas
        # Pandas index is no 0 to len X - 1
        X = X.to_pandas()

        # Make a 0/1 column for if the event happened or not
        X['col_target_bin'] = (y == self.target_positive).astype(int)

        # Make sure time is time
        X[self.col_date] = pd.to_datetime(X[self.col_date])

        # Make sure we don't use target values at current row
        X = X.sort_values(self.col_date)
        X['col_target_bin'] = X.groupby(self.col_group)['col_target_bin'].shift(-1).fillna(0)
        # Go back to original index
        X.sort_index(inplace=True)

        # Create a feature for each of the requested days
        # Initialize the feature result with index set, this way features are added in correct row order even
        # if feature comes out with an unordered index
        feats_df = pd.DataFrame(index=X.index)
        for t in self.timespans:
            # Create feature name
            t_days = str(t) + "d"
            # Create the groups
            groups = X[cols_agg_y].groupby(self.col_group)
            # Run through groups
            feat_list = []
            for _key, _df in groups:
                res = _df.set_index(self.col_date)['col_target_bin'].rolling(t_days, min_periods=1).mean()
                res.index = _df.index
                feat_list.append(res)

            feats_df[t_days] = pd.concat(tuple(feat_list), axis=0)

            self._output_feature_names.append("Y%: " + t_days)
            self._feature_desc.append("Percent of Target with Event in Last " + str(t) + " Days")

        # keep the max days needed of transactions for each person to look up in transform
        time_filter = (
            X[cols_agg].groupby(self.col_group)[self.col_date].transform('max') - X[self.col_date]
        ) <= timedelta(days=max(self.timespans))
        self.lookup_df = X.loc[time_filter].sort_values(self.col_date)

        # :/
        return feats_df

    # This function runs during scoring - it will look to what happened in the training data, so may get stale fast
    # If we see a new customer we return 0 for the features, as there are no key events in their past
    def transform(self, X: dt.Frame):

        features = []

        # group by and time
        cols_agg = self.col_group.copy()
        cols_agg.append(self.col_date)

        # group by, time, and y
        cols_agg_y = cols_agg.copy()
        cols_agg_y.append('col_target_bin')

        # pandas
        X = X.to_pandas()

        # Make sure time is time
        X[self.col_date] = pd.to_datetime(X[self.col_date])

        # combine rows from training and new rows, col_target_bin will be null in new data
        lkup_and_test = pd.concat([self.lookup_df, X], axis=0, sort=False)

        for t in self.timespans:

            t_days = str(t) + "d"

            # for each transaction get % in last 30d
            # note we cannot use the "known" values from X as we wouldn't know these at prediction time
            tst_result = lkup_and_test[cols_agg_y].groupby(self.col_group).apply(
                lambda x: x.set_index(self.col_date)['col_target_bin'].shift(1).fillna(0).rolling(t_days, min_periods=1).mean()
            ).reset_index()

            # return only the rows from X
            feat = pd.merge(
                left=X,
                right=tst_result,
                on=[cols_agg],
                how='left'
            )['col_target_bin']

            features.append(feat)

        return pd.DataFrame(features).transpose()
