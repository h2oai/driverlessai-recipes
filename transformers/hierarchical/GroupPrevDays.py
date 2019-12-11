# Purpose: For transactional grouped data, calculate historical percentage of the target value
#
# Expected ML Problem: Transactional binary classification for many groups
# Example Use Cases: part failure, credit card fraud
# Overview: Takes hardcoded groups and time
#
# DAI Version: 1.8.0
# Important Note: Models with this transformer should be retrained frequently


from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import pandas as pd
from datetime import timedelta

class PCTDaysTransformer(CustomTransformer):
    # only for binary classification
    _regression = False
    _binary = True
    _multiclass = False

    _numeric_output = True
    _is_reproducible = True
    _included_model_classes = None
    _excluded_model_classes = None

    # give easy-to-understand names in the UI
    _allow_transform_to_modify_output_feature_names = True

    # Allow transformer to be used by DAI
    @staticmethod
    def is_enabled():
        return True

    # Cannot use DAI code testing as we have hard coded column names
    @staticmethod
    def do_acceptance_test():
        return False

    # Requires all columns in the data set
    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    # Set the hard coded values about our data set
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.col_date = "event_ts"
        self.col_group = ["customer_id"]
        self.target_positive = "Yes"

        # the number of DAYS or Rows we want to look back
        self.timespans = [30, 60, 90]

    # This function runs during training
    def fit_transform(self, X: dt.Frame, y: np.array = None):

        # values we need to accessible in other parts of DAI
        features = []
        self._output_feature_names = []
        self._feature_desc = []

        # list of group by and time columns
        cols_agg = self.col_group.copy()
        cols_agg.append(self.col_date)

        # list of group by, time, and y columns
        cols_agg_y = cols_agg.copy()
        cols_agg_y.append('col_target_bin')

        # Move to pandas
        # Pandas index is no 0 to len X - 1
        X = X.to_pandas()

        # Make a 0/1 column for if the event happened or not
        X['col_target_bin'] = (y == self.target_positive).astype(int)

        # Make sure time is time
        X[self.col_date] = pd.to_datetime(X[self.col_date])

        # We need to sort by date then groupby customer_id and finally shift 1 event
        X = X.sort_values(self.col_date)
        X['col_target_bin'] = X.groupby(self.col_group)['col_target_bin'].shift(1).fillna(0)
        # Go back to original index
        X.sort_index(inplace=True)

        # Create a feature for each of the requested days / rows
        # Initialize the feature result with index set, this way features are added in correct row order even
        # if feature comes out with an unordered index
        feats_df = pd.DataFrame(index=X.index)
        for t in self.timespans:
            # Create feature name
            t_days = str(t) + "d" # use t instead of t_days if you want to aggregate by rows - other units of time can be used as well
            # Create the groups
            groups = X[cols_agg_y].groupby(self.col_group)
            # Run through groups
            feat_list = []
            for _key, _df in groups:
                # Sort the group by date, this is required for the rolling per time interval to work
                _df = _df.sort_values(self.col_date)
                # Apply rolling window
                res = _df.set_index(self.col_date)['col_target_bin'].rolling(t_days, min_periods=1).mean()
                # It is not an apply and index may not be the original one
                # Therefore make sure the index is the original one
                res.index = _df.index
                # Append current result to the list
                feat_list.append(res)

            # Concat group results and assign to result dataframe
            # Since dataframe is indexed, feature is inluded in the appropriate row_order
            # i.e. pandas make sure index on both sides agree
            feats_df[t_days] = pd.concat(tuple(feat_list), axis=0)

            # Set the column names and descriptions
            self._output_feature_names.append("Y%: " + t_days)
            self._feature_desc.append("Percent of Target with Event in Last " + str(t) + " Days")

        # keep the max days needed of transactions for each person to look up in transform
        time_filter = (
            X[cols_agg].groupby(self.col_group)[self.col_date].transform('max') - X[self.col_date]
        ) <= timedelta(days=max(self.timespans))

        self.lookup_df = X.loc[time_filter].sort_values(self.col_date)

        # return the features
        return feats_df

    # This function runs during scoring - it will look to what happened in the training data, so may get stale fast
    # If we see a new customer we return 0 for the features, as there are no key events in their past
    # It is also used to transform the validation set,
    # which means validation set must be after (time wise) the training set
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
        original_index = X.index

        # Make sure time is time
        X[self.col_date] = pd.to_datetime(X[self.col_date])

        # combine rows from training and new rows, col_target_bin will be null in new data
        lkup_and_test = pd.concat([self.lookup_df, X], axis=0, sort=False).reset_index()

        feats_df = pd.DataFrame(index=lkup_and_test.index)
        for t in self.timespans:
            # Create feature name
            t_days = str(t) + "d"
            # Create the groups
            groups = lkup_and_test[cols_agg_y].groupby(self.col_group)

            # Run through groups
            feat_list = []
            for _key, _df in groups:
                # Sort the group by date, this is required for the rolling per time interval to work
                _df = _df.sort_values(self.col_date)
                # Apply rolling window for last t Days
                res = _df.set_index(self.col_date)['col_target_bin'].fillna(0).rolling(t_days, min_periods=1).mean()
                # Apply rolling window for last t events
                # res = _df.set_index(self.col_date)['col_target_bin'].fillna(0).rolling(t, min_periods=1).mean()

                # It is not an apply and index may not be the original one
                # Therefore make sure the index is the original one
                res.index = _df.index
                # Append current result to the list
                feat_list.append(res)

            # Concat group results and assign to result dataframe
            # Since dataframe is indexed, feature is inluded in the appropriate row_order
            # i.e. pandas make sure index on both sides agree
            feats_df[t_days] = pd.concat(tuple(feat_list), axis=0)

            # Set the column names and descriptions
            self._output_feature_names.append("Y%: " + t_days)
            self._feature_desc.append("Percent of Target with Event in Last " + str(t) + " Days")

        # Only return rows whose index is below self.lookup_df size
        feats_df = feats_df.loc[self.lookup_df.shape[0]:]
        feats_df.index = original_index

        return feats_df
