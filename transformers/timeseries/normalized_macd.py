from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np


class NormalizedMACDTransformer(CustomTimeSeriesTransformer):
    """
    Moving Average Convergence Divergence
    Based on the difference between a long and a short moving average
    Should be used on Financial instruments or at least positively valued features
    """

    @staticmethod
    def get_default_properties():
        """This recipe is for time series
        It expects at least one data column along with the time group columns
        To make that happen you need to specify col_type = "all"
        this will ensure you get the TimeGroupColumns along with the data
        Having the TimeGroupColumns allows us to do the groupby ourselves and apply specific methods on the features
        """
        return dict(col_type="all", min_cols=2, max_cols="all", relative_importance=1)

    @staticmethod
    def normalized_macd(sig, short=7, long=28):
        short_mean = sig.rolling(window=short, min_periods=1).mean()
        long_mean = sig.rolling(window=long, min_periods=1).mean()
        # MACD is expected to be used on financial instruments
        # whose values are positives
        # This can result in np.inf values so add an epsilon
        return (long_mean - short_mean) / (long_mean + short_mean + 1e-10)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        """Nothing to be trained in this transformer so just call transform method"""
        return self.transform(X)

    def transform(self, X: dt.Frame):
        """Transform features once grouped by Time Group Columns (TGC)"""
        # With the col_type set to "all", X can contain text features
        # So restrict to int float and bool types
        # This is easily done in datatable
        X = X[:, [int, float, bool]]
        # If after the filtering there are no features left then just return a zero valued features
        if X.ncols == 0:
            return np.zeros(X.nrows)

        # Move to pandas to use the apply method
        X = X.to_pandas()

        group_cols = [_f for _f in self.tgc if _f != self.time_column]

        # Check if we really have any group columns available
        if len(group_cols) == 0:
            # Apply MACD directly on the available features but drop the time column
            features = [_f for _f in X.columns if _f != self.time_column]
            return self.normalized_macd(X[features])

        # Get the data columns, i.e. not the group columns or time column
        col = np.setdiff1d(X.columns, self.tgc)
        if len(col) > 0:
            # Groupby by the TGC and apply normalized MACD to the data
            # Pandas.apply ios not time effective so should move this to data table
            res = X.groupby(group_cols)[col].apply(self.normalized_macd)

            res.index = X.index
            return res
        else:
            return np.zeros(X.nrows)
