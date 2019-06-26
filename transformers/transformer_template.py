"""Template base class for a custom transformer recipe."""

import datatable as dt
import numpy as np
import pandas as pd

_global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file


class CustomTransformer(DataTableTransformer):
    """Base class for a custom transformer recipe that can be specified externally to Driverless AI.

    Note: Experimental API, will most likely change in future versions.
    """

    """By default, we want a transformer to work with all types of supervised problems.
    Please disable the problem types it cannot support."""
    _regression = True   # y has shape (N,) and is of numeric type, no missing values
    _binary = True       # y has shape (N,) and can be numeric or string, cardinality 2, no missing values
    _multiclass = True   # y has shape (N,) and can be numeric or string, cardinality 3+, no missing values

    """Specify whether the transformer creates numeric output data or not. If not, some models might not be able 
    to consume the transformer's features."""
    _numeric_output = True

    """Specify whether the transformer is expected to create reproducible results. If disabled, transformer might be 
    skipped for experiments run in reproducible mode."""
    _is_reproducible = True

    """Optional list of included/excluded models, specified by their booster string 
    (e.g., _included_model_classes = ['CatBoostModel'], _excluded_model_classes = ['tensorflow'])"""
    _included_model_classes = None   # List[str]
    _excluded_model_classes = None   # List[str]

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    _modules_needed_by_name = []  # List[str], e.g., ["mypackage==1.3.37"]

    """Optional name to show for this transformer during experiment and in results"""
    _display_name = NotImplemented  # str

    """Expert settings for optimal hardware usage"""
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = False   # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = False  # if enabled, can get access to multiple GPUs for single transformer (experimental)

    """Whether this transformer supports MOJO creation.
    If set to True, requires implementation of write_to_mojo() method."""
    _mojo = False

    @staticmethod
    def is_enabled():
        """Return whether recipe is enabled. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

    @staticmethod
    def get_default_properties():
        """
        Return a dictionary with the transformer's applicability and relative importance during feature evolution.

        Args: None

        Returns:
            dict: Dictionary of supported arguments and default values.

            Must contain 4 key-value pairs:

            col_type (str): the type of the original column(s) that this transformer accepts:
                "all"         - all column types
                "numeric"     - numeric int/float column
                "categorical" - string/int/float column considered a categorical for feature engineering
                "numcat"      - allow both numeric or categorical
                "datetime"    - string or int column with raw datetime such as '%Y/%m/%d %H:%M:%S' or '%Y%m%d%H%M'
                "date"        - string or int column with raw date such as '%Y/%m/%d' or '%Y%m%d'
                "text"        - string column containing text (and hence not treated as categorical)
                "time_column" - the time column specified at the start of the experiment (unmodified)

            min_cols (int or str): minimum number of columns accepted as input, of the above col_type, or "all"

            max_cols (int or str): maximum number of columns accepted as input, of the above col_type, or "all"

            relative_importance (int or float): relative importance, 1 is default.
                                 values larger than 1 will lead to over-representation,
                                 values smaller than 1 will lead to under-representation
        """
        return dict(col_type="numeric",
                    min_cols=1,
                    max_cols=1,
                    relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        """
        Return a dictionary with accepted parameters for the initializer, and their allowed values.

        Driverless AI will automatically sample (uniformly) from the values for each key. You will need to
        add repeated values to enforce non-uniformity of returned values, if desired.

        Args: None

        Returns:
            dict: Dictionary of supported arguments and possible values, first value is the default value.

        Example::
            >>> dict(arg1=[3, 1, 1, 2, 2, 5], arg2=list(range(10)))
            {'arg1': [3, 1, 1, 2, 2, 5], 'arg2': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}

            would lead to default instantiation of `__init__(arg1=3, arg2=0)` and then subsequent instantiations
            during the evolutionary algorithm with randomly sampled values such as `__init__(arg1=2, arg2=3)` etc.

        """
        return dict()

    def fit(self, X: dt.Frame, y: np.array = None):
        """
        Fit the transformer. This method is provided for documentation purposes only - never called by Driverless AI.

        Call fit_transform(X, y) instead with the ability to transform the input data X more than on a row-by-row basis.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        """
        Fit the transformer on training data `X` (a subset of the original training frame) and
        return a transformed frame (same number of rows, any number of columns >=1) with new features.

        Is always called before `transform()` is called.
        The output can be different based on whether the `fit_transform()` method is called on the entire frame
        or on a subset of rows. The output must be in the same order as the input data.

        Args:
           X (:obj:`dt.Frame`): `Python datatable github<https://github.com/h2oai/datatable>`
               Shape is `(N, M)` where `min_cols <= M <= max_cols`, and `N <= X.nrows` (number of rows of X).
               Column types are as specified by col_type in `get_default_properties()`.
               Every column can contain missing values.

               `X` can be converted to pandas with `X.to_pandas()` (integers with missing values will become floats)
               `X` can be converted to numpy with `X.to_numpy()` (with masked arrays for integers with missing values)
                                   or `X.to_pandas().values` (without masked arrays, converts to float if needed)

               Example::
                   >>> import datatable as dt
                   >>> X = dt.Frame([None, 1, 2])
                   >>> X.to_pandas()
                       C1
                   0  NaN
                   1  1.0
                   2  2.0
                   >>> X.to_numpy()
                   masked_array(
                     data=[[--],
                           [1],
                           [2]],
                     mask=[[ True],
                           [False],
                           [False]],
                     fill_value=999999,
                     dtype=int8)
                   >>> X.to_pandas().values
                   array([[nan],
                          [ 1.],
                          [ 2.]])

           y (:obj:`np.array`): The target column. Required. Defaults to None for API compatibility.
                Shape is (N,)
                Is of type :obj:`np.float` (regression) or :obj:`np.int` (multiclass) or :obj:`np.bool` (binary)


        Returns:
            :obj:`dt.Frame` or :obj:`pd.DataFrame` or :obj:`np.ndarray` (user's choice)
                shape must be (N, d) where d is >= 1
                column types must be numeric if _numeric_output==True, otherwise can be numeric or string

        Raises:
            ValueError: If acceptance test fails.
            Exception: If implementation has problems.
        """
        raise NotImplementedError("Please implement the fit_transform method.")

    def transform(self, X: dt.Frame):
        """
        Transform the validation or test dataset `X` on a row-by-row basis.

        Is only ever called after `fit_transform()` was called.
        The output must be the same whether the `transform()` method is called on the entire frame or on a subset of rows.
        The output must be in the same order as the input data.

        Args:
           X (:obj:`dt.Frame`): `Python datatable github<https://github.com/h2oai/datatable>`
               Shape is `(N, M)` where `min_cols <= M <= max_cols`, and `N <= X.nrows` (number of rows of X).
               Column types are as specified by col_type in `get_default_properties()`.
               Every column can contain missing values.

               `X` can be converted to pandas with `X.to_pandas()` (integers with missing values will be converted to float)
               `X` can be converted to numpy with `X.to_numpy()` (with masked arrays for integers with missing values)
                                   or `X.to_pandas().values` (without masked arrays, converts to float if needed)

        Returns:
            :obj:`dt.Frame` or :obj:`pd.DataFrame` or :obj:`np.ndarray` (user's choice)
                shape must be (N, d) where d is >= 1
                column types must be numeric if _numeric_output==True, otherwise can be numeric or string

        Raises:
            ValueError: If acceptance test fails.
            Exception: If implementation has problems.
        """
        raise NotImplementedError("Please implement the transform method.")

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame):  # -> MojoFrame:
        """
        Optional method to implement MOJO writing - expert mode
        """
        raise CustomMOJONotImplementedError


class CustomTimeSeriesTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols="all", max_cols="all", relative_importance=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """
        This default initializer for a `CustomTimeSeriesTransformer` ensures that certain attributes are set.
        Note: Experimental API, will most likely change in future versions.
        """
        self.encoder = kwargs['encoder']  # maps the date strings of the time column to datetime (int64)
        self.tgc = kwargs['tgc']  # name(s) of time group columns (also includes time column)
        self.pred_gap = kwargs['pred_gap']  # gap between train and test in periods
        self.pred_periods = kwargs['pred_periods']  # requested forecast horizon in periods
        self.lag_sizes = kwargs['lag_sizes']  # suggested lag sizes
        self.lag_feature = kwargs['lag_feature']  # name of feature/column to lag (can be same as target)
        self.target = kwargs['target']  # name of target column
        self.tsp = kwargs['tsp']  # lots of TS related info like period/frequency and lag autocorrelation sort order
        self.time_column = None
        self._datetime_formats = kwargs['datetime_formats']  # dictionary of date/datetime column name -> date format
        if self.tsp is not None:
            self.time_column = self.tsp._time_column  # name of time column (if present)
