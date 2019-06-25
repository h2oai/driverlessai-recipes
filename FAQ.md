# FAQs for Recipes for H2O Driverless AI

* BYOR TRANSFORMERS

Custom Transformer Recipes are implemented via the following API:

```

import datatable as dt
import numpy as np
import pandas as pd


class CustomTransformer(DataTableTransformer):
    """Base class for a custom transformer recipe that can be specified externally to Driverless AI.

    Note: Experimental API, will most likely change in future versions.
    """

    """By default, we want a transformer to work with all types of supervised problems.
    Please disable the ones it cannot support if needed."""
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
    (e.g., _included_boosters = ['my_arima'], _excluded_boosters = ['tensorflow'])"""
    _included_boosters = None   # List[str]
    _excluded_boosters = None   # List[str]

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    _modules_needed_by_name = []  # List[str]

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
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

    @staticmethod
    def is_enabled():
        """Toggle to enable/disable recipe. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def get_default_properties():
        """
        Static method to specify the transformer's applicability and relative importance during feature evolution.

        Args: None

        Returns:
            dict: Dictionary of supported arguments and default values.

            Must contain 4 key-value pairs:

            col_type (str): the type of the original column(s) that this transformer accepts (unmodified, as in input data):
                "all"         - all column types
                "numeric"     - numeric int/float column
                "categorical" - string/int/float column considered a categorical for feature engineering (internal logic to decide)
                "numcat"      - allow both numeric or categorical
                "datetime"    - string or int column with the original raw datetime stamp such as '%Y/%m/%d %H:%M:%S' or '%Y%m%d%H%M'
                "date"        - string or int column with the original raw date stamp such as '%Y/%m/%d' or '%Y%m%d'
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
        Static method to specify additional parameters for the initializer, and their allowed values.

        Driverless AI will automatically sample (uniformly) from the values for each key.

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
        Optional method to fit a transformer, provided for documentation purposes only - never called by Driverless AI.

        Call fit_transform(X, y) instead with the ability to transform the input data X more than on a row-by-row basis.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        """
        Required method to fit a transformer on training data `X` and to return a frame of new engineered columns/features.

        Is always called before `transform()` is called.
        The output can be different when the `fit_transform()` method is called on the entire frame or on a subset of rows.
        The output must be in the same order as the input data.

        Args:
           X (:obj:`dt.Frame`): `Python datatable github<https://github.com/h2oai/datatable>`
               Shape is `(N, M)` where `min_cols <= M <= max_cols`, and `N <= X.nrows` (number of rows of X).
               Column types are as specified by col_type in `get_default_properties()`.
               Every column can contain missing values.

               `X` can be converted to pandas with `X.to_pandas()` (integers with missing values will be converted to float)
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
        Required method to transform the validation or test set `X` on a row-by-row basis.

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

```
* BYOR MODELS

Custom Model Recipes are implemented via the following API:

```
class CustomModel(BaseCustomModel):
    _boosters = ['custom']  # set this to something that is unique for your model
    _included_transformers = None
    _excluded_transformers = None
    _mojo = False
    _parallel_task = True  # assumes will use n_jobs in params_base
    _fixed_threads = False  # whether have set_threads(max_workers) method for how many threads wil really use
    _can_use_gpu = False
    _can_use_multi_gpu = False
    _is_reproducible = True
    _datatable_in_out = True
    _modules_needed_by_name = []
    # _global_modules_needed_by_name = [] # in module global scope
    _display_name = NotImplemented
    _description = NotImplemented
    _regression = False
    _binary = False
    _multiclass = False

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @property
    def has_pred_contribs(self):
        return False

    @property
    def has_output_margin(self):
        return False

    @staticmethod
    def override_params_for_fs(params, train_shape, accuracy, time_tolerance, interpretability):
        return params

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0):
        return True

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        self.params = {}

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):
        pass

    def __init__(self, context=None,
                 unfitted_pipeline_path=None,  # pipeline that creates features inside of class instance
                 # the complete set of features supposed to be created by the pipeline, in case it's different due to
                 # folds, data etc. - needed for consistency with expectations for pred_contribs (Shapley) etc.
                 transformed_features=None,
                 original_user_cols=None,
                 date_format_strings=dict(),
                 **kwargs):
        '''

        :param context:
        :param unfitted_pipeline_path:
        :param transformed_features:
        :param original_user_cols:
        :param date_format_strings:
        :param kwargs:

        self is ensured to have:
        self.num_classes: Number of classes
        self.labels: labels for multiclass
        self.params_base: dict of parameters for model
        '''
        kwargs['n_gpus'] = 0  # no GPU support for now
        if context is not None:
            self.tmp_dir = context.working_dir
        kwargs['booster'] = self._boosters[0]
        MainModel.__init__(self, context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                           transformed_features=transformed_features, original_user_cols=original_user_cols,
                           date_format_strings=date_format_strings, **kwargs)
        self.params_base['booster'] = self._boosters[0]

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        raise NotImplemented("No fit for %s" % self.__class__.__name__)

    def set_feature_importances(self, feature_importances):
        df_imp = pd.DataFrame()
        df_imp['fi'] = self.feature_names_fitted
        df_imp['fi_depth'] = 0
        df_imp['gain'] = feature_importances
        df_imp['gain'] /= df_imp['gain'].max()
        self.feature_importances = df_imp

    def predict(self, X, **kwargs):
        raise NotImplemented("No predict for %s" % self.__class__.__name__)

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame):  # -> MojoFrame:
        raise CustomMOJONotImplementedError


ts_raw_data_transformers = ['OriginalTransformer', 'CatOriginalTransformer',
                            'DateOriginalTransformer', 'DateTimeOriginalTransformer']


class CustomTimeSeriesModel(CustomModel):
    _included_transformers = ts_raw_data_transformers

    def __init__(self, context=None, unfitted_pipeline_path=None, transformed_features=None,
                 original_user_cols=None, date_format_strings=dict(), **kwargs):
        if self._included_transformers != ts_raw_data_transformers:
            raise ValueError("Must not override _included_transformers for CustomTimeSeriesModel.")
        MainModel.__init__(self, context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                           transformed_features=transformed_features, original_user_cols=original_user_cols,
                           date_format_strings=date_format_strings, **kwargs)

```
* BYOR SCORERS

Custom Scorer Recipes are implemented via the following API:

```
class RegressionScorer(BaseScorer):
    _regression = True
    _maximize = False
    _limit_type = OutlierFilter.REL_BEST
    _perfect_score = 0.0
    score = NotImplemented  # to avoid making instances


class ClassificationScorer(BaseScorer):
    _binary = True
    _multiclass = True
    _maximize = True
    _limit_type = OutlierFilter.NONE
    _perfect_score = 1.0
    score = NotImplemented  # to avoid making instances


class CustomScorer(BaseScorer):
    _is_custom = True
    _description = NotImplemented
    _maximize = True  # whether a higher score is better
    _perfect_score = 1.0  # the ideal score
    _regression = False
    _binary = False
    _multiclass = False
    _modules_needed_by_name = []
    # _global_modules_needed_by_name = [] # in module global scope
    _can_use_gpu = False
    _can_use_multi_gpu = False
    _supports_sample_weight = True

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def is_enabled():
        return True

    def score(
            self,
            actual: np.array,
            predicted: np.array,
            sample_weight: typing.Optional[np.array] = None,
            labels: typing.Optional[np.array] = None) -> float:
        """Override this function to compute correct score."""
        raise NotImplementedError

```
