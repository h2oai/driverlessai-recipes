"""Template base class for a custom model recipe."""

import datatable as dt
import numpy as np
import pandas as pd

_global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file


class CustomModel(BaseCustomModel):
    """Ideally, we want a model to work with all types of supervised problems.
    Please enable the problem types it can support."""
    _regression = False  # y has shape (N,) and is of numeric type, no missing values
    _binary = False  # y has shape (N,) and can be numeric or string, cardinality 2, no missing values
    _multiclass = False  # y has shape (N,) and can be numeric or string, cardinality 3+, no missing values

    """Specify whether the model can handle non-numeric input data or not. If not, some transformers might be skipped
    during feature creation for this model."""
    _can_handle_non_numeric = False

    """Specify whether the model is expected to create reproducible results. If disabled, model might be 
    skipped for experiments run in reproducible mode."""
    _is_reproducible = True

    """Optional list of included/excluded transformers that are allowed to feed data into this model, 
    specified by their class name (e.g., _included_transformers = ["NumToCatWoETransformer"])"""
    _included_transformers = None  # List[str]
    _excluded_transformers = None  # List[str]

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    _modules_needed_by_name = []  # List[str], e.g., ["mypackage==1.3.37"]

    """Optional name to show for this model during experiment and in results"""
    _display_name = NotImplemented  # str

    """Expert settings for optimal hardware usage"""
    _parallel_task = True  # if enabled, params_base['n_jobs'] will be >= 1 (adaptive to system), otherwise 1
    _can_use_gpu = False  # if enabled, will use special job scheduler for GPUs
    _can_use_multi_gpu = False  # if enabled, can get access to multiple GPUs for single transformer (experimental)
    _description = NotImplemented
    _check_stall = True  # whether to check for stall, should disable if separate server running task

    """Whether this model supports MOJO creation.
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
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0):
        """
        Return whether the model can be used given the settings and parameters that are passed in.

        Args:
            accuracy (int): Accuracy setting for this experiment (1 to 10)
                10 is most accurate, expensive

            interpretability (int): Interpretability setting for this experiment (1 to 10)
                1 is most complex, 10 is most interpretable

            train_shape (tuple): Shape of training data

            test_shape (tuple): Shape of test data

            valid_shape (tuple): Shape of validation data

            n_gpus (int): Number of GPUs available on the system (e.g., to disable GPU-only models if no GPUs available)

            **kwargs (dict): Optional dictionary containing system-level information for advanced usage

        Returns: bool
        """
        return True

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        """Set the state of a dictionary containing model parameters to be used during `fit()` and `predict()`.

        Optional. Must set self.params to just the parameters that the __init__() method of the model can accept.

        Args:
            accuracy (int): Accuracy setting for this experiment (1 to 10)
                10 is most accurate, expensive

            time_tolerance (int): Time setting for this experiment (0 to 10)
                10 is most patient, 1 is fast

            interpretability (int): Interpretability setting for this experiment (1 to 10)
                1 is most complex, 10 is most interpretable

            **kwargs (dict): Optional dictionary containing system-level information for advanced usage

        Returns: None
        """
        self.params = {}

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):
        """Mutate `self.params` dictionary of model parameters to be used during `fit()` and `predict()`.

        Called to modify model parameters `self.params` in a self-consistent way, fully controlled by the user.
        If no parameter tuning desired, leave at default.

        Args:
            accuracy (int): Accuracy setting for this experiment (1 to 10)
                10 is most accurate, expensive

            time_tolerance (int): Time setting for this experiment (0 to 10)
                10 is most patient, 1 is fast

            interpretability (int): Interpretability setting for this experiment (1 to 10)
                1 is most complex, 10 is most interpretable

            **kwargs (dict): Optional dictionary containing system-level information for advanced usage

        Returns: None
        """
        pass

    def __init__(self, context=None,
                 unfitted_pipeline_path=None,
                 transformed_features=None,
                 original_user_cols=None,
                 date_format_strings=None,
                 **kwargs):
        """
        :param context: Helper class to provide information about the experiment.
        :param unfitted_pipeline_path: Path to pickled pipeline that creates the feature transformations
        :param transformed_features: Column names of the data out of the feature engineering pipeline into the model
        :param original_user_cols: Column names of original data that went into the feature engineering pipeline
        :param date_format_strings: Date/Datetime format strings for columns of type 'time'
        :param kwargs: Additional internal arguments

        Notes:
            If you intend to override `__init__`, then make sure to call `super().__init__` as in this example.
            Pass all arguments through as is done here, and initialize any attributes you may need.

            As a side-effect of calling `super().__init__`, `self` is ensured to have:
                self.num_classes: Number of classes
                self.labels: Target class labels for classification problems
                self.params_base: Dictionary of internal parameters, can be ignored
        """
        if context is not None:
            self.tmp_dir = context.working_dir  # place to put temporary files, please clean them up afterwards
        super().__init__(context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                         transformed_features=transformed_features, original_user_cols=original_user_cols,
                         date_format_strings=date_format_strings, **kwargs)

    def fit(self, X: dt.Frame, y: np.array, sample_weight: np.array = None,
            eval_set=None, sample_weight_eval_set=None, **kwargs):
        """Fit the model on training data and use optional validation data to tune parameters to avoid overfitting.

        Args:
            X (dt.Frame): training data, concatenated output of all active transformers' `fit_transform()` method
                Shape: (N, p), rows are observations, columns are features (attributes)
            y (np.array): training target values, numeric for regression, numeric or categorical for classification
                Shape: (N, ), 1 target value per observation
            sample_weight (np.array): (optional) training observation weight values, numeric
                Shape: (N, ), 1 observation weight value per observation
            eval_set (list(tuple(dt.Frame, np.array))): (optional) validation data and target values
                Shape: dt.Frame: (M, p), np.array: (M, )), same schema/format as training data, just different rows
            sample_weight_eval_set (np.array): (optional) validation observation weight values, numeric
                Shape: (M, ), 1 observation weight value per observation
            kwargs (dict): Additional internal arguments (see examples)

        Returns: None


        Note:
            Once the model is fitted, you can pass the state to Driverless AI via `set_model_properties()` for later
            retrieval during `predict()`. See examples.

            def set_model_properties(self, model=None, features=None, importances=None, iterations=None):
                :param model: model object that contains all large fitted objects related to model
                :param features: list of feature names fitted on
                :param importances: list of associated numerical importance of features
                :param iterations: number of iterations, used to predict on or re-use for fitting on full training data

        """
        raise NotImplemented("No fit for %s" % self.__class__.__name__)

    def set_feature_importances(self, feature_importances):
        df_imp = pd.DataFrame()
        df_imp['fi'] = self.feature_names_fitted
        df_imp['fi_depth'] = 0
        df_imp['gain'] = feature_importances
        df_imp['gain'] /= df_imp['gain'].max()
        self.feature_importances = df_imp

    def predict(self, X, **kwargs):
        """Make predictions on a test set.

        Use the fitted state stored in `self` to make per-row predictions. Predictions must be independent of order of
        test set rows, and should not depend on the presence of any other rows.

        Args:
            X (dt.Frame): test data, concatenated output of all active transformers' `transform()` method
                Shape: (K, p)
            kwargs (dict): Additional internal arguments (see examples)

        Returns: dt.Frame, np.ndarray or pd.DataFrame, containing predictions (target values or class probabilities)
            Shape: (K, c) where c = 1 for regression or binary classification, and c>=3 for multi-class problems.

        Note:
            Retrieve the fitted state via `get_model_properties()`, which returns the arguments that were passed after
            the call to `set_model_properties()` during `fit()`. See examples.
        """

        raise NotImplemented("No predict for %s" % self.__class__.__name__)

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame):  # -> MojoFrame:
        """
        Optional method to implement MOJO writing - expert mode
        """
        raise CustomMOJONotImplementedError

    @property
    def has_pred_contribs(self):
        """
        Whether the model can create Shapley values when `pred_contribs=True` is passed to the `predict()` method.
        This is optional and only needed for MLI. It would create `p+1` columns with per-feature importances and one
        additional bias term. The row-wise sum of values would need to add up to the predictions in margin space (i.e,
        before applying link functions or target transformations).
        """
        return False

    @property
    def has_output_margin(self):
        """
        Whether the model can create predictions in margin space when `output_margin=True` is passed to
        the `predict()` method. This is optional and only needed for consistency checks of Shapley values.
        """
        return False


ts_raw_data_transformers = ['OriginalTransformer', 'CatOriginalTransformer',
                            'DateOriginalTransformer', 'DateTimeOriginalTransformer']
"""List of transformers that don't alter the original input relevant to custom time series models."""


class CustomTimeSeriesModel(CustomModel):
    """Model class adjusted to simplify customization for time-series problems.

    The model only accepts original (un-modified) numeric, categorical and date/datetime columns in X, in addition to
    the target column (y). All feature engineering in prior states of the pipeline are disabled. This puts full control
    into the hands of the implementer of this class. All time group columns are passed into the model with original
    column names (available in self.params_base["tgc"])
    """

    _can_handle_non_numeric = True  # date format strings and time grouping columns
    _included_transformers = ts_raw_data_transformers  # this enforces the constraint on input features

    def __init__(self, context=None, unfitted_pipeline_path=None, transformed_features=None,
                 original_user_cols=None, date_format_strings=dict(), **kwargs):
        if not self._can_handle_non_numeric:
            raise ValueError("Please do not override _can_handle_non_numeric for CustomTimeSeriesModel.")
        if self._included_transformers != ts_raw_data_transformers:
            raise ValueError("Please do not override _included_transformers for CustomTimeSeriesModel.")
        super().__init__(context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                         transformed_features=transformed_features, original_user_cols=original_user_cols,
                         date_format_strings=date_format_strings, **kwargs)


class CustomTensorFlowModel(CustomModel, TensorFlowModel):
    """
        TensorFlow-based Custom Model
    """
    _tensorflow = True
    _parallel_task = True
    _can_use_gpu = True
    _can_use_multi_gpu = True  # conservative, force user to override

    def setup_keras_session(self):
        """
            Set tensorflow session.
            If didn't do this, all GPU(s) memory would be used
            Can override this method if want more control, like commented commands show
        """
        self.tf_config = self.set_tf_config({})
        import h2oaicore.keras as keras
        keras.backend.set_session(session=TensorFlowModel.make_sess(self.tf_config))

    def setup_keras_simple_session(self):
        """
            Simple tensorflow session.
            Can use to only control some things, overriding DAI defaults
        """
        #
        self.tf_config = self.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        import h2oaicore.keras as keras
        keras.backend.set_session(session=TensorFlowModel.make_sess(self.tf_config))

    def pre_fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        """
           Ensures later import tensorflow uses correct CPU/GPU version
        """
        from h2oaicore.tensorflow_dynamic import got_cpu_tf, got_gpu_tf
        self.setup_keras_session()

    @staticmethod
    def import_tensorflow():
        """
            Used if globally importing tensorflow outside function scope.
            Ensures correct CPU or GPU version of tensorflow used automatically,
            when any next call to import tensorflow [as tf] is called.
        """
        from h2oaicore.tensorflow_dynamic import got_cpu_tf, got_gpu_tf


class CustomTimeSeriesTensorFlowModel(CustomTimeSeriesModel, CustomTensorFlowModel):
    """
        TensorFlow-based Time-Series Custom Model
    """
    pass


