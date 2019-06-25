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

    """Whether this model supports MOJO creation.
    If set to True, requires implementation of write_to_mojo() method."""
    _mojo = False

    @staticmethod
    def is_enabled():
        """Toggle to enable/disable recipe. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

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

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0):
        """
        Whether the model can be used given the settings and parameters that are passed in.

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
        """Method to create a dict containing model parameters to be used during `fit()` and `predict()`.

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
        """Method to mutate the self.params dict containing model parameters to be used during `fit()` and `predict()`.

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

    def fit(self, X: dt.Frame, y: np.array, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
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
        """
        Optional method to implement MOJO writing - expert mode
        """
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
