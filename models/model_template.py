"""Template base class for a custom model recipe."""

import datatable as dt
import numpy as np
import pandas as pd

_global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file


class CustomModel(BaseCustomModel):
    """Ideally, we want a model to work with all types of supervised problems.
    Please enable the problem types it can support."""
    _unsupervised = False  # if True, ignores y
    _regression = False  # y has shape (N,) and is of numeric type, no missing values
    _binary = False  # y has shape (N,) and can be numeric or string, cardinality 2, no missing values
    _multiclass = False  # y has shape (N,) and can be numeric or string, cardinality 3+, no missing values

    """Specify whether the model can handle non-numeric categorical-like input data or not. If not, some transformers might be skipped
    during feature creation for this model."""
    _can_handle_non_numeric = False

    """Specify whether the model can handle arbitrary text input data or not. If not, some transformers might be skipped
    during feature creation for this model."""
    _can_handle_text = False

    """Specify whether the model can handle label-encoded categoricals in special way. If not, some transformers might be skipped
    during feature creation for this model."""
    _can_handle_categorical = False

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
    _get_gpu_lock = False  # whether to lock GPUs for this model before fit and predict
    _description = NotImplemented
    _check_stall = True  # whether to check for stall, should disable if separate server running task

    """Whether this model supports MOJO creation.
    If set to True, requires implementation of write_to_mojo() method."""
    _mojo = False

    """ Whether externally-provided iterations can control the fit
        If _predict_by_iteration=True, use of set_model_properties(iterations=<iterations>) should have <iterations> match value
        that would be used later for predictions that could be passed into predict as kwarg _predict_iteration_name,
        if so predict can use specific model state to avoid overfitting across folds/splits/repeats.
        If _predict_by_iteration=False, _fit_iteration_name will be kwarg passed to fit to re-fit as required to
        avoid overfitting when there are multiple folds/splits/repeats.
        If _fit_by_iteration=False, then no overfit avoidance will be attempted by DAI.
    """
    _fit_by_iteration = False

    """ Name of param passed to constructor or fit as kwargs to control iterations
        The DAI universal name for this is mapped to model.params_base[_fit_iteration_name]
        and model.params[_fit_iteration_name]
    """
    _fit_iteration_name = None

    """ Whether externally-provided iterations can control the predict for fitted model.
        If so, then predictions can be made on well-defined fold/split/repeat-averaged count,
        without re-fitting, and we pass kwargs of name _predict_iteration_name to model predict.
        """
    _predict_by_iteration = False

    """ Name of kwarg passed to predict to control iterations (ignored if _predict_by_iteration=False)"""
    _predict_iteration_name = None

    @staticmethod
    def is_enabled():
        """Return whether recipe is enabled. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def enabled_setting():
        """Return whether recipe operates in "auto", "on", or "off" mode.
           If "auto", then recipe may not always appear as overridden by automatic "smart" choices.
           E.g. if too many classes, DAI will switch to using its TensorFlow model.
           If "on", then will always appear if applicable to class count, etc.
           If "off", then not used.
           "on" is default choice, so any custom model is not automatically removed by smart choices.
        """
        return "on"

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

    @staticmethod
    def acceptance_test_timeout():
        """
        Timeout in minutes for each test of a custom recipe.
        """
        return config.acceptance_test_timeout

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        """
        Return whether the model can be used given the settings and parameters that are passed in.

        Note: If all models selected by include list have can_use of False, then the used list reverts to
        the include list without considering can_use.

        Args:
            accuracy (int): Accuracy setting for this experiment (1 to 10)
                10 is most accurate, expensive

            interpretability (int): Interpretability setting for this experiment (1 to 10)
                1 is most complex, 10 is most interpretable

            train_shape (tuple): Shape of training data

            test_shape (tuple): Shape of test data

            valid_shape (tuple): Shape of validation data

            n_gpus (int): Number of GPUs available on the system (e.g., to disable GPU-only models if no GPUs available)
ll
            **kwargs (dict): Optional dictionary containing system-level information for advanced usage

        Returns: bool
        """
        return True

    def set_default_params(self,
                           accuracy=10, time_tolerance=10, interpretability=1,
                           **kwargs):
        """Set the state of a dictionary containing model parameters to be used during `fit()` and `predict()`.

        Optional. Must set self.params to just the parameters that the __init__() method of the model can accept.

        Recommend calling mutate_params(get_best=True, *args, **kwargs) instead of making separate set_default_params logic

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
                      accuracy=10, time_tolerance=10, interpretability=1,
                      score_f_name: str = None, trial=None,
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

            score_f_name (str): scorer used by DAI, which mutate can use to infer best way to change parameters

            trial: Optuna trial object, used to tell Optuna what chosen for mutation

            **kwargs (dict): Optional dictionary containing system-level information for advanced usage

        Returns: None
        """
        pass

    def __init__(self, context=None,
                 unfitted_pipeline_path=None,
                 transformed_features=None,
                 original_user_cols=None,
                 date_format_strings={},
                 **kwargs):
        """
        :param context: Helper class to provide information about the experiment.
        :param unfitted_pipeline_path: Path to pickled pipeline that creates the feature transformations
        :param transformed_features: Column names of the data out of the feature engineering pipeline into the model
               Treated as expected set for pred_contribs=True during predict, even if model was fit on different set.
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
                list must have length of 1, containing 1 tuple of X and y for validation data
                Shape: dt.Frame: (M, p), np.array: (M, )), same schema/format as training data, just different rows
            sample_weight_eval_set (list(np.array)): (optional) validation observation weight values, numeric
                list must have length of 1, containing 1 np.array for weights
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

        Recipe can raise h2oaicore.systemutils.IgnoreError to ignore error and avoid logging error for genetic algorithm.
        Recipe can raise h2oaicore.systemutils.IgnoreEntirelyError to ignore error in all cases (including acceptance testing)

        """
        raise NotImplemented("No fit for %s" % self.__class__.__name__)

    def set_feature_importances(self, feature_importances, normalize=True):
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

        Recipe can raise h2oaicore.systemutils.IgnoreError to ignore error and avoid logging error for genetic algorithm.
        Recipe can raise h2oaicore.systemutils.IgnoreEntirelyError to ignore error in all cases (including acceptance testing)

        """

        raise NotImplemented("No predict for %s" % self.__class__.__name__)

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame, group_uuid=None, group_name=None):  # -> MojoFrame:
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


ts_raw_data_transformers = ['RawTransformer',
                            'OriginalTransformer', 'CatOriginalTransformer', 'TextOriginalTransformer',
                            'DateOriginalTransformer', 'DateTimeOriginalTransformer']
"""List of transformers that don't alter the original input relevant to custom time series models."""


class CustomTimeSeriesModel(CustomModel):
    """Model class adjusted to simplify customization for time-series problems.

    The model only accepts original (un-modified) numeric, categorical and date/datetime columns in X, in addition to
    the target column (y). All feature engineering in prior states of the pipeline are disabled. This puts full control
    into the hands of the implementer of this class. All time group columns are passed into the model with original
    column names (available in self.params_base["tgc"])
    """
    _is_custom_time_series = True
    _time_series_only = True
    _can_handle_non_numeric = True  # date format strings and time grouping columns
    _can_handle_text = False  # not handling text
    _included_transformers = ts_raw_data_transformers  # this enforces the constraint on input features
    _lag_recipe_allowed = True  # by default allow lag time series recipe (fold split and features)
    _causal_recipe_allowed = True  # by default allow causal validation scheme (no lag features)

    def __init__(self, context=None, unfitted_pipeline_path=None, transformed_features=None,
                 original_user_cols=None, date_format_strings=dict(), **kwargs):
        if not self._can_handle_non_numeric:
            raise ValueError("Please do not override _can_handle_non_numeric for CustomTimeSeriesModel.")
        if self._included_transformers != ts_raw_data_transformers:
            raise ValueError("Please do not override _included_transformers for CustomTimeSeriesModel.")
        super().__init__(context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                         transformed_features=transformed_features, original_user_cols=original_user_cols,
                         date_format_strings=date_format_strings, **kwargs)


class CustomTensorFlowModel(TensorFlowModel, CustomModel):
    """
        TensorFlow-based Custom Model
    """
    _tensorflow = True
    _parallel_task = True
    _can_use_gpu = True
    _get_gpu_lock = True

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
        super().pre_fit(X, y, sample_weight, eval_set, sample_weight_eval_set, **kwargs)
        import_tensorflow()
        self.setup_keras_session()

    @staticmethod
    def import_tensorflow():
        """
            Used if globally importing tensorflow outside function scope.
            Ensures correct CPU or GPU version of tensorflow used automatically,
            when any next call to import tensorflow [as tf] is called.
        """
        import_tensorflow()


class CustomTimeSeriesTensorFlowModel(CustomTensorFlowModel, CustomTimeSeriesModel):
    """
        TensorFlow-based Time-Series Custom Model
    """
    pass


class CustomUnsupervisedModel(UnsupervisedModel, CustomModel):
    # Custom wrapper to do unsupervised learning.
    # It bundles preprocessing of numeric/categorical data, unsupervised learning and scoring.

    # There must be exactly one CustomPreTransformer (pre), for data prep
    # There must be exactly one CustomTransformer (tr), for doing the unsupervised learning
    # y is never used.
    # There is no ensembling, only a single model.

    # The pipeline is this: model.predict(X) := tr(pre(X)) = pred

    # The model's fit method performs tr.fit_transform(pre.fit_transform(X)).
    # The model's predict method returns tr.transform(pre.transform(X)).
    # The scorer takes pre(X) and pred and returns a float.

    # pick one of the following four presets, or make your own pretransformer
    # needed to convert original data into form the transformer below can handle
    _included_pretransformers = [
        'StdFreqPreTransformer']  # standardize numerics, frequency-encode categoricals, drop rest
    # _included_pretransformers = ['OrigPreTransformer']  # pass-through numerics, drop rest
    # _included_pretransformers = ['OrigOHEPreTransformer']  # pass-through numerics, one-hot-encode categoricals, drop rest
    # _included_pretransformers = ['OrigFreqPreTransformer']  # pass-through numerics, frequency-encode categoricals, drop rest

    # select exactly one Transformer that doesn't require y (e.g., a CustomUnsupervisedTransformer) to do the unsupervised learning
    # This transformer will do the fit_transform/transform for unsupervised learning, the model is just a wrapper
    # If this transformer has MOJO support, then the Unsupervised Model will have a MOJO as well
    _included_transformers = ["MyUnsupervisedTransformer"]

    # select one CustomUnsupervisedScorer or use 'UnsupervisedScorer' if nothing to score
    _included_scorers = ['UnsupervisedScorer']

    # no need to override any other methods

