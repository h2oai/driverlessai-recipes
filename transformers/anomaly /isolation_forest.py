"""H2O-3 Distributed Scalable Machine Learning Transformers (IF)
"""
import os
import copy
from typing import Literal, List, Optional, Union
import uuid

import numpy as np
import pandas as pd
import datatable as dt

from h2oaicore.systemutils import config, user_dir, remove, print_debug
from h2oaicore.transformer_utils import CustomTransformer

_global_modules_needed_by_name = ['h2o==3.34.0.7']
import h2o
from h2o import H2OFrame
from h2o.estimators import H2OEstimator


class H2OIFAllNumCatTransformer(CustomTransformer):
    """
    See docs at: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/if.html
    See blog at: https://www.h2o.ai/blog/anomaly-detection-with-isolation-forests-using-h2o/

    Outputs 2 transformed features:
     1) Normalized anomaly score
     2) mean length: average number of splits across all trees to isolate the observation

    This recipe uses h2o_recipes_nthreads toml for thread count, default of 8.
    If have more cores and wish recipe to use them,
     then increase h2o_recipes_nthreads in config.toml before DAI server starts.
    """
    _display_name = "H2OIFAllNumCat"
    _description = "H2O-3 Isolation Forest for All Numeric and All Categorical Columns"
    _regression = True
    _binary = True
    _multiclass = True
    # default _unsupervised=False for CustomTransformer in case y used, but we don't use y
    # speeds-up acceptance testing since only test "unsupervised" mode with no y used
    _unsupervised = True
    # if set self.params['max_runtime_secs'] in fit(), then need to set _is_reproducible=False
    _is_reproducible = True
    _check_stall = False  # avoid stall check. h2o runs as server, and is not a child for which we check CPU/GPU usage
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    # for DAI 1.10.3+
    # can set _unique=True if know default options are sufficient and don't want multiple transformers created by genetic algorithm
    # effectively same ass ignoring possible parameter choices and only ever using first (default) value in list
    _unique = False

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="numcat",
                    min_cols="all",
                    max_cols="all",
                    relative_importance=1,
                    num_default_instances=1,
                    # if _unique=True, maximum number of times transformer is allowed to survive mutation pruning before actually pruned
                    max_perturb=4,
                    )

    @staticmethod
    def get_parameter_choices():
        """
        Copy-pste of init from H2OIsolationForestEstimator, with value as list
        :return: dict: Dictionary of supported arguments and possible values, first value is the default value.
        """
        return dict(score_each_iteration=[False],  # type: bool
                    score_tree_interval=[0],  # type: int
                    ignored_columns=[None],  # type: Optional[List[str]]
                    ignore_const_cols=[True],  # type: bool
                    ntrees=[50, 100, 200],  # type: int
                    max_depth=[8],  # type: int
                    min_rows=[1.0],  # type: float
                    max_runtime_secs=[0.0],  # type: float
                    seed=[1234],  # type: int
                    build_tree_one_node=[False],  # type: bool
                    mtries=[-1],  # type: int
                    sample_size=[256],  # type: int
                    sample_rate=[-1.0],  # type: float
                    col_sample_rate_change_per_level=[1.0],  # type: float
                    col_sample_rate_per_tree=[1.0],  # type: float
                    # type: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder", "sort_by_response", "enum_limited"]
                    categorical_encoding=["auto"],
                    stopping_rounds=[0],  # type: int
                    # type: Literal["auto", "anomaly_score", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "misclassification", "mean_per_class_error"]
                    stopping_metric=["auto"],
                    stopping_tolerance=[0.01],  # type: float
                    export_checkpoints_dir=[None],  # type: Optional[str]
                    contamination=[-1.0],  # type: float
                    validation_frame=[None],  # type: Optional[Union[None, str, H2OFrame]]
                    validation_response_column=[None],  # type: Optional[str]
                    )

    def valid_params_dict(self):
        # copy-pste of init from H2OIsolationForestEstimator
        return dict(
            model_id=None,  # type: Optional[Union[None, str, H2OEstimator]]
            training_frame=None,  # type: Optional[Union[None, str, H2OFrame]]
            score_each_iteration=False,  # type: bool
            score_tree_interval=0,  # type: int
            ignored_columns=None,  # type: Optional[List[str]]
            ignore_const_cols=True,  # type: bool
            ntrees=50,  # type: int
            max_depth=8,  # type: int
            min_rows=1.0,  # type: float
            max_runtime_secs=0.0,  # type: float
            seed=-1,  # type: int
            build_tree_one_node=False,  # type: bool
            mtries=-1,  # type: int
            sample_size=256,  # type: int
            sample_rate=-1.0,  # type: float
            col_sample_rate_change_per_level=1.0,  # type: float
            col_sample_rate_per_tree=1.0,  # type: float
            # type: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder", "sort_by_response", "enum_limited"]
            categorical_encoding="auto",
            stopping_rounds=0,  # type: int
            # type: Literal["auto", "anomaly_score", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "misclassification", "mean_per_class_error"]
            stopping_metric="auto",
            stopping_tolerance=0.01,  # type: float
            export_checkpoints_dir=None,  # type: Optional[str]
            contamination=-1.0,  # type: float
            validation_frame=None,  # type: Optional[Union[None, str, H2OFrame]]
            validation_response_column=None,  # type: Optional[str]
        )

    def __init__(self,
                 num_cols: List[str] = list(),
                 cat_cols: List[str] = list(),

                 # copy-paste from init from H2OIsolationForestEstimator
                 model_id=None,  # type: Optional[Union[None, str, H2OEstimator]]
                 training_frame=None,  # type: Optional[Union[None, str, H2OFrame]]
                 score_each_iteration=False,  # type: bool
                 score_tree_interval=0,  # type: int
                 ignored_columns=None,  # type: Optional[List[str]]
                 ignore_const_cols=True,  # type: bool
                 ntrees=50,  # type: int
                 max_depth=8,  # type: int
                 min_rows=1.0,  # type: float
                 max_runtime_secs=0.0,  # type: float
                 seed=1234,  # type: int
                 build_tree_one_node=False,  # type: bool
                 mtries=-1,  # type: int
                 sample_size=256,  # type: int
                 sample_rate=-1.0,  # type: float
                 col_sample_rate_change_per_level=1.0,  # type: float
                 col_sample_rate_per_tree=1.0,  # type: float
                 # type: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder", "sort_by_response", "enum_limited"]
                 categorical_encoding="auto",
                 stopping_rounds=0,  # type: int
                 # type: Literal["auto", "anomaly_score", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr", "misclassification", "mean_per_class_error"]
                 stopping_metric="auto",
                 stopping_tolerance=0.01,  # type: float
                 export_checkpoints_dir=None,  # type: Optional[str]
                 contamination=-1.0,  # type: float
                 validation_frame=None,  # type: Optional[Union[None, str, H2OFrame]]
                 validation_response_column=None,  # type: Optional[str]

                 output_features_to_drop=list(),
                 **kwargs,
                 ):
        """
        :param model_id: Destination id for this model; auto-generated if not specified.
               Defaults to ``None``.
        :type model_id: Union[None, str, H2OEstimator], optional
        :param training_frame: Id of the training data frame.
               Defaults to ``None``.
        :type training_frame: Union[None, str, H2OFrame], optional
        :param score_each_iteration: Whether to score during each iteration of model training.
               Defaults to ``False``.
        :type score_each_iteration: bool
        :param score_tree_interval: Score the model after every so many trees. Disabled if set to 0.
               Defaults to ``0``.
        :type score_tree_interval: int
        :param ignored_columns: Names of columns to ignore for training.
               Defaults to ``None``.
        :type ignored_columns: List[str], optional
        :param ignore_const_cols: Ignore constant columns.
               Defaults to ``True``.
        :type ignore_const_cols: bool
        :param ntrees: Number of trees.
               Defaults to ``50``.
        :type ntrees: int
        :param max_depth: Maximum tree depth (0 for unlimited).
               Defaults to ``8``.
        :type max_depth: int
        :param min_rows: Fewest allowed (weighted) observations in a leaf.
               Defaults to ``1.0``.
        :type min_rows: float
        :param max_runtime_secs: Maximum allowed runtime in seconds for model training. Use 0 to disable.
               Defaults to ``0.0``.
        :type max_runtime_secs: float
        :param seed: Seed for pseudo random number generator (if applicable)
               Defaults to ``-1``.
        :type seed: int
        :param build_tree_one_node: Run on one node only; no network overhead but fewer cpus used. Suitable for small
               datasets.
               Defaults to ``False``.
        :type build_tree_one_node: bool
        :param mtries: Number of variables randomly sampled as candidates at each split. If set to -1, defaults (number
               of predictors)/3.
               Defaults to ``-1``.
        :type mtries: int
        :param sample_size: Number of randomly sampled observations used to train each Isolation Forest tree. Only one
               of parameters sample_size and sample_rate should be defined. If sample_rate is defined, sample_size will
               be ignored.
               Defaults to ``256``.
        :type sample_size: int
        :param sample_rate: Rate of randomly sampled observations used to train each Isolation Forest tree. Needs to be
               in range from 0.0 to 1.0. If set to -1, sample_rate is disabled and sample_size will be used instead.
               Defaults to ``-1.0``.
        :type sample_rate: float
        :param col_sample_rate_change_per_level: Relative change of the column sampling rate for every level (must be >
               0.0 and <= 2.0)
               Defaults to ``1.0``.
        :type col_sample_rate_change_per_level: float
        :param col_sample_rate_per_tree: Column sample rate per tree (from 0.0 to 1.0)
               Defaults to ``1.0``.
        :type col_sample_rate_per_tree: float
        :param categorical_encoding: Encoding scheme for categorical features
               Defaults to ``"auto"``.
        :type categorical_encoding: Literal["auto", "enum", "one_hot_internal", "one_hot_explicit", "binary", "eigen", "label_encoder",
               "sort_by_response", "enum_limited"]
        :param stopping_rounds: Early stopping based on convergence of stopping_metric. Stop if simple moving average of
               length k of the stopping_metric does not improve for k:=stopping_rounds scoring events (0 to disable)
               Defaults to ``0``.
        :type stopping_rounds: int
        :param stopping_metric: Metric to use for early stopping (AUTO: logloss for classification, deviance for
               regression and anonomaly_score for Isolation Forest). Note that custom and custom_increasing can only be
               used in GBM and DRF with the Python client.
               Defaults to ``"auto"``.
        :type stopping_metric: Literal["auto", "anomaly_score", "deviance", "logloss", "mse", "rmse", "mae", "rmsle", "auc", "aucpr",
               "misclassification", "mean_per_class_error"]
        :param stopping_tolerance: Relative tolerance for metric-based stopping criterion (stop if relative improvement
               is not at least this much)
               Defaults to ``0.01``.
        :type stopping_tolerance: float
        :param export_checkpoints_dir: Automatically export generated models to this directory.
               Defaults to ``None``.
        :type export_checkpoints_dir: str, optional
        :param contamination: Contamination ratio - the proportion of anomalies in the input dataset. If undefined (-1)
               the predict function will not mark observations as anomalies and only anomaly score will be returned.
               Defaults to -1 (undefined).
               Defaults to ``-1.0``.
        :type contamination: float
        :param validation_frame: Id of the validation data frame.
               Defaults to ``None``.
        :type validation_frame: Union[None, str, H2OFrame], optional
        :param validation_response_column: (experimental) Name of the response column in the validation frame. Response
               column should be binary and indicate not anomaly/anomaly.
               Defaults to ``None``.
        :type validation_response_column: str, optional
        """
        super().__init__(**kwargs)

        init_args_dict = locals().copy()
        self.params = {k: v for k, v in init_args_dict.items() if k in self.valid_params_dict()}

        self.column_names = np.sort(num_cols + cat_cols)
        self.num_cols = np.sort(num_cols)
        self.cat_cols = np.sort(cat_cols)
        self._output_features_to_drop = output_features_to_drop

        self.col_types = None
        self.my_log_dir = os.path.abspath(os.path.join(user_dir(),
                                                       config.contrib_relative_directory, "h2o_log"))
        if not os.path.isdir(self.my_log_dir):
            os.makedirs(self.my_log_dir, exist_ok=True)

    def transcribe(self, X):
        pass

    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):
        X = dt.Frame(X)
        X = self.inf_impute(X)
        self.transcribe(X=X)

        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = None

        X_pd = X.to_pandas()

        # fix if few levels for "enum" type.  h2o-3 auto-type is too greedy and only looks at very first rows
        np_real_types = [np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]
        column_types = {}
        for col in X_pd.columns:
            if X_pd[col].dtype.type in np_real_types:
                column_types[col] = 'real'
        nuniques = {}
        for col in X_pd.columns:
            nuniques[col] = len(pd.unique(X_pd[col]))
            print_debug("NumUniques for col: %s: %d" % (col, nuniques[col]))
            if nuniques[col] <= config.max_int_as_cat_uniques and X_pd[col].dtype.type in np_real_types:
                # override original "real"
                column_types[col] = 'enum'
        # if column_types is partially filled, that is ok to h2o-3

        train_X = h2o.H2OFrame(X_pd, column_types=column_types)
        self.col_types = train_X.types

        # see uniques-types dict
        nuniques_and_types = {}
        for col, typ, in self.col_types.items():
            nuniques_and_types[col] = [typ, nuniques[col]]
            print_debug("NumUniques and types for col: %s : %s" % (col, nuniques_and_types[col]))

        train_frame = train_X
        model = None

        try:
            train_kwargs = dict()
            params = copy.deepcopy(self.params)

            # Don't ever use the offset column as a feature
            offset_col = None  # if no column is called offset we will pass "None" and not use this feature
            cols_to_train = []  # list of all non-offset columns

            for col in list(train_X.names):
                if not col.lower() == "offset":
                    cols_to_train.append(col)
                else:
                    offset_col = col

            orig_cols = cols_to_train  # not training on offset

            from h2o.estimators import H2OIsolationForestEstimator, H2OEstimator
            model = H2OIsolationForestEstimator(**params)
            model.train(x=train_X.names, training_frame=train_frame, **train_kwargs)

            preds = self.transform(X, y=y, model=model, **kwargs)

            self.id = model.model_id
            model_path = os.path.join(user_dir(), "h2o_model." + str(uuid.uuid4()))
            model_path = h2o.save_model(model=model, path=model_path)
            with open(model_path, "rb") as f:
                self.raw_model_bytes = f.read()

            return preds
        finally:
            if model_path is not None:
                remove(model_path)
            for xx in [train_frame, train_X, model]:
                if xx is not None:
                    h2o.remove(xx)


    def inf_impute(self, X):
        # Replace -inf/inf values with a value smaller/larger than all observed values
        if not hasattr(self, 'min'):
            self.min = dict()
        numeric_cols = list(X[:, [float, bool, int]].names)
        for col in X.names:
            if col not in numeric_cols:
                continue
            XX = X[:, col]
            if col not in self.min:
                self.min[col] = XX.min1()
                try:
                    if np.isinf(self.min[col]):
                        self.min[col] = -1e10
                    else:
                        self.min[col] -= 1
                except TypeError:
                    self.min[col] = -1e10
            XX.replace(-np.inf, self.min[col])
            X[:, col] = XX
        if not hasattr(self, 'max'):
            self.max = dict()
        for col in X.names:
            if col not in numeric_cols:
                continue
            XX = X[:, col]
            if col not in self.max:
                self.max[col] = XX.max1()
                try:
                    if np.isinf(self.max[col]):
                        self.max[col] = 1e10
                    else:
                        self.max[col] += 1
                except TypeError:
                    self.max[col] = 1e10
            XX.replace(np.inf, self.max[col])
            X[:, col] = XX
        return X

    def transform(self, X, y=None, **kwargs):
        X = dt.Frame(X)
        X = self.inf_impute(X)

        model_file = None
        model = kwargs.get('model')
        if model is None:
            # means came from fresh transform instead of fit_transform
            h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
            model_path = os.path.join(user_dir(), self.id)
            model_file = os.path.join(model_path, "h2o_model." + str(uuid.uuid4()) + ".bin")
            os.makedirs(model_path, exist_ok=True)
            model = self.raw_model_bytes
            with open(model_file, "wb") as f:
                f.write(model)
            model = h2o.load_model(os.path.abspath(model_file))

        test_frame = h2o.H2OFrame(X.to_pandas(), column_types=self.col_types)

        preds_frame = None
        try:
            preds_frame = model.predict(test_frame)
            return preds_frame.as_data_frame(header=False)
        finally:
            # h2o.remove(self.id) # Cannot remove id, do multiple predictions on same model
            h2o.remove(test_frame)
            remove(model_file)
            if preds_frame is not None:
                h2o.remove(preds_frame)


class H2OIFAllNumTransformer(H2OIFAllNumCatTransformer):
    _display_name = "H2OIFAllNum"
    _description = "H2O-3 Isolation Forest for All Numeric Columns"

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric",
                    min_cols="all",
                    max_cols="all",
                    relative_importance=1,
                    num_default_instances=1,
                    )


class H2OIFAllCatTransformer(H2OIFAllNumCatTransformer):
    _display_name = "H2OIFAllCat"
    _description = "H2O-3 Isolation Forest for All Categorical Columns"

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical",
                    min_cols="all",
                    max_cols="all",
                    relative_importance=1,
                    num_default_instances=1,
                    )


class H2OIFNumTransformer(H2OIFAllNumCatTransformer):
    _display_name = "H2OIFNum"
    _description = "H2O-3 Isolation Forest for Sample of Numeric Columns"

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric",
                    min_cols=1,
                    max_cols="any",
                    relative_importance=1,
                    num_default_instances=1,
                    )

class H2OIFCatTransformer(H2OIFAllNumCatTransformer):
    _display_name = "H2OIFCat"
    _description = "H2O-3 Isolation Forest for Sample of Categorical Columns"

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical",
                    min_cols=1,
                    max_cols="any",
                    relative_importance=1,
                    num_default_instances=1,
                    )
