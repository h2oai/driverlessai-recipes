"""H2O-3 Distributed Scalable Machine Learning Models (DL/GLM/GBM/DRF/NB/AutoML)
"""
import copy
import sys
import traceback

from h2oaicore.models import CustomModel
import datatable as dt
import uuid
from h2oaicore.systemutils import config, user_dir, remove, IgnoreEntirelyError, print_debug
import numpy as np
import pandas as pd

_global_modules_needed_by_name = ['h2o==3.32.1.7']
import h2o
import os


class H2OBaseModel:
    _regression = True
    _binary = True
    _multiclass = True
    _can_handle_non_numeric = True
    _can_handle_text = True  # but no special handling by base model, just doesn't fail
    _is_reproducible = False  # since using max_runtime_secs - disable that if need reproducible models
    _check_stall = False  # avoid stall check. h2o runs as server, and is not a child for which we check CPU/GPU usage
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _mutate_all = 'auto'

    _class = NotImplemented

    @staticmethod
    def do_acceptance_test():
        return False  # save time

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.id = None
        self.target = "__target__"
        self.weight = "__weight__"
        self.col_types = None
        self.my_log_dir = os.path.abspath(os.path.join(user_dir(),
                                                       config.contrib_relative_directory, "h2o_log"))
        if not os.path.isdir(self.my_log_dir):
            os.makedirs(self.my_log_dir, exist_ok=True)

    def set_default_params(self, logger=None, num_classes=None, accuracy=10, time_tolerance=10, **kwargs):

        self.params = {}

        if self._is_gbm:
            self.params.update(self.get_gbm_main_params_evolution(num_classes=num_classes,
                                                                  accuracy=accuracy,
                                                                  time_tolerance=time_tolerance,
                                                                  **kwargs))
            self.transcribe()

            self.params['col_sample_rate'] = 0.7
            self.params['sample_rate'] = 1.0
            self.params['max_depth'] = 6
            self.params['stopping_metric'] = 'auto'

        if not self._is_gbm:
            # don't limit time for gbm
            max_runtime_secs = 600
            if accuracy is not None and time_tolerance is not None:
                max_runtime_secs = accuracy * (time_tolerance + 1) * 10  # customize here to your liking
            self.params['max_runtime_secs'] = max_runtime_secs

    def get_iterations(self, model):
        return 0

    def make_instance(self, **kwargs):
        return self.__class__._class(seed=self.random_state, **kwargs)

    def transcribe(self, X=None):
        if 'early_stopping_rounds' in self.params:
            self.params['stopping_rounds'] = self.params.pop('early_stopping_rounds')
        if 'early_stopping_threshold' in self.params:
            self.params['stopping_tolerance'] = self.params.pop('early_stopping_threshold')

        if 'n_estimators' in self.params:
            self.params['ntrees'] = self.params.pop('n_estimators')
        if 'learning_rate' in self.params:
            self.params['learn_rate'] = self.params.pop('learning_rate')

        # TODO:
        # self.params['monotone_constraints']

        # have to enforce in case mutation was 1-by-1 instead of all
        if 'nbins_top_level' in self.params and 'nbins' in self.params:
            self.params['nbins_top_level'] = max(self.params['nbins_top_level'], self.params['nbins'])
        if 'min_rows' in self.params and X is not None:
            self.params["min_rows"] = min(self.params["min_rows"], max(1, int(0.5 * X.nrows)))

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        X = dt.Frame(X)
        X = self.inf_impute(X)
        self.transcribe(X=X)

        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = None

        if isinstance(self, H2ONBModel):
            # NB can only handle weights of 0 / 1
            if sample_weight is not None:
                sample_weight = (sample_weight != 0).astype(int)
            if sample_weight_eval_set is not None and len(sample_weight_eval_set) > 0 and sample_weight_eval_set[0] is not None:
                sample_weight_eval_set1 = sample_weight_eval_set[0]
                sample_weight_eval_set1[sample_weight_eval_set1 != 0] = 1
                sample_weight_eval_set1 = sample_weight_eval_set1.astype(int)
                sample_weight_eval_set = [sample_weight_eval_set1]

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

        train_y = h2o.H2OFrame(y,
                               column_names=[self.target],
                               column_types=['categorical' if self.num_classes >= 2 else 'numeric'])
        train_frame = train_X.cbind(train_y)
        if sample_weight is not None:
            train_w = h2o.H2OFrame(sample_weight,
                                   column_names=[self.weight],
                                   column_types=['numeric'])
            train_frame = train_frame.cbind(train_w)
        valid_frame = None
        valid_X = None
        valid_y = None
        model = None
        if eval_set is not None:
            valid_X = h2o.H2OFrame(eval_set[0][0].to_pandas(), column_types=self.col_types)
            valid_y = h2o.H2OFrame(eval_set[0][1],
                                   column_names=[self.target],
                                   column_types=['categorical' if self.num_classes >= 2 else 'numeric'])
            valid_frame = valid_X.cbind(valid_y)
            if sample_weight is not None:
                if sample_weight_eval_set is None:
                    sample_weight_eval_set = [np.ones(len(eval_set[0][1]))]
                valid_w = h2o.H2OFrame(sample_weight_eval_set[0],
                                       column_names=[self.weight],
                                       column_types=['numeric'])
                valid_frame = valid_frame.cbind(valid_w)

        try:
            train_kwargs = dict()
            params = copy.deepcopy(self.params)
            if not isinstance(self, H2OAutoMLModel):
                # AutoML needs max_runtime_secs in initializer, all others in train() method
                max_runtime_secs = params.pop('max_runtime_secs', 0)
                train_kwargs = dict(max_runtime_secs=max_runtime_secs)
            if valid_frame is not None:
                train_kwargs['validation_frame'] = valid_frame
            if sample_weight is not None:
                train_kwargs['weights_column'] = self.weight

            # Don't ever use the offset column as a feature
            offset_col = None  # if no column is called offset we will pass "None" and not use this feature
            cols_to_train = []  # list of all non-offset columns

            for col in list(train_X.names):
                if not col.lower() == "offset":
                    cols_to_train.append(col)
                else:
                    offset_col = col

            orig_cols = cols_to_train  # not training on offset

            trials = 2
            for trial in range(0, trials):
                try:
                    # Models that can use an offset column
                    model = self.make_instance(**params)
                    if isinstance(model, H2OGBMModel) | isinstance(model, H2ODLModel) | isinstance(model, H2OGLMModel):
                        model.train(x=cols_to_train, y=self.target, training_frame=train_frame,
                                    offset_column=offset_col,
                                    **train_kwargs)
                    else:
                        model.train(x=train_X.names, y=self.target, training_frame=train_frame, **train_kwargs)
                    break
                except Exception as e:
                    print(str(e))
                    t, v, tb = sys.exc_info()
                    ex = ''.join(traceback.format_exception(t, v, tb))
                    if 'Training data must have at least 2 features' in str(ex) and X.ncols != 0:
                        # if had non-zero features but h2o-3 saw as constant, ignore h2o-3 in that case
                        raise IgnoreEntirelyError
                    elif "min_rows: The dataset size is too small to split for min_rows" in str(e):
                        # then h2o-3 counted as rows some reduced set, since we already protect against actual rows vs. min_rows
                        params['min_rows'] = 1  # go down to lowest value
                        # permit another trial
                    else:
                        raise
                    if trial == trials - 1:
                        # if at end of trials, raise no matter what
                        raise

            if isinstance(model, H2OAutoML):
                model = model.leader
            self.id = model.model_id
            model_path = os.path.join(user_dir(), "h2o_model." + str(uuid.uuid4()))
            model_path = h2o.save_model(model=model, path=model_path)
            with open(model_path, "rb") as f:
                raw_model_bytes = f.read()

        finally:
            if model_path is not None:
                remove(model_path)
            for xx in [train_frame, train_X, train_y, model, valid_frame, valid_X, valid_y]:
                if xx is not None:
                    if isinstance(xx, H2OAutoML):
                        h2o.remove(xx.project_name)
                    else:
                        h2o.remove(xx)

        df_varimp = model.varimp(True)
        if df_varimp is None:
            varimp = np.ones(len(orig_cols))
        else:
            df_varimp.index = df_varimp['variable']
            df_varimp = df_varimp.iloc[:, 1]  # relative importance
            for missing in [x for x in orig_cols if x not in list(df_varimp.index)]:
                # h2o3 doesn't handle raw strings all the time, can hit:
                # KeyError: "None of [Index(['0_Str:secret_ChangeTemp'], dtype='object', name='variable')] are in the [index]"
                df_varimp[missing] = 0
            varimp = df_varimp[orig_cols].values  # order by fitted features
            varimp = np.nan_to_num(varimp)

        self.set_model_properties(model=raw_model_bytes,
                                  features=orig_cols,
                                  importances=varimp,
                                  iterations=self.get_iterations(model))

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

    def predict(self, X, **kwargs):
        model, _, _, _ = self.get_model_properties()
        X = dt.Frame(X)
        X = self.inf_impute(X)
        h2o.init(port=config.h2o_recipes_port, log_dir=self.my_log_dir)
        model_path = os.path.join(user_dir(), self.id)
        model_file = os.path.join(model_path, "h2o_model." + str(uuid.uuid4()) + ".bin")
        os.makedirs(model_path, exist_ok=True)
        with open(model_file, "wb") as f:
            f.write(model)
        model = h2o.load_model(os.path.abspath(model_file))
        test_frame = h2o.H2OFrame(X.to_pandas(), column_types=self.col_types)
        preds_frame = None

        try:
            if kwargs.get("pred_contribs"):
                return model.predict_contributions(test_frame).as_data_frame(header=False).values
            preds_frame = model.predict(test_frame)
            preds = preds_frame.as_data_frame(header=False)
            if self.num_classes == 1:
                return preds.values.ravel()
            elif self.num_classes == 2:
                return preds.iloc[:, -1].values.ravel()
            else:
                return preds.iloc[:, 1:].values
        finally:
            # h2o.remove(self.id) # Cannot remove id, do multiple predictions on same model
            h2o.remove(test_frame)
            remove(model_file)
            if preds_frame is not None:
                h2o.remove(preds_frame)


from h2o.estimators.naive_bayes import H2ONaiveBayesEstimator


class H2ONBModel(H2OBaseModel, CustomModel):
    _regression = False

    _display_name = "H2O NB"
    _description = "H2O-3 Naive Bayes"
    _class = H2ONaiveBayesEstimator

    def predict(self, X, **kwargs):
        preds = super().predict(X, **kwargs)
        preds = np.nan_to_num(preds, copy=False)  # get rid of infs
        if self.num_classes > 2 and \
                not np.isclose(np.sum(preds, axis=1), np.ones(preds.shape[0])).all():
            raise IgnoreEntirelyError
        return preds


from h2o.estimators.gbm import H2OGradientBoostingEstimator


class H2OGBMModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GBM"
    _description = "H2O-3 Gradient Boosting Machine"
    _class = H2OGradientBoostingEstimator
    _is_gbm = True
    _fit_by_iteration = True
    _fit_iteration_name = 'ntrees'
    _predict_by_iteration = False

    @property
    def has_pred_contribs(self):
        return self.labels is None or len(self.labels) <= 2

    def get_iterations(self, model):
        return model.params['ntrees']['actual'] + 1

    def mutate_params(self,
                      **kwargs):
        self.params['max_depth'] = int(np.random.choice([2, 3, 4, 5, 5, 6, 6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 12]))
        self.params['col_sample_rate'] = float(np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.params['sample_rate'] = float(np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
        self.params['col_sample_rate_per_tree'] = float(np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

        self.params["min_rows"] = float(np.random.choice([1, 5, 10, 20, 50, 100]))
        self.params['nbins'] = int(np.random.choice([16, 32, 64, 128, 256]))
        self.params['nbins_top_level'] = int(np.random.choice([32, 64, 128, 256, 512, 1024, 2048, 4096]))
        self.params['nbins_top_level'] = max(self.params['nbins_top_level'], self.params['nbins'])
        self.params['nbins_cats'] = int(
            np.random.choice([8, 16, 32, 64, 128, 256, 512, 512, 512, 1024, 1024, 1024, 1024, 2048, 4096]))

        self.params['learn_rate_annealing'] = float(np.random.choice([0.99, 0.999, 1.0, 1.0]))

        self.params['histogram_type'] = str(
            np.random.choice(['auto', 'auto', 'auto', 'auto', 'uniform_adaptive', 'random']))

        # "one_hot_explicit" too slow in general
        self.params['categorical_encoding'] = str(
            np.random.choice(["auto", "auto", "auto", "auto", "auto", "auto",
                              "enum", "binary", "eigen",
                              "label_encoder", "sort_by_response", "enum_limited"]))


from h2o.estimators.random_forest import H2ORandomForestEstimator


class H2ORFModel(H2OBaseModel, CustomModel):
    _display_name = "H2O RF"
    _description = "H2O-3 Random Forest"
    _class = H2ORandomForestEstimator

    @property
    def has_pred_contribs(self):
        return self.labels is None or len(self.labels) <= 2

    def get_iterations(self, model):
        return model.params['ntrees']['actual'] + 1

    def mutate_params(self,
                      **kwargs):
        max_iterations = min(kwargs['n_estimators'],
                             config.max_nestimators) if 'n_estimators' in kwargs else config.max_nestimators
        max_iterations = min(kwargs['iterations'], max_iterations) if 'iterations' in kwargs else max_iterations
        self.params['ntrees'] = max_iterations
        self.params['stopping_rounds'] = int(np.random.choice([5, 10, 20]))
        self.params['max_depth'] = int(np.random.choice(range(2, 11)))
        self.params['nbins'] = int(np.random.choice([16, 32, 64, 128, 256]))
        self.params['sample_rate'] = float(np.random.choice([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))


from h2o.estimators.deeplearning import H2ODeepLearningEstimator


class H2ODLModel(H2OBaseModel, CustomModel):
    _is_reproducible = False

    _display_name = "H2O DL"
    _description = "H2O-3 DeepLearning"
    _class = H2ODeepLearningEstimator

    def mutate_params(self,
                      accuracy=10, time_tolerance=10,
                      **kwargs):
        self.params['activation'] = np.random.choice(["rectifier", "rectifier",  # upweight
                                                      "rectifier_with_dropout",
                                                      "tanh"])
        self.params['hidden'] = np.random.choice([[20, 20, 20],
                                                  [50, 50, 50],
                                                  [100, 100, 100],
                                                  [200, 200], [200, 200, 200],
                                                  [500], [500, 500], [500, 500, 500]])
        self.params['epochs'] = accuracy * max(1, time_tolerance)
        self.params['input_dropout_ratio'] = float(np.random.choice([0, 0.1, 0.2]))


from h2o.estimators.glm import H2OGeneralizedLinearEstimator


class H2OGLMModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GLM"
    _description = "H2O-3 Generalized Linear Model"
    _class = H2OGeneralizedLinearEstimator

    def make_instance(self, **kwargs):
        if self.num_classes == 1:
            return self.__class__._class(seed=self.random_state, family='gaussian')  # tweedie/poisson/tweedie/gamma
        elif self.num_classes == 2:
            return self.__class__._class(seed=self.random_state, family='binomial')
        else:
            return self.__class__._class(seed=self.random_state, family='multinomial')


from h2o.automl import H2OAutoML


class H2OAutoMLModel(H2OBaseModel, CustomModel):
    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return False  # automl inside automl can be too slow, especially given small max_runtime_secs above

    @staticmethod
    def do_acceptance_test():
        return False  # save time

    _display_name = "H2O AutoML"
    _description = "H2O-3 AutoML"
    _class = H2OAutoML
