"""H2O-3 Distributed Scalable Machine Learning Models (DL/GLM/GBM/DRF/NB/AutoML)
"""
import copy
import json
import sys
import traceback

from h2oaicore.models import CustomModel
import datatable as dt
import uuid
from h2oaicore.systemutils import config, user_dir, remove, IgnoreEntirelyError, print_debug, exp_dir, loggerinfo
import numpy as np
import pandas as pd

_global_modules_needed_by_name = ['h2o==3.34.0.7']
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

    _compute_p_values = False
    _show_performance = False
    _show_coefficients = False

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

        gbm_params = self.get_gbm_main_params_evolution(num_classes=num_classes,
                                                                  accuracy=accuracy,
                                                                  time_tolerance=time_tolerance,
                                                                  **kwargs)

        if isinstance(self, H2OGBMModel):
            if 'n_estimators' in gbm_params:
                self.params[self._fit_iteration_name] = gbm_params['n_estimators']
            self.transcribe()

            self.params['col_sample_rate'] = 0.7
            self.params['sample_rate'] = 1.0
            self.params['max_depth'] = 6
            self.params['stopping_metric'] = 'auto'
        elif isinstance(self, (H2ORFModel, H2OGLMModel)):
            if 'n_estimators' in gbm_params:
                self.params[self._fit_iteration_name] = gbm_params['n_estimators']
            self.transcribe()

        if not isinstance(self, (H2OGBMModel, H2OGLMModel, H2ORFModel)):
            # don't limit time for gbm, glm, rf
            max_runtime_secs = 600
            if accuracy is not None and time_tolerance is not None:
                max_runtime_secs = accuracy * (time_tolerance + 1) * 10  # customize here to your liking
            self.params['max_runtime_secs'] = max_runtime_secs

    def get_iterations(self, model):
        if self._fit_iteration_name in model.params and 'actual' in model.params[self._fit_iteration_name]:
            return model.params[self._fit_iteration_name]['actual'] + 1
        elif self._fit_by_iteration:
            return self.params[self._fit_iteration_name]
        else:
            return 0

    def make_instance(self, **kwargs):
        return self.__class__._class(seed=self.random_state, **kwargs)

    def doing_p_values(self):
        return isinstance(self, H2OGLMModel) and self._compute_p_values and self.num_classes <= 2

    def transcribe(self, X=None):
        if self._support_early_stopping and isinstance(self, H2OGLMModel):
            self.params['early_stopping'] = True

        if 'early_stopping_rounds' in self.params:
            self.params['stopping_rounds'] = self.params.pop('early_stopping_rounds')
        if 'early_stopping_threshold' in self.params:
            self.params['stopping_tolerance'] = self.params.pop('early_stopping_threshold')

        if isinstance(self, (H2OGBMModel, H2ORFModel, H2OGLMModel)):
            if self._fit_iteration_name in self.params_base and self._fit_iteration_name not in self.params:
                self.params[self._fit_iteration_name] = self.params_base[self._fit_iteration_name]

            if config.hard_asserts:
                # Shapley too slow even with 50 trees, so avoid for testing
                if self._fit_iteration_name in self.params:
                    self.params[self._fit_iteration_name] = min(self.params[self._fit_iteration_name], 3)
                else:
                    self.params[self._fit_iteration_name] = 3

        if isinstance(self, H2OGBMModel):
            if 'learning_rate' in self.params_base:
                self.params['learn_rate'] = self.params_base['learning_rate']
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

            if self.doing_p_values():
                # https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/compute_p_values.html
                # take a look at the coefficients_table to see the p_values
                params['remove_collinear_columns'] = True
                params['compute_p_values'] = True
                # h2o-3 only supports p-values if lambda=0
                params['lambda_'] = 0
                if self.num_classes == 2:
                    params['family'] = 'binomial'
                params['solver'] = 'IRLSM'
                params.pop('beta_constraints', None)

            trials = 2
            for trial in range(0, trials):
                try:
                    # Models that can use an offset column
                    loggerinfo(self.get_logger(**kwargs), "%s (%s) fit parameters: %s" % (self.display_name, self.__class__.__module__, dict(params)))
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
                    elif "min_rows: The dataset size is too small to split for min_rows" in str(e) and trial == 0:
                        # then h2o-3 counted as rows some reduced set, since we already protect against actual rows vs. min_rows
                        params['min_rows'] = 1  # go down to lowest value
                        # permit another trial
                    elif "min_rows: The dataset size is too small to split for min_rows" in str(e) and trial == 1:
                        raise IgnoreEntirelyError
                    elif " java.lang.AssertionError" in str(ex):
                        # bug in h2o-3, nothing can be done
                        raise IgnoreEntirelyError
                    elif "NotStrictlyPositiveException" in str(ex):
                        # bad input data for given hyperparameters
                        raise IgnoreEntirelyError
                    else:
                        raise
                    if trial == trials - 1:
                        # if at end of trials, raise no matter what
                        raise

            if self._show_performance:
                # retrieve the model performance
                perf_train = model.model_performance(train_frame)
                loggerinfo(self.get_logger(**kwargs), self.perf_to_list(perf_train, which="training"))
                if valid_frame is not None:
                    perf_valid = model.model_performance(valid_frame)
                    loggerinfo(self.get_logger(**kwargs), self.perf_to_list(perf_valid, which="validation"))

            struuid = str(uuid.uuid4())

            if self._show_coefficients:
                coeff_table = model._model_json['output']['coefficients_table']
                # convert table to a pandas dataframe
                coeff_table = coeff_table.as_data_frame()
                is_final = 'IS_FINAL' in kwargs
                json_file = os.path.join(exp_dir(), 'coefficients_table_is_final_%s_%s.json' % (is_final, struuid))
                with open(json_file, "wt") as f:
                    pd.set_option('precision', 16)
                    f.write(json.dumps(json.loads(coeff_table.to_json()), indent=4))
                    pd.set_option('precision', 6)

            if isinstance(model, H2OAutoML):
                model = model.leader
            self.id = model.model_id
            model_path = os.path.join(exp_dir(), "h2o_model." + struuid)
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
            df_varimp = model.varimp(True)

            # deal with categorical levels appended as .<num> or .<str>
            orig_cols_set = set(orig_cols)
            df_varimp.index = df_varimp['variable']
            # try to remove tail end where cat added
            for i in range(3):
                df_varimp.index = [x if x in orig_cols_set else ".".join(x.split(".")[:-2]) for x in df_varimp.index]
            # try to remove stuff after 1st, 2nd, third dot in case above didn't work, e.g. when many .'s in string
            df_varimp.index = [x if x in orig_cols_set else ".".join(x.split(".")[0:1]) for x in df_varimp.index]
            df_varimp.index = [x if x in orig_cols_set else ".".join(x.split(".")[0:2]) for x in df_varimp.index]
            df_varimp.index = [x if x in orig_cols_set else ".".join(x.split(".")[0:3]) for x in df_varimp.index]
            df_varimp.index.name = "___INDEXINTERNAL___"
            df_varimp = df_varimp.groupby(df_varimp.index.name).sum()['scaled_importance']

            missing_features_set = set([x for x in orig_cols if x not in list(df_varimp.index)])
            # must not keep "missing features", even as zero, since h2o-3 won't have them in pred_contribs output
            orig_cols = [x for x in orig_cols if x not in missing_features_set]
            self.col_types = {k: v for k, v in self.col_types.items() if k  not in missing_features_set}
            varimp = df_varimp[orig_cols].values  # order by (and select) fitted features
            varimp = np.nan_to_num(varimp)


        self.set_model_properties(model=raw_model_bytes,
                                  features=orig_cols,
                                  importances=varimp,
                                  iterations=self.get_iterations(model))

    def perf_to_list(self, perf, which="training"):
        perf_list = []
        prefix = "%s (%s) fit %s performance:" % (self.display_name, which, self.__class__.__module__)
        for k, v in perf._metric_json.items():
            if isinstance(v, (int, str, float)):
                perf_list.append(["%s: %s: %s" % (prefix, k, v)])
        return perf_list

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
        model_path = os.path.join(exp_dir(), self.id)
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

            is_final = 'IS_FINAL' in kwargs
            struuid = str(uuid.uuid4())
            json_file = os.path.join(exp_dir(), 'stderr_is_final_%s_%s.json' % (is_final, struuid))

            if self.num_classes == 1:
                if self.doing_p_values():
                    df = preds.iloc[:, 1]
                    with open(json_file, "wt") as f:
                        pd.set_option('precision', 16)
                        f.write(json.dumps(json.loads(df.to_json()), indent=4))
                        pd.set_option('precision', 6)
                    return preds.iloc[:, 0].values.ravel()
                else:
                    return preds.values.ravel()
            elif self.num_classes == 2:
                if self.doing_p_values():
                    df = preds.iloc[:, 2]
                    with open(json_file, "wt") as f:
                        pd.set_option('precision', 16)
                        f.write(json.dumps(json.loads(df.to_json()), indent=4))
                        pd.set_option('precision', 6)
                    return preds.iloc[:, -1 - 1].values.ravel()
                else:
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

    @staticmethod
    def do_acceptance_test():
        return True

    @property
    def has_pred_contribs(self):
        return self.labels is None or len(self.labels) <= 2

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
    _is_gbm = True  # gbm means gbm-like parameters like n_estimators (ntrees) not literally only gbm
    _support_early_stopping = False  # so doesn't assume early stopping done, so no large tree counts by default
    _fit_by_iteration = True
    _fit_iteration_name = 'ntrees'
    _predict_by_iteration = False

    @staticmethod
    def do_acceptance_test():
        return False  # has issue with probs summing up, all probs 0 for multiclass

    @property
    def has_pred_contribs(self):
        return self.labels is None or len(self.labels) <= 2

    def set_default_params(self, logger=None, num_classes=None, accuracy=10, time_tolerance=10, **kwargs):
        super().set_default_params(logger=logger, num_classes=num_classes, accuracy=accuracy, time_tolerance=time_tolerance, **kwargs)
        self.mutate_params(get_best=True, accuracy=accuracy, time_tolerance=time_tolerance, **kwargs)

    def mutate_params(self, get_best=False,
                      accuracy=10, time_tolerance=10,
                      **kwargs):
        n_estimators_list = config.n_estimators_list_no_early_stopping
        if config.hard_asserts:
            # Shapley too slow even with 50 trees, so avoid for testing
            n_estimators_list = [min(3, x) for x in n_estimators_list]

        self.params[self._fit_iteration_name] = self.get_one(n_estimators_list, get_best=get_best, best_type='first',
                                                             name=self._fit_iteration_name)
        self.params['max_depth'] = int(
            self.get_one([6, 2, 3, 4, 5, 7, 8, 9, 10, 11], get_best=get_best, best_type='first', name='max_depth'))
        self.params['nbins'] = int(
            self.get_one([128, 16, 32, 64, 256], get_best=get_best, best_type='first', name='nbins'))
        self.params['sample_rate'] = float(
            self.get_one([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], get_best=get_best, best_type='first', name='sample_rate'))


class H2OEXTRAModel(H2ORFModel):
    _display_name = "H2O XRT"
    _description = "H2O-3 XRT"

    @staticmethod
    def do_acceptance_test():
        return False  # fails with preds of 0,0,0,0

    def mutate_params(self, get_best=False,
                      accuracy=10, time_tolerance=10,
                      **kwargs):
        trial = kwargs.get('trial')

        n_estimators_list = config.n_estimators_list_no_early_stopping
        if config.hard_asserts:
            # Shapley too slow even with 50 trees, so avoid for testing
            n_estimators_list = [min(3, x) for x in n_estimators_list]

        self.params[self._fit_iteration_name] = self.get_one(n_estimators_list, get_best=get_best, best_type='first', name=self._fit_iteration_name, trial=trial)
        if config.enable_genetic_algorithm == "Optuna":
            max_depth_list = [6, 2, 3, 4, 5, 7, 8, 9, 10, 11]
            nbins_list = [20, 16, 32, 64, 256]
        else:
            max_depth_list = [6, 2, 3, 4, 5, 7, 8, 9, 10, 11, 0]
            nbins_list = [20, 16, 32, 64, 256]
        self.params['max_depth'] = self.get_one(max_depth_list, get_best=get_best, best_type='first', name='max_depth', trial=trial)
        self.params['nbins'] = self.get_one(nbins_list, get_best=get_best, best_type='first', name='nbins', trial=trial)
        self.params['sample_rate'] = self.get_one([0.6320000291, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], get_best=get_best, best_type='first', name='sample_rate', trial=trial)
        self.params['histogram_type'] = self.get_one(['Random'], get_best=get_best, best_type='first', name='histogram_type', trial=None)


from h2o.estimators.deeplearning import H2ODeepLearningEstimator


class H2ODLModel(H2OBaseModel, CustomModel):
    _is_reproducible = False

    _display_name = "H2O DL"
    _description = "H2O-3 DeepLearning"
    _class = H2ODeepLearningEstimator
    _fit_by_iteration = True
    _fit_iteration_name = 'epochs'
    _predict_by_iteration = False

    def set_default_params(self, logger=None, num_classes=None, accuracy=10, time_tolerance=10, **kwargs):
        super().set_default_params(logger=logger, num_classes=num_classes, accuracy=accuracy, time_tolerance=time_tolerance, **kwargs)
        self.mutate_params(accuracy=accuracy, time_tolerance=time_tolerance, **kwargs)

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
        if config.hard_asserts:
            # avoid long times for testing
            self.params['epochs'] = min(self.params['epochs'], 3)
        self.params['input_dropout_ratio'] = float(np.random.choice([0, 0.1, 0.2]))


from h2o.estimators.glm import H2OGeneralizedLinearEstimator


class H2OGLMModel(H2OBaseModel, CustomModel):
    _display_name = "H2O GLM"
    _description = "H2O-3 Generalized Linear Model"
    _class = H2OGeneralizedLinearEstimator
    _is_gbm = True  # gbm means gbm-like parameters like n_estimators (ntrees) not literally only gbm
    _fit_by_iteration = True
    _fit_iteration_name = 'max_iterations'
    _predict_by_iteration = False

    @staticmethod
    def do_acceptance_test():
        return True

    def make_instance(self, **params):
        if self.num_classes == 1:
            params.update(dict(seed=self.random_state, family='gaussian'))
            return self.__class__._class(**params)  # tweedie/poisson/tweedie/gamma
        elif self.num_classes == 2:
            params.update(dict(seed=self.random_state, family='binomial'))
            return self.__class__._class(**params)
        else:
            params.update(dict(seed=self.random_state, family='multinomial'))
            return self.__class__._class(**params)


class H2OGLMPValuesModel(H2OGLMModel):
    _display_name = "H2O GLM with p-values"
    _description = "H2O-3 Generalized Linear Model with p-values (lambda=0 only)"
    _multiclass = False  # doesn't support multinomial

    _compute_p_values = True
    _show_coefficients = True
    _show_performance = True


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
