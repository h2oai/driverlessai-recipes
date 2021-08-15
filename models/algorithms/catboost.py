"""CatBoost gradient boosting by Yandex. Currently supports regression and binary classification."""
import copy, os, uuid

import datatable as dt
import numpy as np
import _pickle as pickle
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import CustomModel, MainModel
from h2oaicore.systemutils_more import arch_type
from h2oaicore.systemutils import config, physical_cores_count, ngpus_vis, save_obj, remove, user_dir
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdata
from h2oaicore.models import LightGBMModel
import inspect


# https://github.com/KwokHing/YandexCatBoost-Python-Demo
# https://catboost.ai/docs/concepts/python-usages-examples.html
class CatBoostModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "CatBoost"
    _description = "Yandex CatBoost GBM"
    _can_use_multi_gpu = False  # Can enable, but consumes too much memory
    # WIP: leakage can't find _catboost module, unsure what special.  Probably shift would fail too if used catboost.
    _can_use_gpu = True
    _force_gpu = False  # force use of GPU regardless of what DAI says
    _can_handle_categorical = True
    _can_handle_non_numeric = True
    _used_return_params = True
    _average_return_params = True
    _fit_by_iteration = True
    _fit_iteration_name = 'n_estimators'
    _is_gbm = True  # ensure final model changes n_estimators and learning_rate and complain if early stopping didn't work.
    _predict_by_iteration = True
    _predict_iteration_name = 'ntree_end'
    _save_by_pickle = True  # if False, use catboost save/load model as intermediate binary file
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    # Increase gpu_ram_part if know system is isolated

    _make_logger = True  # set to True to make logger
    _show_logger_test = False  # set to True to see how to send information to experiment logger
    _show_task_test = False  # set to True to see how task is used to send message to GUI

    _min_one_hot_max_size = 4
    _min_learning_rate_catboost = 0.005  # for catboost often for same low learning rate as xgb/lgb, too many trees

    def __init__(self, context=None,
                 unfitted_pipeline_path=None,
                 transformed_features=None,
                 original_user_cols=None,
                 date_format_strings=None,
                 **kwargs):

        super().__init__(context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                         transformed_features=transformed_features, original_user_cols=original_user_cols,
                         date_format_strings=date_format_strings, **kwargs)

        self.input_dict = dict(context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                               transformed_features=transformed_features,
                               original_user_cols=original_user_cols,
                               date_format_strings=date_format_strings, **kwargs)

    @staticmethod
    def is_enabled():
        return not (arch_type == "ppc64le")

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def acceptance_test_timeout():
        return 20.0

    @property
    def has_pred_contribs(self):
        return True

    @property
    def has_output_margin(self):
        return True

    _modules_needed_by_name = ['catboost==0.26.1']

    def set_default_params(self,
                           accuracy=10, time_tolerance=10, interpretability=1,
                           **kwargs):
        # https://catboost.ai/docs/concepts/python-reference_parameters-list.html
        #  https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
        # optimize for final model as transcribed from best lightgbm model
        n_estimators = self.params_base.get('n_estimators', 100)
        learning_rate = self.params_base.get('learning_rate', config.min_learning_rate)
        early_stopping_rounds_default = min(500, max(1, int(n_estimators / 4)))
        early_stopping_rounds = self.params_base.get('early_stopping_rounds', early_stopping_rounds_default)
        self.params = {'bootstrap_type': 'Bayesian',
                       'n_estimators': n_estimators,
                       'learning_rate': learning_rate,
                       'early_stopping_rounds': early_stopping_rounds,
                       'max_depth': 8,
                       'grow_policy': 'depthwise',
                       }

        dummy = kwargs.get('dummy', False)
        ensemble_level = kwargs.get('ensemble_level', 0)
        train_shape = kwargs.get('train_shape', (1, 1))
        valid_shape = kwargs.get('valid_shape', (1, 1))
        self.get_gbm_main_params_evolution(self.params, dummy, accuracy,
                                           self.num_classes,
                                           ensemble_level, train_shape,
                                           valid_shape)

        for k in kwargs:
            if k in self.params:
                self.params[k] = copy.deepcopy(kwargs[k])

        # self.params['has_time'] # should use this if TS problem

        if self._can_handle_categorical:
            # less than 2 is risky, can get stuck in learning
            max_cat_to_onehot_list = [4, 10, 20, 40, config.max_int_as_cat_uniques]
            self.params['one_hot_max_size'] = MainModel.get_one(max_cat_to_onehot_list, get_best=True)
            uses_gpus, n_gpus = self.get_uses_gpus(self.params)
            if uses_gpus:
                self.params['one_hot_max_size'] = min(self.params['one_hot_max_size'], 255)
            else:
                self.params['one_hot_max_size'] = min(self.params['one_hot_max_size'], 65535)

        self.params['learning_rate'] = max(self._min_learning_rate_catboost, self.params['learning_rate'])

        # fill mutatable params with best for left over if default didn't fill
        params = copy.deepcopy(self.params)
        self.mutate_params(accuracy=accuracy, time_tolerance=time_tolerance, interpretability=interpretability, get_best=True, **kwargs)
        params_from_mutate = copy.deepcopy(self.params)
        for k in params_from_mutate:
            if k not in params:
                params[k] = params_from_mutate[k]
        self.params = copy.deepcopy(params)

    def mutate_params(self, **kwargs):
        fake_lgbm_model = LightGBMModel(**self.input_dict)
        fake_lgbm_model.params = self.params.copy()
        fake_lgbm_model.params_base = self.params_base.copy()
        fake_lgbm_model.params.update(fake_lgbm_model.params_base)
        kwargs['train_shape'] = kwargs.get('train_shape', (10000, 500))
        kwargs['from_catboost'] = True
        fake_lgbm_model.mutate_params(**kwargs)
        self.params.update(fake_lgbm_model.params)
        fake_lgbm_model.transcribe_params(params=self.params, **kwargs)
        self.params.update(fake_lgbm_model.lightgbm_params)

        get_best = kwargs.get('get_best', True)
        if get_best is None:
            get_best = True
        trial = kwargs.get('trial', False)
        if trial is None:
            trial = False

        # see what else can mutate, need to know things don't want to preserve
        uses_gpus, n_gpus = self.get_uses_gpus(self.params)
        if not uses_gpus:
            colsample_bylevel_list = [0.3, 0.5, 0.9, 1.0]
            self.params['colsample_bylevel'] = MainModel.get_one(colsample_bylevel_list, get_best=get_best, best_type="first", name="colsample_bylevel", trial=trial)

        if not (uses_gpus and self.num_classes > 2):
            boosting_type_list = ['Plain', 'Ordered']
            self.params['boosting_type'] = MainModel.get_one(boosting_type_list, get_best=get_best, best_type="first", name="boosting_type", trial=trial)

        if self._can_handle_categorical:
            max_cat_to_onehot_list = [4, 10, 20, 40, config.max_int_as_cat_uniques]
            if uses_gpus:
                max_one_hot_max_size = 255
            else:
                max_one_hot_max_size = 65535
            max_cat_to_onehot_list = sorted(set([min(x, max_one_hot_max_size) for x in max_cat_to_onehot_list]))
            log = True if max(max_cat_to_onehot_list) > 1000 else False
            self.params['one_hot_max_size'] = MainModel.get_one(max_cat_to_onehot_list, get_best=get_best, best_type="max", name="one_hot_max_size", trial=trial, log=log)

        if not uses_gpus:
            sampling_frequency_list = ['PerTree', 'PerTreeLevel', 'PerTreeLevel', 'PerTreeLevel']
            self.params['sampling_frequency'] = MainModel.get_one(sampling_frequency_list, get_best=get_best, best_type="first", name="sampling_frequency", trial=trial)

        bootstrap_type_list = ['Bayesian', 'Bayesian', 'Bayesian', 'Bayesian', 'Bernoulli', 'MVS', 'Poisson', 'No']
        if not uses_gpus:
            bootstrap_type_list.remove('Poisson')
        if uses_gpus:
            bootstrap_type_list.remove('MVS')  # undocumented CPU only
        self.params['bootstrap_type'] = MainModel.get_one(bootstrap_type_list, get_best=get_best, best_type="first", name="bootstrap_type", trial=trial)

        # lgbm usage already sets subsample
        #if self.params['bootstrap_type'] in ['Poisson', 'Bernoulli']:
        #    subsample_list = [0.5, 0.66, 0.66, 0.9]
        #    # will get pop'ed if not Poisson/Bernoulli
        #    self.params['subsample'] = MainModel.get_one(subsample_list, get_best=get_best, best_type="first", name="subsample", trial=trial)

        if self.params['bootstrap_type'] in ['Bayesian']:
            bagging_temperature_list = [0.0, 0.1, 0.5, 0.9, 1.0]
            self.params['bagging_temperature'] = MainModel.get_one(bagging_temperature_list, get_best=get_best, best_type="first", name="bagging_temperature", trial=trial)

        # overfit protection different sometimes compared to early_stopping_rounds
        # self.params['od_type']
        # self.params['od_pval']
        # self.params['od_wait']
        self.params['learning_rate'] = max(config.min_learning_rate,
                                           max(self._min_learning_rate_catboost, self.params['learning_rate']))

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        logger = None
        if self._make_logger:
            # Example use of logger, with required import of:
            #  from h2oaicore.systemutils import make_experiment_logger, loggerinfo
            # Can use loggerwarning, loggererror, etc. for different levels
            if self.context and self.context.experiment_id:
                logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                                experiment_tmp_dir=self.context.experiment_tmp_dir)

        if self._show_logger_test:
            loggerinfo(logger, "TestLOGGER: Fit CatBoost")

        if self._show_task_test:
            # Example task sync operations
            if hasattr(self, 'testcount'):
                self.test_count += 1
            else:
                self.test_count = 0

            # The below generates a message in the GUI notifications panel
            if self.test_count == 0 and self.context and self.context.experiment_id:
                warning = "TestWarning: First CatBoost fit for this model instance"
                loggerwarning(logger, warning)
                task = kwargs.get('task')
                if task:
                    task.sync(key=self.context.experiment_id, progress=dict(type='warning', data=warning))
                    task.flush()

            # The below generates a message in the GUI top-middle panel above the progress wheel
            if self.test_count == 0 and self.context and self.context.experiment_id:
                message = "Tuning CatBoost"
                loggerinfo(logger, message)
                task = kwargs.get('task')
                if task:
                    task.sync(key=self.context.experiment_id, progress=dict(type='update', message=message))
                    task.flush()

        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType

        # label encode target and setup type of problem
        lb = LabelEncoder()
        if self.num_classes >= 2:
            lb.fit(self.labels)
            y = lb.transform(y)
            if eval_set is not None:
                valid_X = eval_set[0][0]
                valid_y = eval_set[0][1]
                valid_y = lb.transform(valid_y)
                eval_set = [(valid_X, valid_y)]
            self.params.update({'objective': 'Logloss'})
        if self.num_classes > 2:
            self.params.update({'objective': 'MultiClass'})

        if isinstance(X, dt.Frame):
            orig_cols = list(X.names)
            numeric_cols = list(X[:, [bool, int, float]].names)
        else:
            orig_cols = list(X.columns)
            numeric_cols = list(X.select_dtypes([np.number]).columns)

        # unlike lightgbm that needs label encoded categoricals, catboots can take raw strings etc.
        self.params['cat_features'] = [i for i, x in enumerate(orig_cols) if
                                       'CatOrig:' in x or 'Cat:' in x or x not in numeric_cols]

        if not self.get_uses_gpus(self.params):
            # monotonicity constraints not available for GPU for catboost
            # get names of columns in same order
            X_names = list(dt.Frame(X).names)
            X_numeric = self.get_X_ordered_numerics(X)
            X_numeric_names = list(X_numeric.names)
            _, _, constraints, self.set_monotone_constraints(X=X_numeric, y=y)
            # if non-numerics, then fix those to have 0 constraint
            self.params['monotone_constraints'] = [0] * len(X_names)
            colnumi = 0
            for coli in X_names:
                if X_names[coli] in X_numeric_names:
                    self.params['monotone_constraints'][coli] = constraints[colnumi]
                    colnumi += 1

        if isinstance(X, dt.Frame) and len(self.params['cat_features']) == 0:
            # dt -> catboost internally using buffer leaks, so convert here
            # assume predict is after pipeline collection or in subprocess so needs no protection
            X = X.to_numpy()  # don't assign back to X so don't damage during predict
            X = np.ascontiguousarray(X, dtype=np.float32 if config.data_precision == "float32" else np.float64)
            if eval_set is not None:
                valid_X = eval_set[0][0].to_numpy()  # don't assign back to X so don't damage during predict
                valid_X = np.ascontiguousarray(valid_X,
                                               dtype=np.float32 if config.data_precision == "float32" else np.float64)
                valid_y = eval_set[0][1]
                eval_set = [(valid_X, valid_y)]

        if eval_set is not None:
            valid_X_shape = eval_set[0][0].shape
        else:
            valid_X_shape = None

        X, eval_set = self.process_cats(X, eval_set, orig_cols)

        # modify self.params_base['gpu_id'] based upon actually-available GPU based upon training and valid shapes
        self.acquire_gpus_function(train_shape=X.shape, valid_shape=valid_X_shape)

        params = copy.deepcopy(self.params)  # keep separate, since then can be pulled form lightgbm params
        params = self.transcribe_params(params=params, **kwargs)

        if logger is not None:
            loggerdata(logger, "CatBoost parameters: params_base : %s params: %s catboost_params: %s" % (
                str(self.params_base), str(self.params), str(params)))

        if self.num_classes == 1:
            self.model = CatBoostRegressor(**params)
        else:
            self.model = CatBoostClassifier(**params)
        # Hit sometimes: Exception: catboost/libs/data_new/quantization.cpp:779: All features are either constant or ignored.
        if self.num_classes == 1:
            # assume not mae, which would use median
            # baseline = [np.mean(y)] * len(y)
            baseline = None
        else:
            baseline = None

        kwargs_fit = dict(baseline=baseline, eval_set=eval_set)
        pickle_path = None
        if config.debug_daimodel_level >= 2:
            self.uuid = str(uuid.uuid4())[:6]
            pickle_path = "catboost%s.pickle" % self.uuid
            save_obj((self.model, kwargs_fit), pickle_path)

        # FIT (with migration safety before hyperopt/Optuna function added)
        if hasattr(self, 'dask_or_hyper_or_normal_fit'):
            self.dask_or_hyper_or_normal_fit(X, y, sample_weight=sample_weight, kwargs=kwargs, **kwargs_fit)
        else:
            self.model.fit(X, y, sample_weight=sample_weight, **kwargs_fit)

        if config.debug_daimodel_level <= 2:
            remove(pickle_path)

        # https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
        # need to move to wrapper
        if self.model.get_best_iteration() is not None:
            iterations = self.model.get_best_iteration() + 1
        else:
            iterations = self.params['n_estimators']
        # must always set best_iterations
        self.model_path = None
        importances = copy.deepcopy(self.model.feature_importances_)
        if not self._save_by_pickle:
            self.uuid = str(uuid.uuid4())[:6]
            model_file = "catboost_%s.bin" % str(self.uuid)
            self.model_path = os.path.join(self.context.experiment_tmp_dir, model_file)
            self.model.save_model(self.model_path)
            with open(self.model_path, mode='rb') as f:
                model = f.read()
        else:
            model = self.model
        self.set_model_properties(model=model,  # overwrites self.model object with bytes if not using pickle
                                  features=orig_cols,
                                  importances=importances,
                                  iterations=iterations)

    def process_cats(self, X, eval_set, orig_cols):
        # ensure catboost treats as cat by making str
        if len(self.params['cat_features']) > 0:
            X = X.to_pandas()
            if eval_set is not None:
                valid_X = eval_set[0][0]
                valid_y = eval_set[0][1]
                valid_X = valid_X.to_pandas()
                eval_set = [(valid_X, valid_y)]
            for coli in self.params['cat_features']:
                col = orig_cols[coli]
                if 'CatOrig:' in col:
                    cattype = str
                    # must be string for catboost
                elif 'Cat:' in col:
                    cattype = int
                else:
                    cattype = str  # if was marked as non-numeric, must become string (e.g. for leakage/shift)
                if cattype is not None:
                    if cattype == int:
                        # otherwise would hit: ValueError: Cannot convert non-finite values (NA or inf) to integer
                        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                        X[col] = X[col].fillna(value=0)
                    X[col] = X[col].astype(cattype)
                    if eval_set is not None:
                        valid_X = eval_set[0][0]
                        valid_y = eval_set[0][1]
                        valid_X[col] = valid_X[col].astype(cattype)
                        eval_set = [(valid_X, valid_y)]
        return X, eval_set

    def predict(self, X, y=None, **kwargs):
        model, features, importances, iterations = self.get_model_properties()
        if not self._save_by_pickle:
            from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
            if self.num_classes >= 2:
                from_file = CatBoostClassifier()
            else:
                from_file = CatBoostRegressor()
            with open(self.model_path, mode='wb') as f:
                f.write(model)
            model = from_file.load_model(self.model_path)

        # FIXME: Do equivalent throttling of predict size like def _predict_internal(self, X, **kwargs), wrap-up.
        if isinstance(X, dt.Frame) and len(self.params['cat_features']) == 0:
            # dt -> lightgbm internally using buffer leaks, so convert here
            # assume predict is after pipeline collection or in subprocess so needs no protection
            X = X.to_numpy()  # don't assign back to X so don't damage during predict
            X = np.ascontiguousarray(X, dtype=np.float32 if config.data_precision == "float32" else np.float64)

        X, eval_set = self.process_cats(X, None, self.feature_names_fitted)

        pred_contribs = kwargs.get('pred_contribs', False)
        output_margin = kwargs.get('output_margin', False)
        fast_approx = kwargs.pop('fast_approx', False)
        if fast_approx:
            iterations = min(config.fast_approx_num_trees, iterations)

        # implicit import
        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType, Pool
        n_jobs = max(1, physical_cores_count)
        if not pred_contribs and not output_margin:
            if self.num_classes >= 2:
                preds = model.predict_proba(
                    X,
                    ntree_start=0,
                    ntree_end=iterations,  # index of first tree *not* to be used
                    thread_count=self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                )

                if preds.shape[1] == 2:
                    return preds[:, 1]
                else:
                    return preds
            else:
                return model.predict(
                    X,
                    ntree_start=0,
                    ntree_end=iterations,  # index of first tree *not* to be used
                    thread_count=self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                )
        elif output_margin:
            # uses "predict" for raw for any class
            preds = model.predict(
                    X,
                    prediction_type="RawFormulaVal",
                    ntree_start=0,
                    ntree_end=iterations,  # index of first tree *not* to be used
                    thread_count=self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                )
            if len(preds.shape) > 1 and preds.shape[1] == 2 and self.num_classes == 2:
                return preds[:, 1]
            else:
                return preds
        elif pred_contribs:
            # For Shapley, doesn't come from predict
            # For regression/binary, shap is shape of (rows, features + bias)
            # for multiclass, shap is shape of (rows, classes, features + bias)
            data = Pool(X, label=y, cat_features=self.params['cat_features'])
            if fast_approx:
                # https://github.com/catboost/catboost/issues/1146
                # https://github.com/catboost/catboost/issues/1535
                # can't specify trees, but they have approx version
                # Regular, Exact, or Approximate
                shap_calc_type = "Approximate"
            else:
                shap_calc_type = "Regular"
            # See also shap_mode
            # help(CatBoostClassifier.get_feature_importance)
            print("shap_calc_type: %s" % shap_calc_type)
            preds_shap = model.get_feature_importance(
                data=data,
                thread_count=self.params_base.get('n_jobs', n_jobs),  # -1 is not supported,
                type=EFstrType.ShapValues,
                shap_calc_type=shap_calc_type,
            )
            # repair broken shap sum: https://github.com/catboost/catboost/issues/1125
            preds_raw = model.predict(
                    X,
                    prediction_type="RawFormulaVal",
                    ntree_start=0,
                    ntree_end=iterations,  # index of first tree *not* to be used
                    thread_count=self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                )
            if self.num_classes <= 2:
                axis = 1
            else:
                axis = 2
            orig_sum = np.sum(preds_shap, axis=axis)
            # avoid division by 0, need different trick, e.g. change baseline, to fix that case
            if axis == 1:
                orig_sum[orig_sum[:] == 0.0] = 1.0
                preds_shap = preds_shap * preds_raw[:, None] / orig_sum[:, None]
            else:
                # each feature and each class must sum up
                orig_sum[orig_sum[:, :] == 0.0] = 1.0
                preds_shap = preds_shap * preds_raw[:, :, None] / orig_sum[:, :, None]

            if config.hard_asserts and config.debug_daimodel_level >= 2:
                model.save_model("catshapproblem")
                pickle.dump((X, y, self.params['cat_features']), open("catshapproblem.pkl", "wb"))
                preds_raw = model.predict(
                        X,
                        prediction_type="RawFormulaVal",
                        ntree_start=0,
                        ntree_end=iterations,  # index of first tree *not* to be used
                        thread_count=self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                    )

                assert np.isclose(preds_raw, np.sum(preds_shap, axis=axis)).all(), "catboost shapley does not sum up correctly"
            if axis == 1:
                return preds_shap
            else:
                # DAI expects (shape rows) * (classes x (features + 1)) with "columns" as blocks of
                # feature_0_class_0 feature_0_class_0 ... feature_0_class_1 feature_1_class_1 ...
                return preds_shap.reshape(preds_shap.shape[0], preds_shap.shape[1]*preds_shap.shape[2])
        else:
            raise RuntimeError("No such case")

    def transcribe_params(self, params=None, **kwargs):
        if params is None:
            params = self.params  # reference
        params = params.copy()  # don't contaminate DAI params, since we know we use lgbm-xgb as base

        has_eval_set = self.have_eval_set(kwargs)  # only needs (and does) operate at fit-time
        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        fullspec_regression = inspect.getfullargspec(CatBoostRegressor)
        kwargs_regression = {k: v for k, v in zip(fullspec_regression.args, fullspec_regression.defaults)}
        fullspec_classification = inspect.getfullargspec(CatBoostClassifier)
        kwargs_classification = {k: v for k, v in zip(fullspec_classification.args, fullspec_classification.defaults)}

        if self.num_classes == 1:
            allowed_params = kwargs_regression
        else:
            allowed_params = kwargs_classification

        params_copy = copy.deepcopy(params)
        for k, v in params_copy.items():
            if k not in allowed_params.keys():
                del params[k]

        # now transcribe
        k = 'boosting_type'
        if k in params:
            params[k] = 'Plain'

        k = 'grow_policy'
        if k in params:
            params[k] = 'Depthwise' if params[k] == 'depthwise' else 'Lossguide'

        k = 'eval_metric'
        if k in params and params[k] is not None and params[k].upper() == 'AUC':
            params[k] = 'AUC'

        map = {'regression': 'RMSE', 'mse': 'RMSE', 'mae': 'MAE', "mape": 'MAPE', "huber": 'Huber', "fair": 'FairLoss',
               "rmse": "RMSE",
               "gamma": "RMSE",  # unsupported by catboost
               "tweedie": "Tweedie", "poisson": "Poisson", "quantile": "Quantile",
               'binary': 'Logloss',
               'auc': 'AUC', "xentropy": 'CrossEntropy',
               'multiclass': 'MultiClass'}

        k = 'objective'
        if k in params and params[k] in map.keys():
            params[k] = map[params[k]]

        k = 'eval_metric'
        if k in params and params[k] is not None and params[k] in map.keys():
            params[k] = map[params[k]]

        if 'objective' in params:
            # don't randomly choose these since then model not stable GA -> final
            # but backup shouldn't really be used AFAIK
            if params['objective'] == 'Huber':
                backup = float(config.huber_alpha_list[0])
                params['delta'] = params.pop('alpha', backup)
            if params['objective'] == 'Quantile':
                backup = float(config.quantile_alpha[0])
                params['delta'] = params.pop('alpha', backup)
            if params['objective'] == 'Tweedie':
                backup = float(config.tweedie_variance_power_list[0])
                params['tweedie_variance_power'] = params.pop('tweedie_variance_power', backup)
            if params['objective'] == 'FairLoss':
                backup = float(config.fair_c_list[0])
                params['smoothness'] = params.pop('fair_c', backup)

        params.pop('verbose', None)
        params.pop('verbose_eval', None)
        params.pop('logging_level', None)

        if 'grow_policy' in params:
            if params['grow_policy'] == 'Lossguide':
                params.pop('max_depth', None)
            if params['grow_policy'] == 'Depthwise':
                params.pop('num_leaves', None)
        else:
            params['grow_policy'] = 'SymmetricTree'

        uses_gpus, n_gpus = self.get_uses_gpus(params)

        if params['task_type'] == 'CPU':
            params.pop('grow_policy', None)
            params.pop('num_leaves', None)
            params.pop('max_leaves', None)
            params.pop('min_data_in_leaf', None)
            params.pop('min_child_samples', None)

        if params['task_type'] == 'GPU':
            params.pop('colsample_bylevel', None)  # : 0.35

        if 'grow_policy' in params and params['grow_policy'] in ['Depthwise', 'SymmetricTree']:
            if 'max_depth' in params and params['max_depth'] in [0, -1]:
                params['max_depth'] = max(2, int(np.log(params.get('num_leaves', 2 ** 6))))
        else:
            params.pop('max_depth', None)
            params.pop('depth', None)
        if 'grow_policy' in params and params['grow_policy'] == 'Lossguide':
            # if 'num_leaves' in params and params['num_leaves'] == -1:
            #    params['num_leaves'] = 2 ** params.get('max_depth', 6)
            if 'max_leaves' in params and params['max_leaves'] in [0, -1]:
                params['max_leaves'] = 2 ** params.get('max_depth', 6)
        else:
            params.pop('max_leaves', None)
        if 'num_leaves' in params and 'max_leaves' in params:
            params.pop('num_leaves', None)
        # apply limits
        if 'max_leaves' in params:
            params['max_leaves'] = min(params['max_leaves'], 65536)
        if 'max_depth' in params:
            params['max_depth'] = min(params['max_depth'], 16)

        params.update({'train_dir': user_dir(),
                       'allow_writing_files': False,
                       'thread_count': self.params_base.get('n_jobs', 4)})

        if 'reg_lambda' in params and params['reg_lambda'] <= 0.0:
            params['reg_lambda'] = 3.0  # assume meant unset

        if self._can_handle_categorical:
            if 'max_cat_to_onehot' in params:
                params['one_hot_max_size'] = params['max_cat_to_onehot']
                params.pop('max_cat_to_onehot', None)
            if uses_gpus:
                params['one_hot_max_size'] = min(params.get('one_hot_max_size', 255), 255)
            else:
                params['one_hot_max_size'] = min(params.get('one_hot_max_size', 65535), 65535)

        if 'one_hot_max_size' in params:
            params['one_hot_max_size'] = max(self._min_one_hot_max_size, params['one_hot_max_size'])

        params['max_bin'] = params.get('max_bin', 254)
        if params['task_type'] == 'CPU':
            params['max_bin'] = min(params['max_bin'], 254)  # https://github.com/catboost/catboost/issues/1010
        if params['task_type'] == 'GPU':
            params['max_bin'] = min(params['max_bin'], 127)  # https://github.com/catboost/catboost/issues/1010

        if uses_gpus:
            # https://catboost.ai/docs/features/training-on-gpu.html
            params['devices'] = "%d-%d" % (
                self.params_base.get('gpu_id', 0), self.params_base.get('gpu_id', 0) + n_gpus - 1)
            params['gpu_ram_part'] = 0.3  # per-GPU, assumes GPU locking or no other experiments running

        if self.num_classes > 2:
            params.pop("eval_metric", None)

        params['train_dir'] = self.context.experiment_tmp_dir
        params['allow_writing_files'] = False

        # assume during fit self.params_base could have been updated
        assert 'n_estimators' in params
        assert 'learning_rate' in params
        params['n_estimators'] = self.params_base.get('n_estimators', 100)
        params['learning_rate'] = self.params_base.get('learning_rate', config.min_learning_rate)
        params['learning_rate'] = min(params['learning_rate'], 0.5)  # 1.0 leads to illegal access on GPUs
        params['learning_rate'] = max(config.min_learning_rate,
                                           max(self._min_learning_rate_catboost, params['learning_rate']))
        if 'early_stopping_rounds' not in params and has_eval_set:
            params['early_stopping_rounds'] = 150  # temp fix
            # assert 'early_stopping_rounds' in params

        if uses_gpus:
            params.pop('sampling_frequency', None)

        if not uses_gpus and params['bootstrap_type'] == 'Poisson':
            params['bootstrap_type'] = 'Bayesian'  # revert to default
        if uses_gpus and params['bootstrap_type'] == 'MVS':
            params['bootstrap_type'] = 'Bayesian'  # revert to default

        if 'bootstrap_type' not in params or params['bootstrap_type'] not in ['Poisson', 'Bernoulli']:
            params.pop('subsample', None)  # only allowed for those 2 bootstrap_type settings

        if params['bootstrap_type'] not in ['Bayesian']:
            params.pop('bagging_temperature', None)

        if not (self.num_classes == 2 and params['objective'] == 'Logloss'):
            params.pop('scale_pos_weight', None)

        # go back to some default eval_metric
        if self.num_classes == 1:
            if 'eval_metric' not in params or params['eval_metric'] not in ['MAE', 'MAPE', 'Poisson', 'Quantile',
                                                                            'RMSE', 'LogLinQuantile', 'Lq',
                                                                            'Huber', 'Expectile', 'FairLoss',
                                                                            'NumErrors', 'SMAPE', 'R2', 'MSLE',
                                                                            'MedianAbsoluteError']:
                params['eval_metric'] = 'RMSE'
        elif self.num_classes == 2:
            if 'eval_metric' not in params or params['eval_metric'] not in ['Logloss', 'CrossEntropy', 'Precision',
                                                                            'Recall', 'F1', 'BalancedAccuracy',
                                                                            'BalancedErrorRate', 'MCC', 'Accuracy',
                                                                            'CtrFactor', 'AUC',
                                                                            'NormalizedGini', 'BrierScore', 'HingeLoss',
                                                                            'HammingLoss', 'ZeroOneLoss',
                                                                            'Kappa', 'WKappa',
                                                                            'LogLikelihoodOfPrediction']:
                params['eval_metric'] = 'Logloss'
        else:
            if 'eval_metric' not in params or params['eval_metric'] not in ['MultiClass', 'MultiClassOneVsAll',
                                                                            'Precision', 'Recall', 'F1', 'TotalF1',
                                                                            'MCC', 'Accuracy', 'HingeLoss',
                                                                            'HammingLoss', 'ZeroOneLoss', 'Kappa',
                                                                            'WKappa', 'AUC']:
                params['eval_metric'] = 'MultiClass'

        # set system stuff here
        params['silent'] = self.params_base.get('silent', True)
        if config.debug_daimodel_level >= 1:
            params['silent'] = False  # Can enable for tracking improvement in console/dai.log if have access
        params['random_state'] = self.params_base.get('random_state', 1234)
        params['thread_count'] = self.params_base.get('n_jobs', max(1, physical_cores_count))  # -1 is not supported

        return params

    def get_uses_gpus(self, params):
        params['task_type'] = 'CPU' if self.params_base.get('n_gpus', 0) == 0 else 'GPU'
        if self._force_gpu:
            params['task_type'] = 'GPU'

        n_gpus = self.params_base.get('n_gpus', 0)
        if self._force_gpu:
            n_gpus = 1
        if n_gpus == -1:
            n_gpus = ngpus_vis
        uses_gpus = params['task_type'] == 'GPU' and n_gpus > 0
        return uses_gpus, n_gpus
