"""Extremely Randomized Trees (ExtraTrees) model from sklearn"""
from math import ceil

import datatable as dt
import numpy as np
from h2oaicore.models import CustomModel
from h2oaicore.models_main import MainModel
from h2oaicore.utils import optimal_nthreads_model
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import physical_cores_count, config, loggerinfo, max_threads, get_num_threads


class ExtraTreesModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "ExtraTrees"
    _description = "Extra Trees Model based on sklearn"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _parallel_task = True
    _use_single_core_if_many = True

    # e.g. can choose 2000, and then only final model would use that many
    # 0 means ignore option
    _force_final_n_estimators = 0

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        if config.hard_asserts:
            # for bigger data, too slow to test even with 1 iteration
            use = True
            use &= train_shape is not None and train_shape[0] * train_shape[1] < 1024 * 1024 or train_shape is None
            use &= valid_shape is not None and valid_shape[0] * valid_shape[1] < 1024 * 1024 or valid_shape is None
            use &= test_shape is not None and test_shape[0] * test_shape[1] < 1024 * 1024 or test_shape is None
            # too slow for walmart with only 421k x 15
            use &= train_shape is not None and train_shape[1] < 10 or train_shape is None
            return use
        else:
            return True

    def set_default_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        kwargs.pop('get_best', None)
        self.mutate_params(accuracy=accuracy, time_tolerance=time_tolerance, interpretability=interpretability,
                           get_best=True, **kwargs)

    def estimators_list(self, accuracy=None):
        # could use config.n_estimators_list_no_early_stopping
        if accuracy is None:
            accuracy = 10
        if accuracy >= 9:
            estimators_list = [100, 200, 300, 500, 1000, 2000]
        elif accuracy >= 8:
            estimators_list = [100, 200, 300, 500, 1000]
        elif accuracy >= 5:
            estimators_list = [50, 100, 200]
        else:
            estimators_list = [10, 50, 100]
        return estimators_list

    def mutate_params(self, accuracy=10, time_tolerance=10, interpretability=1, get_best=False, **kwargs):
        # Modify certain parameters for tuning
        user_choice = config.recipe_dict.copy()
        self.params = dict()
        trial = kwargs.get('trial')
        self.params["n_estimators"] = MainModel.get_one(self.estimators_list(accuracy=accuracy), get_best=get_best,
                                                        best_type="first", name="n_estimators",
                                                        trial=trial, user_choice=user_choice)
        criterions = ["gini", "entropy"] if self.num_classes >= 2 else ["squared_error", "absolute_error"]
        self.params["criterion"] = MainModel.get_one(criterions, get_best=get_best,
                                                     best_type="first", name="criterion",
                                                     trial=trial, user_choice=user_choice)
        if config.enable_genetic_algorithm == 'Optuna':
            min_samples_split_list = list(range(2, 30))
            min_samples_leaf_list = list(range(1, 30))
        else:
            min_samples_split_list = list(range(2, 10))
            min_samples_leaf_list = list(range(1, 10))

        # max_depth just kept at pure leave mode (default)

        self.params['min_samples_split'] = MainModel.get_one(min_samples_split_list, get_best=get_best,
                                                             best_type="first", name="min_samples_split",
                                                             trial=trial,
                                                             user_choice=user_choice)
        self.params['min_samples_leaf'] = MainModel.get_one(min_samples_leaf_list, get_best=get_best,
                                                            best_type="first", name="min_samples_leaf",
                                                            trial=trial,
                                                            user_choice=user_choice)
        self.params['bootstrap'] = MainModel.get_one([False, True], get_best=get_best,
                                                     best_type="first", name="bootstrap",
                                                     trial=trial, user_choice=user_choice)
        self.params['oob_score'] = MainModel.get_one([False, True], get_best=get_best,
                                                     best_type="first", name="oob_score",
                                                     trial=trial, user_choice=user_choice)
        if self.num_classes > 1:
            self.params['class_weight'] = MainModel.get_one(['None', 'balanced', 'balanced_subsample'],
                                                            get_best=get_best,
                                                            best_type="first", name="class_weight",
                                                            trial=trial, user_choice=user_choice)
        self.params["random_state"] = MainModel.get_one([self.params_base.get("random_state", 1234)], get_best=get_best,
                                                        best_type="first", name="random_state",
                                                        trial=None,  # not for Optuna
                                                        user_choice=user_choice)

    def transcribe_params(self, params=None, **kwargs):
        """
        Fixups of params to avoid any conflicts not expressible easily for Optuna
        Or system things only need to set at fit time
        :param params:
        :return:
        """
        params_was_None = False
        if params is None:
            params = self.params  # reference, so goes back into self.params
            params_was_None = True

        if not params.get('bootstrap', False):
            params['oob_score'] = False
        if params.get('class_weight') == 'None':
            params['class_weight'] = None

        if params_was_None:
            # in case some function didn't preserve reference
            self.params = params
        return params  # default is no transcription

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        # if config.hard_asserts:
        #    from h2oaicore.utils import kwargs_has_stage, kwargs_missing_stage
        #    assert kwargs_has_stage(kwargs), kwargs_missing_stage(kwargs)

        # system thing, doesn't need to be set in default or mutate, just at runtime in fit, into self.params so can see
        self.params["n_jobs"] = self.params_base.get('n_jobs', max(1, physical_cores_count))
        params = self.params.copy()
        params = self.transcribe_params(params)

        if self._force_final_n_estimators > 0 and kwargs.get('IS_FINAL', False):
            self.params['n_estimators'] = params['n_estimators'] = self._force_final_n_estimators

        loggerinfo(self.get_logger(**kwargs), "%s fit params: %s" % (self.display_name, dict(params)))
        loggerinfo(self.get_logger(**kwargs), "%s data: %s %s" % (self.display_name, X.shape, y.shape))

        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)
            model = ExtraTreesClassifier(**params)
        else:
            params.pop('class_weight', None)
            model = ExtraTreesRegressor(**params)

        X = self.basic_impute(X)
        X = X.to_numpy()

        model.fit(X, y, sample_weight=sample_weight)
        importances = np.array(model.feature_importances_)
        self.set_model_properties(model=(model, self.min),
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=params['n_estimators'])

    def basic_impute(self, X):
        # scikit extra trees internally converts to np.float32 during all operations,
        # so if float64 datatable, need to cast first, in case will be nan for float32
        from h2oaicore.systemutils import update_precision
        X = update_precision(X, data_type=np.float32, override_with_data_type=True, fixup_almost_numeric=True)
        # Replace missing values with a value smaller than all observed values
        if not hasattr(self, 'min') or not isinstance(self.min, dict):
            self.min = dict()
        for col in X.names:
            XX = X[:, col]
            if col not in self.min:
                self.min[col] = XX.min1()
                if self.min[col] is None or np.isnan(self.min[col]) or np.isinf(self.min[col]):
                    self.min[col] = -1e10
                else:
                    self.min[col] -= 1
            XX.replace([None, np.inf, -np.inf], self.min[col])
            X[:, col] = XX
            assert X[dt.isna(dt.f[col]), col].nrows == 0
        return X

    def predict(self, X, **kwargs):
        # if config.hard_asserts:
        #    from h2oaicore.utils import kwargs_has_stage, kwargs_missing_stage
        #    assert kwargs_has_stage(kwargs), kwargs_missing_stage(kwargs)

        assert X is not None
        model_tuple, _, _, _ = self.get_model_properties()
        if len(model_tuple) == 2:
            model, self.min = model_tuple
        else:
            # migration for old recipe version
            model = model_tuple
            self.min = dict()

        # update in case changed predict core count, e.g. for FS or scoring
        model.n_jobs = self.update_n_jobs(X.shape, **kwargs)

        X = dt.Frame(X)
        X = self.basic_impute(X)
        X = X.to_numpy()

        if self.num_classes == 1:
            func = model.predict
        else:
            func = model.predict_proba

        return self.predict_in_batch(func, X, **kwargs)

    def update_n_jobs(self, X_shape, **kwargs):
        if 'IS_SCORER' in kwargs:
            # i.e. don't listen to fit-time n_jobs for predict if scoring
            parent_max_workers = max(1, kwargs.get('parent_max_workers', kwargs.get('max_workers', 1)) or 1)
            new_optimal = optimal_nthreads_model(X_shape, max_workers=parent_max_workers)
            if config.hard_asserts:
                # then in testing situation, maybe multiple experiments, so drop down
                self.params['n_jobs'] = self.params_base['n_jobs'] = min(self.params_base['n_jobs'], new_optimal)
            else:
                self.params['n_jobs'] = self.params_base['n_jobs'] = new_optimal
        else:
            # still doing experiment, shared resources assumed, re-use self.params_base['n_jobs'] from fit
            pass
        if config.max_fit_cores_override > 0:
            self.params_base['n_jobs'] = config.max_fit_cores_override
        # self.params_base['n_jobs'] can change for predict
        n_jobs = min(max_threads(), get_num_threads(), self.params_base['n_jobs'])

        return n_jobs

    def predict_in_batch(self, func, X, **kwargs):
        # sklearn not very good at handling frames, no internal batching for row-by-row operations,
        # yet predict can use much more memory than fit for same frame size
        assert X is not None
        assert isinstance(X, np.ndarray)
        nrows = X.shape[0]

        # see what shape would be
        idx = X.shape[0] - 1
        Xslice = X[idx:, :]
        preds_1 = func(Xslice)
        pred_cols = int(np.prod(preds_1.shape[1:]))

        # make empty numpy frame
        preds = np.ones((nrows, pred_cols)) * np.nan

        mem_used_per_row = 100E9 * (self.params['n_estimators'] * X.shape[1]) / (2000 * 100000 * 289)
        mem_max = 1E9

        batch_size = max(1, int(mem_max / mem_used_per_row))
        loggerinfo(self.get_logger(**kwargs), "%s predict using batch_size %d with %d batches" %
                   (self.display_name, min(nrows, batch_size), max(1, ceil(nrows / batch_size))))
        start = 0
        while start < preds.shape[0]:
            end = min(start + batch_size, preds.shape[0])
            Xslice = X[start:end, :]

            p = func(Xslice)
            preds[start:end, :] = p.reshape(end - start, pred_cols)

            start = end
        return preds
