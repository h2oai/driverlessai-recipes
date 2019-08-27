"""
Sklearn Logistic Regression.
Useful when weak or no interactions between features,
or large inherent number of levels in categorical features

Other useful DAI options if want to only use feature made internally by this model:
config.prob_prune_genes = False
config.prob_prune_by_features = False
_included_transformers = ['CatOriginalTransformer']


"""
import datatable as dt
import numpy as np
import random
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, make_scorer

from h2oaicore.models import CustomModel
from h2oaicore.systemutils import config, physical_cores_count
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from h2oaicore.transformers import CatOriginalTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score


class LogisticRegressionModel(CustomModel):
    _grid_search = False  # WIP
    _grid_search_iterations = True
    _cv_search = True  # can be used in conjunction with _grid_search_iterations = True or False
    # _impute_type = 'oob'
    _impute_type = 'sklearn'

    _mutate_all = True  # tell DAI we fully controls mutation
    _mutate_by_one = False  # recipe only changes one key at a time, can limit exploration if set as True
    _overfit_limit_iteration_step = 10
    _randomized_random_state = True

    _regression = False
    _binary = True
    _multiclass = True
    _parallel_task = True if _grid_search or _grid_search_iterations else False
    _fit_by_iteration = True
    _fit_iteration_name = 'max_iter'
    _can_handle_non_numeric = True
    _display_name = "LR"
    _description = "Logistic Regression"

    _used_return_params = True
    _average_return_params = True

    _always_defaults = False

    _use_numerics = True
    _use_ohe_encoding = True
    _use_target_encoding = False
    _use_target_encoding_other = False
    _use_ordinal_encoding = False
    _use_catboost_encoding = False  # Note: Requires data be randomly shuffled w.r.t. target
    _use_woe_encoding = False

    _modules_needed_by_name = ['category_encoders']
    if _use_target_encoding_other:
        _modules_needed_by_name.extend(['target_encoding'])

    def set_default_params(self, accuracy=None, time_tolerance=None,
                           interpretability=None, **kwargs):
        # Fill up parameters we care about
        self.params = {}
        self.mutate_params(get_default=True)

    def mutate_params(self, accuracy=10, **kwargs):
        get_default = 'get_default' in kwargs and kwargs['get_default'] or self._always_defaults
        params_orig = copy.deepcopy(self.params)

        # control some behavior by how often the model was mutated.
        # Good models that improve get repeatedly mutated, bad models tend to be one-off mutations of good models
        if get_default:
            self.params['mutation_count'] = 0
        else:
            if 'mutation_count' in self.params:
                self.params['mutation_count'] += 1
            else:
                self.params['mutation_count'] = 0

        self.params['random_state'] = kwargs.get("random_state", 1234)
        if self._randomized_random_state:
            self.params['random_state'] = random.randint(0, 32000)
        self.params['n_jobs'] = self.params_base.get('n_jobs', max(1, physical_cores_count))

        # Modify certain parameters for tuning
        C_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.0]
        self.params["C"] = float(np.random.choice(C_list)) if not get_default else 0.1

        tol_list = [1e-4, 1e-3, 1e-5]
        self.params["tol"] = float(np.random.choice(tol_list)) if not get_default else 1e-4

        # solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        # newton-cg too slow
        # sag too slow
        # solver_list = ['lbfgs', 'liblinear', 'saga']
        solver_list = ['lbfgs']
        self.params["solver"] = str(np.random.choice(solver_list)) if not get_default else 'lbfgs'

        max_iter_list = [100, 200, 1000]
        self.params["max_iter"] = int(np.random.choice(max_iter_list)) if not get_default else 100
        # self.params["max_iter"] = 5 # HACK

        if self.params["solver"] in ['lbfgs', 'newton-cg', 'sag']:
            penalty_list = ['l2', 'none']
        elif self.params["solver"] in ['saga']:
            penalty_list = ['l1', 'l2', 'none']
        elif self.params["solver"] in ['liblinear']:
            penalty_list = ['l1']
        else:
            raise RuntimeError("No such solver: %s" % self.params['solver'])
        self.params["penalty"] = str(np.random.choice(penalty_list)) if not get_default else 'l2'

        if self.params["penalty"] == 'elasticnet':
            l1_ratio_list = [0, 0.25, 0.5, 0.75, 1.0]
            self.params["l1_ratio"] = float(np.random.choice(l1_ratio_list))
        else:
            self.params.pop('l1_ratio', None)
        if self.params["penalty"] == 'none':
            self.params.pop('C', None)
        else:
            self.params['C'] = float(np.random.choice(C_list)) if not get_default else 0.1
        if self.num_classes > 2:
            self.params['multi_class'] = 'auto'

        strategy_list = ['mean', 'median', 'most_frequent', 'constant']
        self.params['strategy'] = str(np.random.choice(strategy_list)) if not get_default else 'mean'

        if self._use_target_encoding:
            min_samples_leaf_list = [1, 10, 50, 100]
            self.params['min_samples_leaf'] = float(np.random.choice(min_samples_leaf_list))
            smoothing_list = [1.0, 0.5, 10.0, 50.0]
            self.params['smoothing'] = float(np.random.choice(smoothing_list))

        if self._use_catboost_encoding:
            sigma_list = [None, 1.0, 0.5, 10.0, 50.0]
            self.params['sigma'] = random.choice(sigma_list)

        if self._use_woe_encoding:
            randomized_list = [True, False]
            self.params['randomized'] = random.choice(randomized_list)
            sigma_woe_list = [0.05, 0.025, 0.1, 0.2]
            self.params['sigma_woe'] = random.choice(sigma_woe_list)
            regularization_list = [1.0, 0.5, 2.0, 10.0]
            self.params['regularization'] = random.choice(regularization_list)

        if self._mutate_by_one and not get_default and params_orig:
            pick_key = str(np.random.choice(list(self.params.keys()), size=1)[0])
            value = self.params[pick_key]
            self.params = copy.deepcopy(params_orig)
            self.params[pick_key] = value
            # WIP, need to validate parameters in clean way
            if pick_key == 'penalty':
                # has restrictions need to switch other keys if mismatched
                if self.params["solver"] in ['lbfgs', 'newton-cg', 'sag']:
                    penalty_list = ['l2', 'none']
                elif self.params["solver"] in ['saga']:
                    penalty_list = ['l1', 'l2', 'none']
                elif self.params["solver"] in ['liblinear']:
                    penalty_list = ['l1']
                if not self.params['penalty'] in penalty_list:
                    self.params['penalty'] = penalty_list[0]  # just choose first

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        orig_cols = list(X.names)
        if self.num_classes >= 2:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)

        if self._impute_type == 'oob':
            # Replace missing values with a value smaller than all observed values
            self.min = dict()
            for col in X.names:
                XX = X[:, col]
                self.min[col] = XX.min1()
                if self.min[col] is None or np.isnan(self.min[col]):
                    self.min[col] = -1e10
                else:
                    self.min[col] -= 1
                XX.replace(None, self.min[col])
                X[:, col] = XX
                assert X[dt.isna(dt.f[col]), col].nrows == 0

        X = X.to_pandas()
        X_names = list(X.columns)

        lr_params = copy.deepcopy(self.params)

        # cat_features = [x for x in X_names if CatOriginalTransformer.is_me_transformed(x)]
        # noncat_features = [x for x in X_names if x not in cat_features]
        numerical_features = X.dtypes == 'float'
        categorical_features = ~numerical_features
        cat_X = X.loc[:, categorical_features]
        num_X = X.loc[:, numerical_features]

        full_features_list = []
        transformers = []
        if self._use_numerics and any(numerical_features.values):
            impute_params = {}
            impute_params['strategy'] = lr_params.pop('strategy', None)
            full_features_list.extend(list(num_X.columns))
            transformers.append(
                (make_pipeline(SimpleImputer(**impute_params), StandardScaler()), numerical_features)
            )
        # http://contrib.scikit-learn.org/categorical-encoding/
        if self._use_ordinal_encoding and any(categorical_features.values):
            ord_params = dict(handle_missing='value', handle_unknown='value')
            full_features_list.extend(list(cat_X.columns))
            # Note: OrdinalEncoder doesn't handle unseen features, while CategoricalEncoder used too
            import category_encoders as ce
            transformers.append(
                (ce.OrdinalEncoder(**ord_params), categorical_features)
            )
        if self._use_catboost_encoding and any(categorical_features.values):
            cb_params = dict(handle_missing='value', handle_unknown='value')
            cb_params['sigma'] = lr_params.pop('sigma')
            full_features_list.extend(list(cat_X.columns))
            import category_encoders as ce
            transformers.append(
                (ce.CatBoostEncoder(**cb_params), categorical_features)
            )
        if self._use_woe_encoding and any(categorical_features.values):
            woe_params = dict(handle_missing='value', handle_unknown='value')
            woe_params['randomized'] = lr_params.pop('randomized')
            woe_params['sigma'] = lr_params.pop('sigma_woe')
            woe_params['regularization'] = lr_params.pop('regularization')
            full_features_list.extend(list(cat_X.columns))
            import category_encoders as ce
            transformers.append(
                (ce.WOEEncoder(**woe_params), categorical_features)
            )
        if self._use_target_encoding and any(categorical_features.values):
            te_params = dict(handle_missing='value', handle_unknown='value')
            te_params['min_samples_leaf'] = lr_params.pop('min_samples_leaf')
            te_params['smoothing'] = lr_params.pop('smoothing')
            full_features_list.extend(list(cat_X.columns))
            import category_encoders as ce
            transformers.append(
                (ce.TargetEncoder(**te_params), categorical_features)
            )
        if self._use_target_encoding_other and any(categorical_features.values):
            full_features_list.extend(list(cat_X.columns))
            len_uniques = []
            cat_X_copy = cat_X.copy()
            test = pd.DataFrame(None, columns=cat_X.columns)
            for c in cat_X.columns:
                le = LabelEncoder()
                le.fit(pd.concat([categorical_features[c], test[c]]))
                cat_X_copy[c] = le.transform(cat_X_copy[c])
                # test[c] = le.transform(test[c])
                len_uniques.append(len(le.classes_))
            ALPHA = 75
            MAX_UNIQUE = max(len_uniques)
            FEATURES_COUNT = cat_X.shape[1]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params['random_state'])
            from target_encoding import TargetEncoder
            transformers.append(
                (TargetEncoder(alpha=ALPHA, max_unique=MAX_UNIQUE, used_features=FEATURES_COUNT, split=[cv]),
                 categorical_features)
            )
        if self._use_ohe_encoding and any(categorical_features.values):
            transformers.append(
                (OneHotEncoder(handle_unknown='ignore', sparse=True), categorical_features)
            )
        assert len(transformers) > 0, "should have some features"

        preprocess = make_column_transformer(*transformers)

        # estimator
        lr_defaults = dict(penalty='l2', dual=False, tol=1e-4, C=1.0,
                           fit_intercept=True, intercept_scaling=1, class_weight=None,
                           random_state=None, solver='warn', max_iter=100,
                           multi_class='warn', verbose=0, warm_start=False, n_jobs=None,
                           l1_ratio=None)
        allowed_lr_kwargs_keys = lr_defaults.keys()
        lr_params_copy = copy.deepcopy(lr_params)
        for k, v in lr_params_copy.items():
            if k not in allowed_lr_kwargs_keys:
                lr_params.pop(k, None)
        del lr_params_copy

        can_score = self.num_classes == 2 and 'AUC' in self.params_base['score_f_name'].upper()
        if can_score:
            scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
        else:
            scorer = None

        # pipeline
        if not self._cv_search:
            estimator = LogisticRegression(**lr_params)
            estimator_name = 'logisticregression'
        else:
            lr_params_cv = copy.deepcopy(lr_params)
            lr_params_cv['Cs'] = self.get_param_range(self.params['C'], self.params['mutation_count'], func_type='log')
            print("CV: Cs: %s" % str(lr_params_cv['Cs']))
            if 'l1_ratios' in lr_params:
                lr_params_cv['l1_ratios'] = self.get_param_range(self.params['l1_ratio'], self.params['mutation_count'], func_type='linear')
                print("CV: l1_ratios: %s" % str(lr_params_cv['l1_ratios']))
            lr_params_cv.pop('n_jobs', None)
            lr_params_cv.pop('C', None)
            lr_params_cv.pop('l1_ratio', None)
            if lr_params_cv['penalty'] == 'none':
                lr_params_cv['penalty'] = 'l2'
            estimator = LogisticRegressionCV(n_jobs=self.params['n_jobs'],
                                             cv=3, refit=True, scoring=scorer, **lr_params_cv)
            estimator_name = 'logisticregressioncv'
        model = make_pipeline(
            preprocess,
            estimator)

        # fit
        if self._grid_search_iterations and can_score:
            # WIP FIXME for multiclass and other scorers
            from sklearn.model_selection import GridSearchCV

            max_iter_range = self.get_param_range(self.params['max_iter'], self.params['mutation_count'],
                                                  range_limit=self._overfit_limit_iteration_step, func_type='log')
            print("max_iter_range: %s" % str(max_iter_range))
            param_grid = {
                '%s__max_iter' % estimator_name: max_iter_range,
            }
            grid_clf = GridSearchCV(model, param_grid, n_jobs=self.params['n_jobs'],
                                    cv=3, iid=True, refit=True, scoring=scorer)
            grid_clf.fit(X, y)
            model = grid_clf.best_estimator_
            print("best_index=%d best_score: %g best_params: %s" % (
                grid_clf.best_index_, grid_clf.best_score_, str(grid_clf.best_params_)))
        elif self._grid_search:
            # WIP
            from sklearn.model_selection import GridSearchCV

            param_grid = {
                'columntransformer__pipeline__simpleimputer__strategy': ['mean', 'median'],
                '%s__C' % estimator_name: [0.1, 0.5, 1.0],
            }
            grid_clf = GridSearchCV(model, param_grid, cv=10, iid=False)
            grid_clf.fit(X, y)
            model = grid_clf.best_estimator_
            # self.best_params = grid_clf.best_params_
        else:
            model.fit(X, y)

        lr_model = model.named_steps[estimator_name]

        # average importances over classes
        importances = np.average(np.array(lr_model.coef_), axis=0)
        # average iterations over classes (can't take max_iter per class)
        iterations = np.average(lr_model.n_iter_)

        # reduce OHE features to original names
        ohe_features_short = []
        if self._use_ohe_encoding and any(categorical_features.values):
            if self._use_ohe_encoding:
                ohe_features = pd.Series(
                    model.named_steps['columntransformer'].named_transformers_['onehotencoder'].get_feature_names(
                        input_features=cat_X.columns))

                def f(x):
                    return '_'.join(x.split('_')[:-1])

                # identify OHE features
                ohe_features_short = ohe_features.apply(lambda x: f(x))
                full_features_list.extend(list(ohe_features_short))

        msg = "num=%d cat=%d : ohe=%d : imp=%d full=%d" % (
            len(num_X.columns), len(cat_X.columns), len(ohe_features_short), len(importances), len(full_features_list))
        print(msg)
        assert len(importances) == len(full_features_list), msg

        # aggregate
        importances = pd.Series(np.abs(importances), index=full_features_list).groupby(level=0).mean()
        assert len(importances) == len(X_names)
        print("LRiterations: %d" % iterations)

        # save hyper parameter searched results for next search
        self.params['max_iter'] = iterations
        self.params['C'] = np.average(lr_model.C_, axis=0)
        if 'l1_ratios' in lr_params:
            self.params['l1_ratio'] = np.average(lr_model.l1_ratio_, axis=0)

        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=iterations)

    def get_param_range(self, param, mutation_count, range_limit=None, func_type='linear'):
        if func_type == 'log':
            f = np.log
            inv_f = np.exp
            bottom = 1.0
            top = 2.0
        else:
            f = np.abs
            inv_f = np.abs
            top = bottom = 1.0
        # bisect toward optimal param
        step_count = 3
        params_step = 2 + mutation_count
        start_range = param * (1.0 - bottom / params_step)
        end_range = param * (1.0 + top / params_step)
        if range_limit is not None:
            if end_range - start_range < range_limit:
                # if below some threshold, don't keep refining to avoid overfit
                return [param]
        start = f(start_range)
        end = f(end_range)
        step = 1.0 * (end - start) / step_count
        print(start, end, step)
        param_range = np.arange(start, end, step)
        if type(param) == int:
            param_range = [int(inv_f(x)) for x in param_range if int(inv_f(x)) > 0]
        else:
            param_range = [inv_f(x) for x in param_range if inv_f(x) > 0]
        if param not in param_range:
            param_range.append(param)
        param_range = sorted(param_range)
        return param_range

    def predict(self, X, **kwargs):
        X = dt.Frame(X)
        if self._impute_type == 'oob':
            for col in X.names:
                XX = X[:, col]
                XX.replace(None, self.min[col])
                X[:, col] = XX
        model, _, _, _ = self.get_model_properties()
        X = X.to_pandas()
        if self.num_classes == 1:
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return preds
