"""
Logistic Regression based upon sklearn.
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
from h2oaicore.transformers_more import CatTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score


class LogisticRegressionModel(CustomModel):
    """
    Logistic Regression

    Useful when weak or no interactions between features,
    or large inherent number of levels in categorical features

    Other useful DAI options if want to only use feature made internally by this model:
    config.prob_prune_genes = False
    config.prob_prune_by_features = False
    # Useful if want training to ultimately see all data with validated max_iter
    config.fixed_ensemble_level=0

    Recipe to do:

    1) Add separate LogisticRegressionEarlyStopping class to use warm start to take iterations a portion at a time,
    and score with known/given metric, and early stop to avoid overfitting on validation.

    2) Improve bisection stepping for search

    3) Consider from deployml.sklearn import LogisticRegressionBase

    4) Implement LinearRegression/ElasticNet (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)

    5) Implement other categorical missing encodings (same strategies as numerics)

    6) Implement other scorers (i.e. checking score_f_name -> sklearn metric or using DAI metrics)

    """
    # numerical imputation for all columns (could be done per column chosen by mutations)
    _impute_num_type = 'sklearn'  # best for linear models
    # _impute_num_type = 'oob'  # risky for linear models, but can be used for testing

    _impute_int_type = 'oob'

    _impute_bool_type = 'oob'
    _oob_bool = False

    # categorical imputation for all columns (could be done per column chosen by mutations)
    _impute_cat_type = 'oob'
    _oob_cat = "__OOB_CAT__"

    # unique identifier for OHE feature names
    _ohe_postfix = "_*#!^()^{}"

    # not required to be this strict, but good starting point to only use this recipe's features
    _included_transformers = ['CatOriginalTransformer', 'OriginalTransformer', 'CatTransformer']
    _can_handle_non_numeric = True  # tell DAI we can handle non-numeric (i.e. strings)
    _can_handle_categorical = True  # tell DAI we can handle numerically encoded categoricals for use as categoricals
    _num_as_cat = False  # treating numeric as categorical best handled per column, but can force all numerics as cats

    _mutate_all = True  # tell DAI we fully controls mutation
    _mutate_by_one = False  # tell our recipe only changes one key at a time, can limit exploration if set as True
    _always_defaults = False
    _randomized_random_state = False
    _overfit_limit_iteration_step = 10

    # tell DAI want to keep track of self.params changes during fit, and to average numeric values across folds (if any)
    _used_return_params = True
    _average_return_params = True

    # other DAI vars
    _regression = False
    _binary = True
    _multiclass = True
    _parallel_task = True  # set to False may lead to faster performance if not doing grid search or cv search
    _fit_by_iteration = True
    _fit_iteration_name = 'max_iter'
    _display_name = "LR"
    _description = "Logistic Regression"

    # recipe vars for encoding choices
    _use_numerics = True
    _use_ohe_encoding = True
    _use_target_encoding = False
    _use_target_encoding_other = False
    _use_ordinal_encoding = False
    _use_catboost_encoding = False  # Note: Requires data be randomly shuffled so target is not in special order
    _use_woe_encoding = False

    # tell DAI what pip modules we will use
    _modules_needed_by_name = ['category_encoders']
    if _use_target_encoding_other:
        _modules_needed_by_name.extend(['target_encoding'])

    def set_default_params(self, accuracy=10, time_tolerance=10,
                           interpretability=1, **kwargs):
        # Fill up parameters we care about
        self.params = {}
        self.mutate_params(get_default=True, accuracy=accuracy, time_tolerance=time_tolerance,
                           interpretability=interpretability, **kwargs)

    def mutate_params(self, accuracy=10, time_tolerance=10, interpretability=1, **kwargs):
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

        # keep track of fit count, for other control over hyper parameter search in this recipe
        if 'fit_count' not in self.params:
            self.params['fit_count'] = 0

        self.params['random_state'] = kwargs.get("random_state", 1234)
        if self._randomized_random_state:
            self.params['random_state'] = random.randint(0, 32000)
        self.params['n_jobs'] = self.params_base.get('n_jobs', max(1, physical_cores_count))

        # Modify certain parameters for tuning
        C_list = [0.05, 0.075, 0.1, 0.15, 0.2, 1.0, 5.0]
        self.params["C"] = float(np.random.choice(C_list)) if not get_default else 0.1

        tol_list = [1e-4, 1e-3, 1e-5]
        default_tol = 1e-6 if accuracy >= 6 else 1e-4
        if default_tol not in tol_list:
            tol_list.append(default_tol)
        self.params["tol"] = float(np.random.choice(tol_list)) if not get_default else default_tol

        # solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        # newton-cg too slow
        # sag too slow
        # solver_list = ['lbfgs', 'liblinear', 'saga']
        solver_list = ['lbfgs']
        self.params["solver"] = str(np.random.choice(solver_list)) if not get_default else 'lbfgs'

        max_iter_list = [50, 100, 150, 200, 250, 300]
        self.params["max_iter"] = int(np.random.choice(max_iter_list)) if not get_default else 200

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

        # control search in recipe
        self.params['grid_search_iterations'] = accuracy >= 7
        # cv search for hyper parameters, can be used in conjunction with _grid_search_by_iterations = True or False
        self.params['cv_search'] = accuracy >= 8

        if self._mutate_by_one and not get_default and params_orig:
            pick_key = str(np.random.choice(list(self.params.keys()), size=1)[0])
            value = self.params[pick_key]
            self.params = copy.deepcopy(params_orig)
            self.params[pick_key] = value
            # validate parameters to avoid single key leading to invalid overall parameters
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

        # save pre-datatable-imputed X
        X_dt = X

        # Apply OOB imputation
        self.oob_imputer = OOBImpute(self._impute_num_type, self._impute_int_type, self._impute_bool_type,
                                     self._impute_cat_type, self._oob_bool, self._oob_cat)
        X = self.oob_imputer.fit_transform(X)

        # convert to pandas for sklearn
        X = X.to_pandas()
        # print("LR: pandas dtypes: %s" % (str(list(X.dtypes))))
        X_names = list(X.columns)

        # FEATURE GROUPS

        # Choose which features are numeric or categorical
        cat_features = [x for x in X_names if CatOriginalTransformer.is_me_transformed(x)]
        catlabel_features = [x for x in X_names if CatTransformer.is_me_transformed(x)]
        # can add explicit column name list to below force_cats
        force_cats = cat_features + catlabel_features

        # choose if numeric is treated as categorical
        if not self._num_as_cat:
            numerical_features = (X.dtypes == 'float') | (X.dtypes == 'float32') | (X.dtypes == 'float64')
        else:
            numerical_features = X.dtypes == 'invalid'
            # force oob imputation for numerics
            self.oob_imputer = OOBImpute('oob', 'oob', 'oob',
                                         self._impute_cat_type, self._oob_bool, self._oob_cat)
            X = self.oob_imputer.fit_transform(X_dt)
            X = X.to_pandas()

        categorical_features = ~numerical_features
        # below can lead to overlap between what is numeric and what is categorical
        more_cats = (pd.Series([True if x in force_cats else False for x in list(categorical_features.index)], index=categorical_features.index))
        categorical_features = (categorical_features) | (more_cats)
        cat_X = X.loc[:, categorical_features]
        num_X = X.loc[:, numerical_features]
        # print("LR: Cat names: %s" % str(list(cat_X.columns)))
        # print("LR: Num names: %s" % str(list(num_X.columns)))

        # TRANSFORMERS
        lr_params = copy.deepcopy(self.params)
        lr_params.pop('grid_search_by_iterations', None)
        lr_params.pop('cv_search', None)
        grid_search = False  # WIP

        full_features_list = []
        transformers = []
        if self._use_numerics and any(numerical_features.values):
            impute_params = {}
            impute_params['strategy'] = lr_params.pop('strategy', 'mean')
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

        # ESTIMATOR
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
        # print("LR: can_score: %s" % str(can_score))
        if can_score:
            scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
        else:
            scorer = None

        if not ('C' in lr_params or 'l1_ratios' in lr_params):
            # override
            self.params['cv_search'] = False

        if not self.params['cv_search']:
            estimator = LogisticRegression(**lr_params)
            estimator_name = 'logisticregression'
        else:
            lr_params_cv = copy.deepcopy(lr_params)
            if 'C' in lr_params:
                lr_params_cv['Cs'] = self.get_param_range(self.params['C'], self.params['fit_count'], func_type='log')
                # print("LR: CV: Cs: %s" % str(lr_params_cv['Cs']))
            if 'l1_ratios' in lr_params:
                lr_params_cv['l1_ratios'] = self.get_param_range(self.params['l1_ratio'], self.params['fit_count'],
                                                                 func_type='linear')
                # print("LR: CV: l1_ratios: %s" % str(lr_params_cv['l1_ratios']))
            lr_params_cv.pop('n_jobs', None)
            lr_params_cv.pop('C', None)
            lr_params_cv.pop('l1_ratio', None)
            if lr_params_cv['penalty'] == 'none':
                lr_params_cv['penalty'] = 'l2'
            estimator = LogisticRegressionCV(n_jobs=self.params['n_jobs'],
                                             cv=3, refit=True, scoring=scorer, **lr_params_cv)
            estimator_name = 'logisticregressioncv'

        # PIPELINE
        model = make_pipeline(
            preprocess,
            estimator)

        # FIT
        if self.params['grid_search_iterations'] and can_score:
            # WIP FIXME for multiclass and other scorers
            from sklearn.model_selection import GridSearchCV

            max_iter_range = self.get_param_range(self.params['max_iter'], self.params['fit_count'],
                                                  range_limit=self._overfit_limit_iteration_step, func_type='log')
            # print("LR: max_iter_range: %s" % str(max_iter_range))
            param_grid = {
                '%s__max_iter' % estimator_name: max_iter_range,
            }
            grid_clf = GridSearchCV(model, param_grid, n_jobs=self.params['n_jobs'],
                                    cv=3, iid=True, refit=True, scoring=scorer)
            grid_clf.fit(X, y)
            model = grid_clf.best_estimator_
            # print("LR: best_index=%d best_score: %g best_params: %s" % (
            #    grid_clf.best_index_, grid_clf.best_score_, str(grid_clf.best_params_)))
        elif grid_search:
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

        # get actual LR model
        lr_model = model.named_steps[estimator_name]

        # average importances over classes
        importances = np.average(np.array(lr_model.coef_), axis=0)
        # average iterations over classes (can't take max_iter per class)
        iterations = np.average(lr_model.n_iter_)
        # print("LR: iterations: %d" % iterations)

        # reduce OHE features to original names
        ohe_features_short = []
        if self._use_ohe_encoding and any(categorical_features.values):
            if self._use_ohe_encoding:
                input_features = [x + self._ohe_postfix for x in cat_X.columns]
                ohe_features = pd.Series(
                    model.named_steps['columntransformer'].named_transformers_['onehotencoder'].get_feature_names(
                        input_features=input_features))

                def f(x):
                    return '_'.join(x.split(self._ohe_postfix + '_')[:-1])

                # identify OHE features
                ohe_features_short = ohe_features.apply(lambda x: f(x))
                full_features_list.extend(list(ohe_features_short))

        msg = "LR: num=%d cat=%d : ohe=%d : imp=%d full=%d" % (
            len(num_X.columns), len(cat_X.columns), len(ohe_features_short), len(importances), len(full_features_list))
        # print(msg)
        assert len(importances) == len(full_features_list), msg

        # aggregate importances
        importances = pd.Series(np.abs(importances), index=full_features_list).groupby(level=0).mean()
        assert len(importances) == len(X_names), "%d %d %s" % (len(importances), len(X_names), msg)

        # save hyper parameter searched results for next search
        self.params['max_iter'] = iterations
        if self.params['cv_search']:
            self.params['C'] = np.average(lr_model.C_, axis=0)
        if 'l1_ratios' in lr_params and self.params['cv_search']:
            self.params['l1_ratio'] = np.average(lr_model.l1_ratio_, axis=0)
        if 'fit_count' in self.params:
            self.params['fit_count'] += 1
        else:
            self.params['fit_count'] = 0

        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=iterations)

    def get_param_range(self, param, fit_count, range_limit=None, func_type='linear'):
        if func_type == 'log':
            f = np.log
            inv_f = np.exp
            bottom = 1.0
            top = 1.0
        else:
            f = np.abs
            inv_f = np.abs
            top = bottom = 1.0
        # bisect toward optimal param
        step_count = 3
        params_step = 2 + fit_count
        start_range = param * (1.0 - bottom / params_step)
        end_range = param * (1.0 + top / params_step)
        if range_limit is not None:
            if end_range - start_range < range_limit:
                # if below some threshold, don't keep refining to avoid overfit
                return [param]
        start = f(start_range)
        end = f(end_range)
        step = 1.0 * (end - start) / step_count
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
        X = self.oob_imputer.transform(X)
        model, _, _, _ = self.get_model_properties()
        X = X.to_pandas()
        if self.num_classes == 1:
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return preds


class OOBImpute(object):
    def __init__(self, impute_num_type, impute_int_type, impute_bool_type, impute_cat_type, oob_bool, oob_cat):
        self._impute_num_type = impute_num_type
        self._impute_int_type = impute_int_type
        self._impute_bool_type = impute_bool_type
        self._impute_cat_type = impute_cat_type
        self._oob_bool = oob_bool
        self._oob_cat = oob_cat

    def fit(self, X: dt.Frame):
        # just ignore output
        self.fit_transform(X)

    def fit_transform(self, X: dt.Frame):
        # IMPUTE
        # print("LR: types number of columns: %d : %d %d %d %d" % (len(X.names), len(X[:, [float]].names), len(X[:, [int]].names), len(X[:, [bool]].names), len(X[:, [str]].names)))
        for col in X[:, [float]].names:
            XX = X[:, col]
            XX.replace(None, np.nan)
            X[:, col] = XX
        if self._impute_num_type == 'oob':
            # Replace missing values with a value smaller than all observed values
            self.min = dict()
            for col in X[:, [float]].names:
                XX = X[:, col]
                self.min[col] = XX.min1()
                if self.min[col] is None or np.isnan(self.min[col]):
                    self.min[col] = -1e10
                else:
                    self.min[col] -= 1
                XX.replace(None, self.min[col])
                X[:, col] = XX
                assert X[dt.isna(dt.f[col]), col].nrows == 0
        if self._impute_int_type == 'oob':
            # Replace missing values with a value smaller than all observed values
            self.min_int = dict()
            for col in X[:, [int]].names:
                XX = X[:, col]
                self.min_int[col] = XX.min1()
                if self.min_int[col] is None or np.isnan(self.min_int[col]):
                    self.min_int[col] = 0
                XX.replace(None, self.min_int[col])
                X[:, col] = XX
                assert X[dt.isna(dt.f[col]), col].nrows == 0
        if self._impute_bool_type == 'oob':
            for col in X[:, [bool]].names:
                XX = X[:, col]
                XX.replace(None, self._oob_bool)
                X[:, col] = XX
                assert X[dt.isna(dt.f[col]), col].nrows == 0
        if self._impute_cat_type == 'oob':
            for col in X[:, [str]].names:
                XX = X[:, col]
                XX.replace(None, self._oob_cat)
                X[:, col] = XX
                assert X[dt.isna(dt.f[col]), col].nrows == 0
        return X

    def transform(self, X: dt.Frame):
        if self._impute_num_type == 'oob':
            for col in X[:, [float]].names:
                XX = X[:, col]
                XX.replace(None, self.min[col])
                X[:, col] = XX
        if self._impute_int_type == 'oob':
            for col in X[:, [int]].names:
                XX = X[:, col]
                XX.replace(None, self.min_int[col])
                X[:, col] = XX
        if self._impute_bool_type == 'oob':
            for col in X[:, [bool]].names:
                XX = X[:, col]
                XX.replace(None, self._oob_bool)
                X[:, col] = XX
        if self._impute_cat_type == 'oob':
            for col in X[:, [str]].names:
                XX = X[:, col]
                XX.replace(None, self._oob_cat)
                X[:, col] = XX
        return X
