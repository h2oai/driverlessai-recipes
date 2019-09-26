"""
Logistic Regression based upon sklearn.
"""
import datatable as dt
import numpy as np
import random
import pandas as pd
import os
import copy
import codecs
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, make_scorer

from h2oaicore.models import CustomModel
from h2oaicore.systemutils import config, physical_cores_count, save_obj, load_obj, DefaultOrderedDict
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from h2oaicore.transformers import CatOriginalTransformer, FrequentTransformer, CVTargetEncodeTransformer
from h2oaicore.transformer_utils import Transformer
from h2oaicore.transformers_more import CatTransformer, LexiLabelEncoderTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier


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
    _kaggle = False  # some kaggle specific optimizations for https://www.kaggle.com/c/cat-in-the-dat
    # with _kaggle_features=False and no catboost features:
    # gives 0.8043 DAI validation for some seeds/runs,
    # which leads to 0.80802 public score after only 2 minutes of running on accuracy=2, interpretability=1

    # with _kaggle_features=False and catboost features:
    # gives 0.8054 DAI validation for some seeds/runs,
    # which leads to 0.80814 public score after only 10 minutes of running on accuracy=7, interpretability=1

    # whether to generate features for kaggle
    # these features do not help the score, but do make sense as plausible features to build
    _kaggle_features = False

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
    if _kaggle and 'CatTransformer' in _included_transformers:
        # Just handle all cats directly
        _included_transformers.remove('CatTransformer')
    _can_handle_non_numeric = True  # tell DAI we can handle non-numeric (i.e. strings)
    _can_handle_categorical = True  # tell DAI we can handle numerically encoded categoricals for use as categoricals
    _num_as_cat = False or _kaggle  # treating numeric as categorical best handled per column, but can force all numerics as cats
    _num_as_num = False

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
    _allow_basis_of_default_individuals = False
    _fs_permute_must_use_self = True
    _check_stall = False  # avoid stall check, joblib loky stuff detatches sometimes

    # recipe vars for encoding choices
    _use_numerics = True
    _use_ohe_encoding = True
    _use_target_encoding = False
    _use_target_encoding_other = False
    _use_ordinal_encoding = False
    _use_catboost_encoding = False or _kaggle  # Note: Requires data be randomly shuffled so target is not in special order
    _use_woe_encoding = False

    # tell DAI what pip modules we will use
    _modules_needed_by_name = ['category_encoders']
    if _use_target_encoding_other:
        _modules_needed_by_name.extend(['target_encoding'])
        # _modules_needed_by_name.extend(['git+https://github.com/h2oai/target_encoding#egg=target_encoding'])

    # whether to show debug prints and write munged view to disk
    _debug = True
    # wehther to cache feature results, only by transformer instance and X shape, so risky to use without care.
    _cache = False

    _ensemble = False

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
        if self._kaggle:
            C_list = [0.095, 0.1, 0.115, 0.11, 0.105, 0.12, 0.125, 0.13, 0.14]
        else:
            C_list = [0.05, 0.075, 0.1, 0.15, 0.2, 1.0, 5.0]
        self.params["C"] = float(np.random.choice(C_list)) if not get_default else 0.12

        tol_list = [1e-4, 1e-3, 1e-5]
        if accuracy < 5:
            default_tol = 1e-4
        elif accuracy < 6:
            default_tol = 1e-5
        elif accuracy <= 7:
            default_tol = 1e-6
        else:
            default_tol = 1e-7
        if self._kaggle:
            default_tol = 1e-8
        if default_tol not in tol_list:
            tol_list.append(default_tol)
        self.params["tol"] = float(np.random.choice(tol_list)) if not (self._kaggle or get_default) else default_tol

        # solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        # newton-cg too slow
        # sag too slow
        # solver_list = ['lbfgs', 'liblinear', 'saga']
        solver_list = ['lbfgs']
        self.params["solver"] = str(np.random.choice(solver_list)) if not get_default else 'lbfgs'

        if self._kaggle:
            max_iter_list = [300, 350, 400, 450, 500, 700, 800, 900, 1000]
        else:
            max_iter_list = [150, 175, 200, 225, 250, 300]
        self.params["max_iter"] = int(np.random.choice(max_iter_list)) if not get_default else 700
        # self.params["max_iter"] = 37

        if self.params["solver"] in ['lbfgs', 'newton-cg', 'sag']:
            penalty_list = ['l2', 'none']
        elif self.params["solver"] in ['saga']:
            penalty_list = ['l1', 'l2', 'none']
        elif self.params["solver"] in ['liblinear']:
            penalty_list = ['l1']
        else:
            raise RuntimeError("No such solver: %s" % self.params['solver'])
        self.params["penalty"] = str(np.random.choice(penalty_list)) if not (self._kaggle or get_default) else 'l2'

        if self.params["penalty"] == 'elasticnet':
            l1_ratio_list = [0, 0.25, 0.5, 0.75, 1.0]
            self.params["l1_ratio"] = float(np.random.choice(l1_ratio_list))
        else:
            self.params.pop('l1_ratio', None)
        if self.params["penalty"] == 'none':
            self.params.pop('C', None)
        else:
            self.params['C'] = float(np.random.choice(C_list)) if not get_default else 0.12
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
            if self._kaggle:
                sigma_list = [None, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]
            else:
                sigma_list = [None, 0.01, 0.05, 0.1, 0.5]
            self.params['sigma'] = random.choice(sigma_list)

        if self._use_woe_encoding:
            randomized_list = [True, False]
            self.params['randomized'] = random.choice(randomized_list)
            sigma_woe_list = [0.05, 0.001, 0.01, 0.1, 0.005]
            self.params['sigma_woe'] = random.choice(sigma_woe_list)
            regularization_list = [1.0, 0.1, 2.0]
            self.params['regularization'] = random.choice(regularization_list)

        # control search in recipe
        self.params['grid_search_iterations'] = accuracy >= 8
        # cv search for hyper parameters, can be used in conjunction with _grid_search_by_iterations = True or False
        self.params['cv_search'] = accuracy >= 9

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

        min_count = np.min(np.unique(y, return_counts=True)[1])
        if min_count < 9:
            self.params['cv_search'] = False
        if min_count < 3:
            self.params['grid_search_iterations'] = False
            self.params['cv_search'] = False

        if self._ensemble:
            self.params['grid_search_iterations'] = False
            self.params['cv_search'] = False

        # save pre-datatable-imputed X
        X_dt = X

        # Apply OOB imputation
        self.oob_imputer = OOBImpute(self._impute_num_type, self._impute_int_type, self._impute_bool_type,
                                     self._impute_cat_type, self._oob_bool, self._oob_cat)
        X = self.oob_imputer.fit_transform(X)

        # convert to pandas for sklearn
        X = X.to_pandas()
        X_orig_cols_names = list(X.columns)
        if self._kaggle_features:
            self.features = make_features(cache=self._cache)
            X = self.features.fit_transform(X, y)
        else:
            self.features = None
        # print("LR: pandas dtypes: %s" % (str(list(X.dtypes))))

        # FEATURE GROUPS

        # Choose which features are numeric or categorical
        cat_features = [x for x in X_orig_cols_names if CatOriginalTransformer.is_me_transformed(x)]
        catlabel_features = [x for x in X_orig_cols_names if CatTransformer.is_me_transformed(x)]
        # can add explicit column name list to below force_cats
        force_cats = cat_features + catlabel_features

        actual_numerical_features = (X.dtypes == 'float') | (X.dtypes == 'float32') | (X.dtypes == 'float64')# | (X.dtypes == 'int') | (X.dtypes == 'int32') | (X.dtypes == 'int64') | (X.dtypes == 'bool')
        # choose if numeric is treated as categorical
        if not self._num_as_cat or self._num_as_num:
            # treat (e.g.) binary as both numeric and categorical
            numerical_features = copy.deepcopy(actual_numerical_features)
        else:
            # no numerics
            numerical_features = X.dtypes == 'invalid'

        if self._num_as_cat:
            # then can't have None sent to cats, impute already up front
            # force oob imputation for numerics
            self.oob_imputer = OOBImpute('oob', 'oob', 'oob',
                                         self._impute_cat_type, self._oob_bool, self._oob_cat)
            X = self.oob_imputer.fit_transform(X_dt)
            X = X.to_pandas()
            if self._kaggle_features:
                X = self.features.fit_transform(X, y)
        if self._kaggle_features:
            numerical_features = self.features.update_numerical_features(numerical_features)

        if not self._num_as_cat:
            # then cats are only things that are not numeric
            categorical_features = ~actual_numerical_features
        else:
            # then everything is a cat
            categorical_features = ~numerical_features  # (X.dtypes == 'invalid')
        # below can lead to overlap between what is numeric and what is categorical
        more_cats = (pd.Series([True if x in force_cats else False for x in list(categorical_features.index)],
                               index=categorical_features.index))
        categorical_features = (categorical_features) | (more_cats)
        if self._kaggle_features:
            categorical_features = self.features.update_categorical_features(categorical_features)

        cat_X = X.loc[:, categorical_features]
        num_X = X.loc[:, numerical_features]
        if self._debug:
            print("LR: Cat names: %s" % str(list(cat_X.columns)))
            print("LR: Num names: %s" % str(list(num_X.columns)))

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
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.params['random_state'])
            split_cv = [cv]
            # split_cv = [3, 3]
            ALPHA, MAX_UNIQUE, FEATURES_COUNT = get_TE_params(cat_X, debug=self._debug)
            from target_encoding import TargetEncoder
            transformers.append(
                (TargetEncoder(alpha=ALPHA, max_unique=MAX_UNIQUE, split_in=split_cv),
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
        if not self._ensemble:
            model = make_pipeline(
                preprocess,
                estimator, memory="./")
        else:
            ALPHA, MAX_UNIQUE, FEATURES_COUNT = get_TE_params(cat_X, debug=self._debug)
            from target_encoding import TargetEncoderClassifier
            te_estimator = TargetEncoderClassifier(alpha=ALPHA, max_unique=MAX_UNIQUE, used_features=FEATURES_COUNT)
            estimators = [(estimator_name, estimator), ('teclassifier', te_estimator)]
            model = make_pipeline(
                preprocess,
                VotingClassifier(estimators))

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
        importances = np.average(np.fabs(np.array(lr_model.coef_)), axis=0)
        # average iterations over classes (can't take max_iter per class)
        iterations = int(np.average(lr_model.n_iter_))
        # print("LR: iterations: %d" % iterations)

        if self._debug:
            full_features_list_copy = copy.deepcopy(full_features_list)

        # reduce OHE features to original names
        ohe_features_short = []
        if self._use_ohe_encoding and any(categorical_features.values):
            input_features = [x + self._ohe_postfix for x in cat_X.columns]
            ohe_features = pd.Series(
                model.named_steps['columntransformer'].named_transformers_['onehotencoder'].get_feature_names(
                    input_features=input_features))

            def f(x):
                return '_'.join(x.split(self._ohe_postfix + '_')[:-1])

            # identify OHE features
            ohe_features_short = ohe_features.apply(lambda x: f(x))
            full_features_list.extend(list(ohe_features_short))

            if self._debug:
                full_features_list_copy.extend(list(ohe_features))
                imp = pd.Series(importances, index=full_features_list_copy).sort_values(ascending=False)
                import uuid
                struuid = str(uuid.uuid4())
                imp.to_csv("prepreimp_%s.csv" % struuid)


        if self._debug:
            imp = pd.Series(importances, index=full_features_list).sort_values(ascending=False)
            import uuid
            struuid = str(uuid.uuid4())
            imp.to_csv("preimp_%s.csv" % struuid)

        # aggregate our own features
        if self._kaggle_features:
            full_features_list = self.features.aggregate(full_features_list, importances)

        msg = "LR: num=%d cat=%d : ohe=%d : imp=%d full=%d" % (
            len(num_X.columns), len(cat_X.columns), len(ohe_features_short), len(importances), len(full_features_list))
        if self._debug:
            print(msg)
        assert len(importances) == len(full_features_list), msg
        if self._debug:
            imp = pd.Series(importances, index=full_features_list).sort_values(ascending=False)
            import uuid
            struuid = str(uuid.uuid4())
            imp.to_csv("imp_%s.csv" % struuid)

        # aggregate importances by dai feature name
        importances = pd.Series(np.abs(importances), index=full_features_list).groupby(level=0).mean()
        assert len(importances) == len(X_orig_cols_names), "lenimp=%d lenorigX=%d msg=%s : X.columns=%s dtypes=%s : full_features_list=%s" % (
            len(importances), len(X_orig_cols_names), msg,
            str(list(X.columns)), str(list(X.dtypes)), str(full_features_list))

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

        importances_list = importances.tolist()
        importances_list = list(np.array(importances_list) / np.max(importances_list))
        self.set_model_properties(model=(model, self.features),
                                  features=orig_cols,
                                  importances=importances_list,
                                  iterations=iterations)
        self.features = None

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
        model_tuple, _, _, _ = self.get_model_properties()
        model, features = model_tuple
        X = X.to_pandas()
        if self._kaggle_features and features is not None:
            X = features.transform(X)

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


class make_features(object):
    _postfix = "@%@(&#%@))){}#"

    def __init__(self, cache=False):
        self.cache = cache
        self.dai_te = False
        self.other_te = True

        self.new_names_dict = {}
        self.raw_names_dict = {}
        self.raw_names_dict_reversed = {}
        self.spring = None
        self.summer = None
        self.fall = None
        self.winter = None

        self.monthcycle1 = None
        self.monthcycle2 = None

        self.weekend = None
        self.daycycle1 = None
        self.daycycle2 = None

        self.lexi = None
        self.ord5sorted = None

        self.ord5more1 = None
        self.ord5more2 = None

    def apply_clone(self, src):
        for k, v in src.__dict__.items():
            setattr(self, k, v)

    def fit_transform(self, X: pd.DataFrame, y=None, transform=False):
        self.orig_cols = list(X.columns)
        self.raw_names_dict = {Transformer.raw_feat_name(v): v for v in list(X.columns)}
        self.raw_names_dict_reversed = {v: k for k, v in self.raw_names_dict.items()}

        file = "munged_%s_%s_%d_%d.csv" % (__name__, transform, X.shape[0], X.shape[1])
        file = file.replace("csv", "pkl")
        file2 = file.replace("munged", "clone")
        if self.cache and os.path.isfile(file) and os.path.isfile(file2):
            #X = pd.read_csv(file, sep=',', header=0)
            X = load_obj(file)
            X = X.drop("target", axis=1, errors='ignore')
            if not transform:
                self.apply_clone(load_obj(file2))
            return X

        if 'bin_0' in self.raw_names_dict:
            X.drop(self.raw_names_dict['bin_0'], errors='ignore')
        if 'bin_3' in self.raw_names_dict:
            X.drop(self.raw_names_dict['bin_3'], errors='ignore')

        # use circular color wheel position for nom_0
        def nom12num(x):
            # use number of sides
            d = {'Circle': 0, 'Polygon': -1, 'Star': 10, 'Triangle': 3, 'Square': 4, 'Trapezoid': 5}
            return d[x]

        X, self.sides = self.make_feat(X, 'nom_1', 'sides', nom12num)

        def nom22num(x):
            # use family level features expanded encoding or relative size for nom_2
            # ordered by height
            d = {'Snake': 0, 'Axolotl': 1, 'Hamster': 2, 'Cat': 3, 'Dog': 4, 'Lion': 5}
            return d[x]

        X, self.animal = self.make_feat(X, 'nom_2', 'animal', nom22num)

        #def has_char(x, char):
        #    x_str = str(x)
        #    return 1 if char.upper() in x_str.upper() else 0

        #self.haschars = [None] * len(self.orig_cols)
        #for ni, c in enumerate(self.orig_cols):
        #    X, self.lenfeats[ni] = self.make_feat(X, c, 'len', get_len)

        def get_len(x):
            x_str = str(x)
            return len(x_str)

        self.lenfeats = [None] * len(self.orig_cols)
        for ni, c in enumerate(self.orig_cols):
            X, self.lenfeats[ni] = self.make_feat(X, c, 'len', get_len)
        #
        def get_first(x):
            x_str = str(x)
            return x_str[0] if len(x_str) > 0 else ""

        self.firstchar = [None] * len(self.orig_cols)
        for ni, c in enumerate(self.orig_cols):
            X, self.firstchar[ni] = self.make_feat(X, c, 'firstc', get_first, is_float=False)

        #
        def get_last(x):
            x_str = str(x)
            return x_str[-1] if len(x_str) > 0 else ""

        self.lastchar = [None] * len(self.orig_cols)
        for ni, c in enumerate(self.orig_cols):
            X, self.lastchar[ni] = self.make_feat(X, c, 'lastc', get_last, is_float=False)

        #
        hex_strings = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']
        #
        if True:
            # convert hex to binary and use as 8-feature (per hex feature) encoding
            def get_charnum(x, i=None):
                return str(x)[i]

            width = 9
            self.hexchar = [None] * len(hex_strings) * width
            for ni, c in enumerate(hex_strings):
                for nii in range(0, width):
                    X, self.hexchar[ni * width + nii] = self.make_feat(X, c, 'hexchar%d' % nii, get_charnum, is_float=False, i=nii)
        #
        def hex_to_int(x):
            x_int = int(eval('0x' + str(x)))
            return x_int

        self.hexints = [None] * len(hex_strings)
        for ni, c in enumerate(hex_strings):
            X, self.hexints[ni] = self.make_feat(X, c, 'hex2int', hex_to_int)

        #
        if False:  # ValueError: could not convert string to float: b'\x05\x0f\x11k\xcf'
            def hex_to_string(x):
                try:
                    x_str = codecs.decode('0' + x, 'hex')
                except:
                    x_str = codecs.decode(x, 'hex')
                return x_str

            self.hexstr = [None] * len(hex_strings)
            for ni, c in enumerate(hex_strings):
                X, self.hexstr[ni] = self.make_feat(X, c, 'hex2str', hex_to_string, is_float=False)

        def bin012a(x):
            return bool(x[0]) & bool(x[1]) & bool(x[2])

        X, self.bin012a = self.make_feat(X, ['bin_0', 'bin_1', 'bin_2'], 'bin012a', bin012a)

        def bin012b(x):
            return (bool(x[0]) ^ bool(x[1])) ^ bool(x[2])

        X, self.bin012b = self.make_feat(X, ['bin_0', 'bin_1', 'bin_2'], 'bin012b', bin012b)

        def bin012c(x):
            return bool(x[0]) ^ (bool(x[1]) ^ bool(x[2]))

        X, self.bin012c = self.make_feat(X, ['bin_0', 'bin_1', 'bin_2'], 'bin012c', bin012c)

        # TODO: manual OHE fixed width for out of 16 digits always (not sure all rows lead to all values)

        # one-hot encode text by each character

        # use geo-location for nom_3

        # use static mapping encoding for ord_2 and ord_1
        def ord12num1(x):
            # ordered label
            d = {'Novice': 0, 'Contributor': 1, 'Expert': 2, 'Master': 3, 'Grandmaster': 4}
            return d[x]

        X, self.kaggle1 = self.make_feat(X, 'ord_1', 'kaggle1', ord12num1)

        def ord12num2(x):
            # medals total
            d = {'Novice': 0, 'Contributor': 0, 'Expert': 2, 'Master': 3, 'Grandmaster': 6}
            return d[x]

        X, self.kaggle2 = self.make_feat(X, 'ord_1', 'kaggle2', ord12num2)

        def ord1master(x):
            return 1 if 'master' in x or 'Master' in x else 0

        X, self.kaggle3 = self.make_feat(X, 'ord_1', 'kaggle3', ord1master)

        def ord22num(x):
            # ordered label
            d = {'Freezing': 0, 'Cold': 1, 'Warm': 2, 'Hot': 3, 'Boiling Hot': 4, 'Lava Hot': 5}
            return d[x]

        X, self.temp1 = self.make_feat(X, 'ord_2', 'temp1', ord22num)

        def ord22num2(x):
            # temp in F
            d = {'Freezing': 32, 'Cold': 50, 'Warm': 80, 'Hot': 100, 'Boiling Hot': 212, 'Lava Hot': 1700}
            return d[x]

        X, self.temp2 = self.make_feat(X, 'ord_2', 'temp2', ord22num2)

        def ord2hot(x):
            return 1 if 'hot' in x or 'Hot' in x else 0

        X, self.temp4 = self.make_feat(X, 'ord_2', 'temp4', ord2hot)

        # lower ord_5
        def ord5more0(x):
            return x.lower()

        X, self.ord5more0 = self.make_feat(X, 'ord_5', 'more0', ord5more0, is_float=False)

        # 1st char, keep for OHE
        def ord5more1(x):
            return x[0]

        X, self.ord5more1 = self.make_feat(X, 'ord_5', 'more1', ord5more1, is_float=False)

        # 2nd char, keep for OHE
        def ord5more2(x):
            return x[1]

        X, self.ord5more2 = self.make_feat(X, 'ord_5', 'more2', ord5more2, is_float=False)

        # 1st char, keep for OHE
        def ord5more3(x):
            return x[0].lower()

        X, self.ord5more3 = self.make_feat(X, 'ord_5', 'more3', ord5more3, is_float=False)

        # 2nd char, keep for OHE
        def ord5more4(x):
            return x[1].lower()

        X, self.ord5more4 = self.make_feat(X, 'ord_5', 'more4', ord5more4, is_float=False)

        # 1st word, keep for OHE
        def ord2more1(x):
            return x.split(" ")[0]

        X, self.ord2more1 = self.make_feat(X, 'ord_2', 'more1', ord2more1, is_float=False)

        # 2nd word, keep for OHE
        def ord2more2(x):
            a = x.split(" ")
            if len(a) > 1:
                return a[1]
            else:
                return a[0]

        X, self.ord2more2 = self.make_feat(X, 'ord_2', 'more2', ord2more2, is_float=False)

        # use lexi LE directly as integers for alphabetical (ord_5, ord_4, ord_3)
        orig_feat_names = ['ord_5', 'ord_4', 'ord_3',
                           'nom_0', 'nom_1', 'nom_2',
                           'nom_3', 'nom_4', 'nom_5',
                           'nom_6', 'nom_7', 'nom_8',
                           'nom_9', 'ord_1', 'ord_2']
        orig_feat_names = [self.raw_names_dict_reversed[x] for x in list(self.orig_cols)]  # try just encoding all columns
        new_names = ['lexi%d' % x for x in range(len(orig_feat_names))]
        if not transform:
            self.lexi = [None] * len(orig_feat_names)
            self.lexi_names = [None] * len(orig_feat_names)
        for ni, (new_name, orig_feat_name) in enumerate(zip(new_names, orig_feat_names)):
            if orig_feat_name in self.raw_names_dict and self.raw_names_dict[orig_feat_name] in X.columns:
                dai_feat_name = self.raw_names_dict[orig_feat_name]
                if transform:
                    Xnew = self.lexi[ni].transform(X[[dai_feat_name]])
                else:
                    self.lexi[ni] = LexiLabelEncoderTransformer([dai_feat_name])
                    Xnew = self.lexi[ni].fit_transform(X[[dai_feat_name]])
                extra_name = self._postfix + new_name
                new_feat_name = dai_feat_name + extra_name
                Xnew.columns = [new_feat_name]
                assert not any(pd.isnull(Xnew).values.ravel())
                X = pd.concat([X, Xnew], axis=1)
                self.new_names_dict[new_feat_name] = [dai_feat_name]
                self.lexi_names[ni] = new_feat_name

        if False: # already done by lexi encoding
            # sorted label encoding of ord_5, use for numeric
            orig_feat_name = 'ord_5'
            new_name = 'ord5sorted'
            if orig_feat_name in self.raw_names_dict and self.raw_names_dict[orig_feat_name] in X.columns:
                dai_feat_name = self.raw_names_dict[orig_feat_name]
                extra_name = self._postfix + new_name
                new_feat_name = dai_feat_name + extra_name
                if not transform:
                    self.ord_5_sorted = sorted(list(set(X[dai_feat_name].values)))
                    self.ord_5_sorted = dict(zip(self.ord_5_sorted, range(len(self.ord_5_sorted))))
                X.loc[:, new_feat_name] = X[dai_feat_name].apply(
                    lambda x: self.ord_5_sorted[x] if x in self.ord_5_sorted else -1).astype(np.float32)
                self.new_names_dict[new_feat_name] = [dai_feat_name]
                self.ord5sorted = new_feat_name

        # frequency encode everything
        # keep as cat for OHE
        if not transform:
            self.freq = [None] * len(self.orig_cols)
            self.freq_names = [None] * len(self.orig_cols)
        for ni, c in enumerate(list(self.orig_cols)):
            new_name = "freq%d" % ni
            dai_feat_name = c
            if transform:
                Xnew = self.freq[ni].transform(X[[dai_feat_name]].astype(str)).to_pandas()
            else:
                self.freq[ni] = FrequentTransformer([dai_feat_name])
                Xnew = self.freq[ni].fit_transform(X[[dai_feat_name]].astype(str)).to_pandas()
            extra_name = self._postfix + new_name
            new_feat_name = dai_feat_name + extra_name
            Xnew.columns = [new_feat_name]
            assert not any(pd.isnull(Xnew).values.ravel())
            X = pd.concat([X, Xnew], axis=1)
            self.new_names_dict[new_feat_name] = [dai_feat_name]
            self.freq_names[ni] = new_feat_name

        if self.dai_te:
            # target encode everything
            # use as numeric and categorical
            if not transform:
                self.te = [None] * len(self.orig_cols)
                self.te_names = [None] * len(self.orig_cols)
            for ni, c in enumerate(list(self.orig_cols)):
                new_name = "te%d" % ni
                dai_feat_name = c
                if transform:
                    Xnew = self.te[ni].transform(X[[dai_feat_name]].astype(str), y).to_pandas()
                else:
                    self.te[ni] = CVTargetEncodeTransformer([dai_feat_name])
                    Xnew = self.te[ni].fit_transform(X[[dai_feat_name]].astype(str), y).to_pandas()
                extra_name = self._postfix + new_name
                new_feat_name = dai_feat_name + extra_name
                Xnew.columns = [new_feat_name]
                assert not any(pd.isnull(Xnew).values.ravel())
                X = pd.concat([X, Xnew], axis=1)
                self.new_names_dict[new_feat_name] = [dai_feat_name]
                self.te_names[ni] = new_feat_name

        if self.other_te:
            # target encode lexilabel encoded features
            # use as numeric and categorical
            if not transform:
                self.teo = [None] * len(self.lexi_names)
                self.teo_names = [None] * len(self.lexi_names)
            for ni, c in enumerate(self.lexi_names):
                if c is None:
                    continue
                new_name = "teo%d" % ni
                dai_feat_name = c
                X_local = X.loc[:, [dai_feat_name]].astype(str)
                if transform:
                    Xnew = pd.DataFrame(self.teo[ni].transform_test(X_local))
                else:
                    from target_encoding import TargetEncoder
                    ALPHA, MAX_UNIQUE, FEATURES_COUNT = get_TE_params(X_local, debug=False)
                    self.teo[ni] = TargetEncoder(alpha=ALPHA, max_unique=MAX_UNIQUE, split_in=[3])
                    Xnew = pd.DataFrame(self.teo[ni].transform_train(X=X_local, y=y))
                extra_name = self._postfix + new_name
                new_feat_name = dai_feat_name + extra_name
                Xnew.columns = [new_feat_name]
                assert not any(pd.isnull(Xnew).values.ravel())
                X = pd.concat([X, Xnew], axis=1)
                self.new_names_dict[new_feat_name] = self.new_names_dict[dai_feat_name]  # 2nd layer derived
                self.teo_names[ni] = new_feat_name

        # Encode months by count of holidays, etc.

        # Bin months to seasons
        def month2spring(x):
            return 1 if x >= 3 and x <= 5 else 0

        X, self.spring = self.make_feat(X, 'month', 'spring', month2spring)

        def month2summer(x):
            return 1 if x >= 6 and x <= 8 else 0

        X, self.summer = self.make_feat(X, 'month', 'summer', month2summer)

        def month2fall(x):
            return 1 if x >= 9 and x <= 11 else 0

        X, self.fall = self.make_feat(X, 'month', 'fall', month2fall)

        def month2winter(x):
            return 1 if x >= 12 or x <= 2 else 0

        X, self.winter = self.make_feat(X, 'month', 'winter', month2winter)

        # Cycle months
        def month2cycle1(x):
            return np.sin(2.0 * np.pi * x / 12.0)

        X, self.monthcycle1 = self.make_feat(X, 'month', 'month_cycle1', month2cycle1)

        def month2cycle2(x):
            return np.cos(2.0 * np.pi * x / 12.0)

        X, self.monthcycle2 = self.make_feat(X, 'month', 'month_cycle2', month2cycle2)

        # Bin day to weekend
        def day2weekend(x):
            return 1 if x == 1 or x == 2 else 0

        X, self.weekend = self.make_feat(X, 'day', 'weekend', day2weekend)

        # Cycle days
        def day2cycle1(x):
            return np.sin(2.0 * np.pi * x / 7.0)

        X, self.daycycle1 = self.make_feat(X, 'day', 'day_cycle1', day2cycle1)

        def day2cycle2(x):
            return np.cos(2.0 * np.pi * x / 7.0)

        X, self.daycycle2 = self.make_feat(X, 'day', 'day_cycle2', day2cycle2)

        if self.cache and (not os.path.isfile(file) or not os.path.isfile(file2)):
            Xy = X.copy()
            if not transform:
                Xy.loc[:, 'target'] = y
            #Xy.to_csv(file, index=False)
            save_obj(Xy, file)
            if not transform:
                save_obj(copy.deepcopy(self), file2)
        return X

    def transform(self, X: pd.DataFrame):
        return self.fit_transform(X, transform=True)

    def make_feat(self, X, orig_feats, new_name, func, is_float=True, **kwargs):
        simple = False
        if not isinstance(orig_feats, list):
            orig_feats = [orig_feats]
            simple = True
        new_feat_name = None
        in_raw = all([orig_feat in self.raw_names_dict for orig_feat in orig_feats])
        if in_raw:
            in_X = all([self.raw_names_dict[orig_feat] in X for orig_feat in orig_feats])
        else:
            in_X = False
        extra_name = self._postfix + new_name
        if in_raw and in_X:
            dai_feat_names = [self.raw_names_dict[orig_feat] for orig_feat in orig_feats]
            new_feat_name = [dai_feat_name + extra_name for dai_feat_name in dai_feat_names][0]
            if simple:
                X[new_feat_name] = X[dai_feat_names[0]].apply(lambda x: func(x, **kwargs))
            else:
                X[new_feat_name] = X[dai_feat_names].apply(lambda x: func(x, **kwargs), axis=1)
            if is_float:
                X[new_feat_name] = X[new_feat_name].astype(np.float32)
            self.new_names_dict[new_feat_name] = dai_feat_names
        return X, new_feat_name

    def aggregate(self, full_features_list, importances):
        # for purposes of aggregation back to original space, re-assign generated features
        for vi, v in enumerate(full_features_list):
            if v in self.new_names_dict:
                full_features_list[vi] = self.new_names_dict[v][0]  # FIXME: just take first element for now
                # print("Derived importance: %s %s %g" % (v, self.new_names_dict[v], importances[vi]))
                indices = [i for i, name in enumerate(full_features_list) if self.new_names_dict[v][0] == name]
                for index in indices:
                    if vi != index:
                        pass
                        # print("matching other imp: %s %g" % (v, importances[index]))
        return full_features_list

    def update_numerical_features(self, features):
        force = [
            # self.raw_names_dict['month'],
            # self.raw_names_dict['day'],
            self.monthcycle1,
            self.monthcycle2,
            self.daycycle1,
            self.daycycle2,
            self.temp1,
            self.temp2,
            self.kaggle1,
            self.kaggle2,
            self.sides,
            self.animal,
            self.ord5sorted,
        ]
        force.extend(self.lexi_names)
        force.extend(self.lenfeats)
        force.extend(self.hexints)
        if self.other_te:
            force.extend(self.teo_names)
        if self.dai_te:
            force.extend(self.te_names)
        self.update_cat_and_num(force)
        force = [x for x in force if x is not None]
        more_nums = (pd.Series([True if x in force else False for x in list(features.index)], index=features.index))
        features = (features) | (more_nums)
        return features

    def update_categorical_features(self, features):
        # remove numerical as cat
        avoid_as_cat = [
            self.monthcycle1,
            self.monthcycle2,
            self.daycycle1,
            self.daycycle2,
            self.temp1,
            self.temp2,
            self.kaggle1,
            self.kaggle2,
            self.sides,
            self.animal,
            self.ord5sorted,
        ]
        avoid_as_cat.extend(self.lexi_names)
        avoid_as_cats = (pd.Series([False if x in avoid_as_cat else True for x in list(features.index)],
                                   index=features.index))
        features = (features) & (avoid_as_cats)

        # include
        force = []
        self.update_cat_and_num(force)
        force = [x for x in force if x is not None]
        more_cats = (pd.Series([True if x in force else False for x in list(features.index)],
                               index=features.index))
        features = (features) | (more_cats)

        return features

    def update_cat_and_num(self, force):
        #if self.other_te:
        #    force.extend(self.teo_names)  # numeric and cat
        #if self.dai_te:
        #    force.extend(self.te_names)  # numeric and cat

        #if 'bin_0' in self.raw_names_dict:
        #    force.append(self.raw_names_dict['bin_0'])
        #if 'bin_1' in self.raw_names_dict:
        #    force.append(self.raw_names_dict['bin_1'])
        #if 'bin_2' in self.raw_names_dict:
        #    force.append(self.raw_names_dict['bin_2'])
        return force


def get_TE_params(cat_X, debug=False):
    len_uniques = []
    cat_X_copy = cat_X.copy()
    for c in cat_X.columns:
        le = LabelEncoder()
        le.fit(cat_X[c])
        cat_X_copy[c] = le.transform(cat_X_copy[c])
        len_uniques.append(len(le.classes_))
    if debug:
        uniques_series = pd.Series(len_uniques, index=list(cat_X.columns))
        print("uniques_series: %s" % uniques_series)
    ALPHA = 75
    MAX_UNIQUE = max(len_uniques)
    FEATURES_COUNT = cat_X.shape[1]
    return ALPHA, MAX_UNIQUE, FEATURES_COUNT

