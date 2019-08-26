"""Sklearn Logistic Regression."""
import datatable as dt
import numpy as np
import random
import pandas as pd
import copy
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
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
    # _impute_type = 'oob'
    _impute_type = 'sklearn'
    _mutate_by_one = False

    _regression = False
    _binary = True
    _multiclass = True
    _mutate_all = True
    _parallel_task = True if _grid_search or _grid_search_iterations else False
    _fit_by_iteration = True
    _fit_iteration_name = 'max_iter'
    _can_handle_non_numeric = True
    _display_name = "LR"
    _description = "Logistic Regression"

    _always_defaults = False
    _extra_effort = False

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

        self.params['random_state'] = kwargs.get("random_state", 1234)
        self.params['n_jobs'] = self.params_base.get('n_jobs', max(1, physical_cores_count))

        # Modify certain parameters for tuning
        C_list = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.0]
        self.params["C"] = float(np.random.choice(C_list)) if not get_default else 0.1

        tol_list = [1e-4, 1e-3, 1e-5]
        self.params["tol"] = float(np.random.choice(tol_list)) if not get_default else 1e-4
        if self._extra_effort:
            self.params["tol"] = 1e-6

        # solver_list = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        # newton-cg too slow
        # sag too slow
        # solver_list = ['lbfgs', 'liblinear', 'saga']
        solver_list = ['lbfgs']
        self.params["solver"] = str(np.random.choice(solver_list)) if not get_default else 'lbfgs'

        max_iter_list = [100, 200, 1000]
        self.params["max_iter"] = int(np.random.choice(max_iter_list)) if not get_default else 100
        if self._extra_effort:
            self.params["max_iter"] = 1000

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
            l1_ratio_list = [0, 0.5, 1.0]
            self.params["l1_ratio"] = float(np.random.choice(l1_ratio_list))
        else:
            self.params.pop('l1_ratio', None)
        if self.params["penalty"] == 'none':
            self.params.pop('C', None)
            self.params.pop('l1_ratio', None)
        if self.num_classes > 2:
            self.params['multi_class'] = 'auto'
        strategy_list = ['mean', 'median', 'most_frequent', 'constant']
        self.params['strategy'] = str(np.random.choice(strategy_list)) if not get_default else 'mean'

        min_samples_leaf_list = [1, 10, 50, 100]
        self.params['min_samples_leaf'] = float(np.random.choice(min_samples_leaf_list))
        smoothing_list = [1.0, 0.5, 10.0, 50.0]
        self.params['smoothing'] = float(np.random.choice(smoothing_list))

        sigma_list = [None, 1.0, 0.5, 10.0, 50.0]
        self.params['sigma'] = random.choice(sigma_list)

        randomized_list = [True, False]
        self.params['randomized'] = random.choice(randomized_list)
        sigma_woe_list = [0.05, 0.025, 0.1, 0.2]
        self.params['sigma_woe'] = random.choice(sigma_woe_list)
        regularization_list = [1.0, 0.5, 2.0, 10.0]
        self.params['regularization'] = random.choice(regularization_list)

        if self._mutate_by_one:
            pick_key = str(np.random.choice(list(self.params.keys()), size=1)[0])
            value = self.params[pick_key]
            self.params = copy.deepcopy(params_orig)
            self.params[pick_key] = value
            # WIP, need to validate parameters in clean way

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

        impute_params = {}
        impute_params['strategy'] = lr_params.pop('strategy', None)

        ord_params = dict(handle_missing='value', handle_unknown='value')

        te_params = dict(handle_missing='value', handle_unknown='value')
        te_params['min_samples_leaf'] = lr_params.pop('min_samples_leaf')
        te_params['smoothing'] = lr_params.pop('smoothing')

        cb_params = dict(handle_missing='value', handle_unknown='value')
        cb_params['sigma'] = lr_params.pop('sigma')

        woe_params = dict(handle_missing='value', handle_unknown='value')
        woe_params['randomized'] = lr_params.pop('randomized')
        woe_params['sigma'] = lr_params.pop('sigma_woe')
        woe_params['regularization'] = lr_params.pop('regularization')

        # cat_features = [x for x in X_names if CatOriginalTransformer.is_me_transformed(x)]
        # noncat_features = [x for x in X_names if x not in cat_features]
        numerical_features = X.dtypes == 'float'
        categorical_features = ~numerical_features
        cat_X = X.loc[:, categorical_features]
        num_X = X.loc[:, numerical_features]

        full_features_list = []
        transformers = []
        if self._use_numerics and any(numerical_features.values):
            full_features_list.extend(list(num_X.columns))
            transformers.append(
                (make_pipeline(SimpleImputer(**impute_params), StandardScaler()), numerical_features)
            )
        # http://contrib.scikit-learn.org/categorical-encoding/
        if self._use_ordinal_encoding and any(categorical_features.values):
            full_features_list.extend(list(cat_X.columns))
            # Note: OrdinalEncoder doesn't handle unseen features, while CategoricalEncoder used too
            import category_encoders as ce
            transformers.append(
                (ce.OrdinalEncoder(**ord_params), categorical_features)
            )
        if self._use_catboost_encoding and any(categorical_features.values):
            full_features_list.extend(list(cat_X.columns))
            import category_encoders as ce
            transformers.append(
                (ce.CatBoostEncoder(**cb_params), categorical_features)
            )
        if self._use_woe_encoding and any(categorical_features.values):
            full_features_list.extend(list(cat_X.columns))
            import category_encoders as ce
            transformers.append(
                (ce.WOEEncoder(**woe_params), categorical_features)
            )
        if self._use_target_encoding and any(categorical_features.values):
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
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
        model = make_pipeline(
            preprocess,
            LogisticRegression(**lr_params))

        if self._grid_search_iterations and self.num_classes == 2 and 'AUC' in self.params_base['score_f_name'].upper():
            # WIP FIXME for multiclass and other scorers
            from sklearn.model_selection import GridSearchCV

            if self._extra_effort or False:
                max_iter_range = range(0, 2000, 10)
            else:
                max_iter_range = np.arange(int(self.params['max_iter'] / 2), int(self.params['max_iter'] * 2),
                                           int(self.params['max_iter'] / 2))
            print("max_iter_range: %s" % str(max_iter_range))
            param_grid = {
                'logisticregression__max_iter': max_iter_range,
            }
            scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)
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
                'logisticregression__C': [0.1, 0.5, 1.0],
            }
            grid_clf = GridSearchCV(model, param_grid, cv=10, iid=False)
            grid_clf.fit(X, y)
            model = grid_clf.best_estimator_
            # self.best_params = grid_clf.best_params_
        else:
            model.fit(X, y)

        lr_model = model.named_steps['logisticregression']

        # average importances over classes
        importances = np.average(np.array(lr_model.coef_), axis=0)
        # average iterations over classes (can't take max_iter per class)
        iterations = np.average(lr_model.n_iter_, axis=0)

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

        # aggregate
        msg = "num=%d cat=%d : ohe=%d : imp=%d full=%d" % (
            len(num_X.columns), len(cat_X.columns), len(ohe_features_short), len(importances), len(full_features_list))
        print(msg)
        assert len(importances) == len(full_features_list), msg
        importances = pd.Series(np.abs(importances), index=full_features_list).groupby(level=0).mean()
        assert len(importances) == len(X_names)
        # Below for dummy testing
        # importances = np.array([1.0] * len(X_names))
        print("LRiterations: %d" % iterations)

        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=importances.tolist(),
                                  iterations=iterations)

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
