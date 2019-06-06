import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import CustomModel
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, physical_cores_count

# Kaggle Problem: Amazon.com - Employee Access Challenge
# https://www.kaggle.com/c/amazon-employee-access-challenge

# Data: https://www.kaggle.com/c/amazon-employee-access-challenge/data

# Run DAI with this config.toml file and 5/10/7 settings, AUC scorer

# START config.toml
exclude_transformers = "['NumToCatWoETransformer','NumToCatWoEMonotonicTransformer','NumToCatTETransformer','NumCatTETransformer','OriginalTransformer','InteractionsTransformer','IsHolidayTransformer','LagsTransformer','LagsInteractionTransformer','LagsAggregatesTransformer','TextLinModelTransformer','TextTransformer','TruncSVDNumTransformer','EwmaLagsTransformer','DatesTransformer','DateTimeOriginalTransformer','DateOriginalTransformer','ClusterTETransformer','ClusterIdTransformer','ClusterDistTransformer','CVCatNumEncode']"
enable_xgboost_gbm = "off"
enable_xgboost_dart = "off"
enable_glm = "off"
enable_lightgbm = "off"
enable_rf = "off"
enable_tensorflow = "off"
enable_rulefit = "off"
enable_ftrl = "on"
# END config.toml

# All columns are of integer types, but are actually categorical,
# so we convert all numeric columns into string columns here, to not treat them as numerical.
class MyToStringTransformer(CustomTransformer):
    _numeric_output = False
    _included_boosters = ['catboost', 'ftrl']  # only want FTRL and CatBoost model below to consume these features

    @property
    def display_name(self):
        return "Str"

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X[:, dt.stype.str32(dt.f[0])]


# And we add the CatBoost Classifier since it can handle string input directly
# https://github.com/KwokHing/YandexCatBoost-Python-Demo
class CatBoostModel(CustomModel):
    _can_handle_non_numeric = True
    _binary = True
    _display_name = "CatBoost"
    _description = "Yandex CatBoost GBM"
    _boosters = ['catboost']
    _modules_needed_by_name = ['catboost']

    @property
    def has_pred_contribs(self):
        return True

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        n_jobs = max(1, physical_cores_count)
        max_iterations = min(kwargs['n_estimators'],
                             config.max_nestimators) if 'n_estimators' in kwargs else config.max_nestimators
        max_iterations = min(kwargs['iterations'], max_iterations) if 'iterations' in kwargs else max_iterations
        self.params = {'iterations': max_iterations,
                       'learning_rate': config.min_learning_rate,
                       'train_dir': config.data_directory,
                       'allow_writing_files': False,
                       'thread_count': self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                       'early_stopping_rounds': 20
                       }

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        lb = LabelEncoder()
        lb.fit(self.labels)
        y = lb.transform(y)

        if isinstance(X, dt.Frame):
            orig_cols = list(X.names)
            X = X.to_numpy()
            if eval_set is not None:
                valid_X = eval_set[0][0].to_numpy()  # don't assign back to X so don't damage during predict
                valid_y = eval_set[0][1]
                valid_y = lb.transform(valid_y)
                eval_set[0] = (valid_X, valid_y)

        model = CatBoostClassifier(**self.params)
        model.fit(X, y=y,
                  sample_weight=sample_weight,
                  baseline=None,
                  eval_set=eval_set,
                  early_stopping_rounds=kwargs.get('early_stopping_rounds', None),
                  verbose=self.params.get('verbose', False)
                  )

        # need to move to wrapper
        if model.get_best_iteration() is not None:
            iterations = model.get_best_iteration() + 1
        else:
            iterations = self.params['iterations'] + 1
        # must always set best_iterations
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=model.feature_importances_,
                                  iterations=iterations)

    def predict(self, X, **kwargs):
        model, features, importances, iterations = self.get_model_properties()
        X = X.to_numpy()
        pred_contribs = kwargs.get('pred_contribs', None)
        kwargs['ntree_limit'] = iterations - 1

        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        if not pred_contribs:
            preds = model.predict_proba(X,
                                        ntree_start=iterations - 1,
                                        thread_count=self.params['thread_count'])
            if preds.shape[1] == 2:
                return preds[:, 1]
            else:
                return preds
        else:
            return model.get_feature_importance(data=X,
                                                ntree_start=iterations - 1,
                                                thread_count=self.params['thread_count'],
                                                type=EFstrType.ShapValues)
