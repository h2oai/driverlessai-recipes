"""Recipe for Kaggle Competition: Amazon.com - Employee Access Challenge"""
import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import CustomModel, BaseCustomModel, LightGBMModel
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, physical_cores_count


# https://www.kaggle.com/c/amazon-employee-access-challenge

# Data: https://www.kaggle.com/c/amazon-employee-access-challenge/data

# # Run DAI with 5/2/1 settings, AUC scorer
# Recommended: Include only LIGHTGBMDEEP and CATBOOST (in expert settings -> custom recipes -> include models).


class MyLightGBMDeep(BaseCustomModel, LightGBMModel):
    _boosters = ['lightgbmdeep']
    _binary = True
    _display_name = "MYLGBMDEEP"
    _description = "LightGBM with more depth"
    _included_transformers = ['NumToCatWoETransformer', 'CVTargetEncodeTransformer']

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        # First call the parent set_default_params
        LightGBMModel.set_default_params(
            self,
            accuracy=accuracy,
            time_tolerance=time_tolerance,
            interpretability=interpretability,
            **kwargs
        )
        # Then modify the parameters
        self.params["grow_policy"] = "lossguide"
        self.params["max_leaves"] = 8192
        self.params["max_depth"] = -1


class CatBoostModel(CustomModel):
    _can_handle_non_numeric = True
    _binary = True
    _display_name = "CatBoost"
    _description = "Yandex CatBoost GBM"
    _boosters = ['catboost']
    _modules_needed_by_name = ['catboost']
    _included_transformers = ['MyToStringTransformer', 'CVTargetEncodeTransformer']

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        lb = LabelEncoder()
        lb.fit(self.labels)
        y = lb.transform(y)
        orig_cols = list(X.names)
        XX = X.to_pandas()
        params = {
            'train_dir': config.data_directory,
            'allow_writing_files': False,
            'thread_count': 10,
            # 'loss_function': 'Logloss'
        }
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**params)
        model.fit(XX, y=y, sample_weight=sample_weight, verbose=False,
                  cat_features=list(X[:, [str, int]].names))  # Amazon specific, also no early stopping

        # must always set best_iterations
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=model.feature_importances_,
                                  iterations=0)

    def predict(self, X, **kwargs):
        model, features, importances, iterations = self.get_model_properties()
        X = X.to_pandas()
        kwargs['ntree_limit'] = iterations - 1
        preds = model.predict_proba(X, thread_count=10)
        if preds.shape[1] == 2:
            return preds[:, 1]
        else:
            return preds


# Not necessary, but nice to demonstrate creation of string input for CatBoost
class MyToStringTransformer(CustomTransformer):
    _numeric_output = False
    _included_boosters = ['catboost']

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
