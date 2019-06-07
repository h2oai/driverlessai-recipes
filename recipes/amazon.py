import datatable as dt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import CustomModel, BaseCustomModel, LightGBMModel
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import config, physical_cores_count


# Kaggle Problem: Amazon.com - Employee Access Challenge
# https://www.kaggle.com/c/amazon-employee-access-challenge

# Data: https://www.kaggle.com/c/amazon-employee-access-challenge/data

# # Run DAI with 5/2/1 settings, AUC scorer
# Recommended: Exclude all models except for LIGHTGBMDEEP, CATBOOST in expert settings -> custom recipes -> exclude specific models.


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


class TargetEncodingExpandingMean(CustomTransformer):
    @staticmethod
    def do_acceptance_test():
        return False  # transformer can fail for small data

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        X = X.to_pandas()
        self.learned_values = {}
        self.dataset_mean = np.mean(y)
        X["__target__"] = y
        for c in X.columns:
            stats = (X[[c, "__target__"]]
                     .groupby(c)["__target__"]
                     .agg(['mean', 'size']))  #
            stats["__target__"] = stats["mean"]
            stats = (stats
                     .drop([x for x in stats.columns if x not in ["__target__", c]], axis=1)
                     .reset_index())
            self.learned_values[c] = stats

        # Expanding mean transform
        X = X[self.columns_names].copy().reset_index(drop=True)
        X["__target__"] = y
        X["index"] = X.index
        X_transformed = pd.DataFrame()
        for c in self.columns_names:
            X_shuffled = X[[c, "__target__", "index"]].copy()
            X_shuffled = X_shuffled.sample(n=len(X_shuffled), replace=False)
            X_shuffled["cnt"] = 1
            X_shuffled["cumsum"] = (X_shuffled
                                    .groupby(c, sort=False)['__target__']
                                    .apply(lambda x: x.shift().cumsum()))
            X_shuffled["cumcnt"] = (X_shuffled
                                    .groupby(c, sort=False)['cnt']
                                    .apply(lambda x: x.shift().cumsum()))
            X_shuffled["encoded"] = X_shuffled["cumsum"] / X_shuffled["cumcnt"]
            X_shuffled["encoded"] = X_shuffled["encoded"].fillna(self.dataset_mean)
            X_transformed[c] = X_shuffled.sort_values("index")["encoded"].values
        return X_transformed

    def transform(self, X: dt.Frame):
        X = X.to_pandas()
        transformed_X = X[self.columns_names].copy()
        for c in transformed_X.columns:
            transformed_X[c] = (transformed_X[[c]]
                                .merge(self.learned_values[c], on=c, how='left')
                                )["__target__"]
        transformed_X = transformed_X.fillna(self.dataset_mean)
        return transformed_X


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


# class MyLabelEncoderTransformer(CustomTransformer):
#     _included_boosters = ['lightgbmdeep']
#
#     @property
#     def display_name(self):
#         return "LabelEnc"
#
#     @staticmethod
#     def get_default_properties():
#         return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)
#
#     def fit_transform(self, X: dt.Frame, y: np.array = None):
#         self.lb = LabelEncoder()
#         self.lb.fit(X[:, self.input_feature_names[0]].to_numpy().ravel())
#         return self.transform(X)
#
#     def transform(self, X: dt.Frame):
#         return self.lb.transform(X[:, self.input_feature_names[0]].to_numpy().ravel())

