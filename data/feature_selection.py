"""Perform feature selection by using target perturbation technique"""
# Article: https://academic.oup.com/bioinformatics/article/26/10/1340/193348

from typing import Union, List
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

##### Global variables #####
cols2ignore = [
    "ID"
]  # this columns will be ignored by feature selection and remain in the dataset
target = "target"  # target column to use
is_regression = False  # problem type, regression or classification

number_of_iterations = 100  # how many perturbations to perform
threshold = 75  # percentile of NULL importance score distribution to use (higher number - more aggressive feature pruning)
importance = "gain"  # which importance measurement to use? 'gain' or 'split'

lgbm_params = {
    "n_estimators": 250,
    "num_leaves": 256,
    "boosting_type": "rf",
    "colsample_bytree": 0.75,
    "subsample": 0.632,  # Standard RF bagging fraction
    "subsample_freq": 1,
    "n_jobs": -1,
}


##### Global variables #####


def get_feature_importances(data, shuffle, cats=[], seed=None):
    # Gather real features
    train_features = [f for f in data if f not in [target] + cols2ignore]

    # Shuffle target if required
    y = data[target].copy()
    if shuffle:
        y = data[target].copy().sample(frac=1.0, random_state=seed + 4)
    from h2oaicore.lightgbm_dynamic import import_lightgbm

    lgbm = import_lightgbm()
    import lightgbm as lgbm

    if is_regression:
        model = lgbm.LGBMRegressor(
            random_state=seed, importance_type=importance, **lgbm_params
        )
    else:
        model = lgbm.LGBMClassifier(
            random_state=seed, importance_type=importance, **lgbm_params
        )
        y = LabelEncoder().fit_transform(y)
    # Fit LightGBM in RF mode, yes it's quicker than sklearn RandomForest
    model.fit(data[train_features], y, categorical_feature=cats)
    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(train_features)
    imp_df["importance"] = model.feature_importances_

    return imp_df


class FeatureSelection(CustomData):
    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def create_data(X: dt.Frame = None):
        if X is None:
            return []

        data = X.to_pandas().copy()

        # identify categorical colmns and trasform them
        cats = [
            x
            for x in data.select_dtypes(exclude=np.number).columns
            if x not in [target] + cols2ignore
        ]

        for c in cats:
            data[c] = OrdinalEncoder().fit_transform(
                data[c].astype(str).values.reshape(-1, 1)
            )

        # Get the actual importance, i.e. without shuffling
        actual_imp_df = get_feature_importances(
            data=data, cats=cats, shuffle=False, seed=42
        )

        # Seed the unexpected randomness of this world
        np.random.seed(123)

        seeds = np.random.randint(0, 2**30, size=number_of_iterations)
        null_imp_df = pd.DataFrame()

        for i, s in enumerate(seeds):
            # Get current run importances
            imp_df = get_feature_importances(data=data, cats=cats, shuffle=True, seed=s)
            imp_df["run"] = i + 1
            # Concat the latest importances with the old ones
            null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

        feature_scores = []
        for _f in actual_imp_df["feature"].unique():
            f_null_imps_gain = null_imp_df.loc[
                null_imp_df["feature"] == _f, "importance"
            ].values
            f_act_imps_gain = actual_imp_df.loc[
                actual_imp_df["feature"] == _f, "importance"
            ].mean()
            _score = np.log(
                1e-10
                + f_act_imps_gain
                / (1 + np.percentile(f_null_imps_gain, max(75, min(99, threshold))))
            )

            feature_scores.append((_f, _score))

        scores_df = pd.DataFrame(feature_scores, columns=["feature", "score"])
        # final feature selection
        selected_features = scores_df[scores_df["score"] > 0]["feature"].values.tolist()
        selected_features = np.unique(selected_features).tolist()

        data = X.to_pandas().copy()
        return data[cols2ignore + selected_features + [target]]
