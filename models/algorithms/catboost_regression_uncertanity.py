import inspect
import os

import copy
import datatable as dt
import numpy as np
import pandas as pd
from datetime import datetime
from h2oaicore.models import CustomModel, MainModel
from h2oaicore.systemutils import config, physical_cores_count, user_dir
from h2oaicore.systemutils import (
    loggerinfo,
)

"""
This custom recipe of CatBoost is intended to support Uncertainty estimation
    - Supports only regression can be extended to classification
    - GPU support is OFF for sake of simplicity
    - Predictions returned by Driverless AI only include mean predictions of the target variable. The Data Uncertainty 
        and Knowledge Uncertainty are written to experiment artifacts
    - Make sure the ensemble level for the final model is set to zero in Driverless AI
    - Currently, we only support a single final model(fixed_ensemble_lesvel=0) for CatBoostUncertanity to make any sense of the model predictions
        written to summary artifact
    - The model_predictions.json file is updated in experiment artifacts each time we call predict function using DAI/Py Client
"""

""" 
TODO
    - Is there a better way to specify logger
    - Can use Optuna during the model fit function
    - If possible, update the predict function to return all three outputs, i.e. [Mean Prediction, Knowledge Uncertainty, Data Uncertainty]. 
        Currently limited to getting only one value as output from DAI and MLOps
"""


class CatBoostRegressionUncertanityModel(CustomModel):
    _can_handle_non_numeric = True
    _can_handle_text = False
    _regression = True
    _can_handle_categorical = True

    _can_use_gpu = False

    _min_one_hot_max_size = 4
    _min_learning_rate_catboost = 0.005

    _display_name = "CatBoostRegressionUncertanity"
    _description = "Yandex CatBoost GBM"
    _modules_needed_by_name = ["catboost==1.0.5"]

    @staticmethod
    def do_acceptance_test():
        return True

    def set_default_params(
            self, accuracy=10, time_tolerance=10, interpretability=1, **kwargs
    ):
        kwargs.pop("get_best", None)
        self.mutate_params(
            accuracy=accuracy,
            time_tolerance=time_tolerance,
            interpretability=interpretability,
            get_best=True,
            **kwargs
        )

    def estimators_list(self, accuracy=None):
        # Range of values can be changed as required!
        if accuracy is None:
            accuracy = 10
        if accuracy >= 9:
            estimators_list = [1000, 500, 100, 200, 300, 2000]
        elif accuracy >= 8:
            estimators_list = [300, 500, 1000, 100, 200]
        elif accuracy >= 5:
            estimators_list = [500, 700, 900]
        else:
            estimators_list = [500, 600, 700]
        return estimators_list

    def mutate_params(
            self,
            accuracy=10,
            time_tolerance=10,
            interpretability=1,
            get_best=False,
            **kwargs
    ):
        """Mutate `self.params` dictionary of model parameters to be used during `fit()` and `predict()`.
        Called to modify model parameters `self.params` in a self-consistent way, fully controlled by the user.
        If no parameter tuning desired, leave at default.
        Args:
            accuracy (int): Accuracy setting for this experiment (1 to 10)
                10 is most accurate, expensive
            time_tolerance (int): Time setting for this experiment (0 to 10)
                10 is most patient, 1 is fast
            interpretability (int): Interpretability setting for this experiment (1 to 10)
                1 is most complex, 10 is most interpretable
            score_f_name (str): scorer used by DAI, which mutate can use to infer best way to change parameters
            trial: Optuna trial object, used to tell Optuna what chosen for mutation
            **kwargs (dict): Optional dictionary containing system-level information for advanced usage
        Returns: None
        """

        # We can import user choice to have some level of control like setting a parameter value with out actually changing the recipe itself
        # Overrides the parameter value
        user_choice = config.recipe_dict.copy()
        self.params = dict()
        uses_gpus, n_gpus = self.get_uses_gpus(self.params)

        get_best = kwargs.get("get_best", True)
        if get_best is None:
            get_best = True

        # Setting trail=True will enable optuna
        trial = kwargs.get("trial", False)
        if trial is None:
            trial = False

        # MainModel.get_one is a helper function to set default params for model and also help us pick some random values for parameter in model tuning stage
        # self.params["n_estimators"] = self.params_base.get("n_estimators", 600)
        self.params["n_estimators"] = MainModel.get_one(
            self.estimators_list(accuracy=accuracy),
            get_best=get_best,
            best_type="first",
            name="n_estimators",
            trial=trial,
            user_choice=user_choice,
        )

        self.params["learning_rate"] = self.params_base.get(
            "learning_rate", config.min_learning_rate
        )

        early_stopping_rounds_default = min(
            500, max(1, int(self.params["n_estimators"] / 4))
        )
        self.params["early_stopping_rounds"] = self.params_base.get(
            "early_stopping_rounds", early_stopping_rounds_default
        )

        # No GPU use
        # Set colsample_bylevel parameter
        if not uses_gpus:
            colsample_bylevel_list = [0.3, 0.5, 0.9, 1.0]
            self.params["colsample_bylevel"] = MainModel.get_one(
                colsample_bylevel_list,
                get_best=get_best,
                best_type="first",
                name="colsample_bylevel",
                trial=trial,
            )

        # Set one_hot_max_size parameter
        if self._can_handle_categorical:
            max_cat_to_onehot_list = [4, 10, 20, 40, config.max_int_as_cat_uniques]
            if uses_gpus:
                max_one_hot_max_size = 255
            else:
                max_one_hot_max_size = 65535
            max_cat_to_onehot_list = sorted(
                set([min(x, max_one_hot_max_size) for x in max_cat_to_onehot_list])
            )
            log = True if max(max_cat_to_onehot_list) > 1000 else False
            self.params["one_hot_max_size"] = MainModel.get_one(
                max_cat_to_onehot_list,
                get_best=get_best,
                best_type="max",
                name="one_hot_max_size",
                trial=trial,
                log=log,
            )

        if not uses_gpus:
            sampling_frequency_list = [
                "PerTree",
                "PerTreeLevel",
                "PerTreeLevel",
                "PerTreeLevel",
            ]
            self.params["sampling_frequency"] = MainModel.get_one(
                sampling_frequency_list,
                get_best=get_best,
                best_type="first",
                name="sampling_frequency",
                trial=trial,
            )

        bootstrap_type_list = [
            "Bayesian",
            "Bayesian",
            "Bayesian",
            "Bayesian",
            "Bernoulli",
            "MVS",
            "No",
        ]
        if uses_gpus:
            bootstrap_type_list.remove("MVS")  # MVS is only supported in CPU mode only
        self.params["bootstrap_type"] = MainModel.get_one(
            bootstrap_type_list,
            get_best=get_best,
            best_type="first",
            name="bootstrap_type",
            trial=trial,
        )

        if self.params["bootstrap_type"] in ["Poisson", "Bernoulli"]:
            subsample_list = [0.5, 0.66, 0.66, 0.9]
            # will get pop'ed if not Poisson/Bernoulli
            self.params["subsample"] = MainModel.get_one(
                subsample_list,
                get_best=get_best,
                best_type="first",
                name="subsample",
                trial=trial,
            )

        if self.params["bootstrap_type"] == "Bayesian":
            bagging_temperature_list = [0.0, 0.1, 0.5, 0.9, 1.0]
            self.params["bagging_temperature"] = MainModel.get_one(
                bagging_temperature_list,
                get_best=get_best,
                best_type="first",
                name="bagging_temperature",
                trial=trial,
            )

        self.params["random_state"] = MainModel.get_one(
            [self.params_base.get("random_state", 1234)],
            get_best=get_best,
            best_type="first",
            name="random_state",
            trial=None,  # not for Optuna tuning
            user_choice=user_choice,
        )

    def fit(
            self,
            X,
            y,
            sample_weight=None,
            eval_set=None,
            sample_weight_eval_set=None,
            **kwargs
    ):
        """Fit the model on training data and use optional validation data to tune parameters to avoid overfitting.
        Args:
            X (dt.Frame): training data, concatenated output of all active transformers' `fit_transform()` method
                Shape: (N, p), rows are observations, columns are features (attributes)
            y (np.array): training target values, numeric for regression, numeric or categorical for classification
                Shape: (N, ), 1 target value per observation
            sample_weight (np.array): (optional) training observation weight values, numeric
                Shape: (N, ), 1 observation weight value per observation
            eval_set (list(tuple(dt.Frame, np.array))): (optional) validation data and target values
                list must have length of 1, containing 1 tuple of X and y for validation data
                Shape: dt.Frame: (M, p), np.array: (M, )), same schema/format as training data, just different rows
            sample_weight_eval_set (list(np.array)): (optional) validation observation weight values, numeric
                list must have length of 1, containing 1 np.array for weights
                Shape: (M, ), 1 observation weight value per observation
            kwargs (dict): Additional internal arguments (see examples)
        Returns: None
        """

        if isinstance(X, dt.Frame):
            orig_cols = list(X.names)
            numeric_cols = list(X[:, [bool, int, float]].names)
        else:
            orig_cols = list(X.columns)
            numeric_cols = list(X.select_dtypes([np.number]).columns)

        # Catboots can handle raw strings
        self.params["cat_features"] = [
            i
            for i, x in enumerate(orig_cols)
            if "CatOrig:" in x or "Cat:" in x or x not in numeric_cols
        ]

        if isinstance(X, dt.Frame) and len(self.params["cat_features"]) == 0:
            orig_cols = list(X.names)
            # dt -> lightgbm internally using buffer leaks, so convert here
            # assume predict is after pipeline collection or in subprocess so needs no protection
            X = X.to_numpy()  # don't assign back to X so don't damage during predict
            X = np.ascontiguousarray(
                X,
                dtype=np.float32
                if config.data_precision in ["float32", "datatable"]
                else np.float64,
            )
            if eval_set is not None:
                valid_X = eval_set[0][
                    0
                ].to_numpy()  # don't assign back to X so don't damage during predict
                valid_X = np.ascontiguousarray(
                    valid_X,
                    dtype=np.float32
                    if config.data_precision in ["float32", "datatable"]
                    else np.float64,
                )
                valid_y = eval_set[0][1]

                eval_set = [(valid_X, valid_y)]

        X, eval_set = self.process_cats(X, eval_set, orig_cols)

        ####
        # Start transcribe
        ####
        params = copy.deepcopy(self.params)  # keep separate
        params = self.transcribe_params(params=params, **kwargs)

        # Make sure the objective is poped and loss is set to RMSEWithUncertainty
        params.pop("objective", None)
        params["loss_function"] = "RMSEWithUncertainty"

        from catboost import CatBoostRegressor

        model = CatBoostRegressor(**params)
        model.fit(
            X,
            y=y,
            eval_set=eval_set,
            baseline=None,
            sample_weight=sample_weight,
            verbose=True,
        )

        if model.get_best_iteration() is not None:
            iterations = model.get_best_iteration() + 1
        else:
            iterations = self.params["n_estimators"]

        # must always set best_iterations
        self.set_model_properties(
            model=model,
            features=orig_cols,
            importances=model.feature_importances_,
            iterations=iterations,
        )

    def predict(self, X, y=None, **kwargs):
        """Make predictions on a test set.
        Use the fitted state stored in `self` to make per-row predictions. Predictions must be independent of order of
        test set rows, and should not depend on the presence of any other rows.
        Args:
            X (dt.Frame): test data, concatenated output of all active transformers' `transform()` method
                Shape: (K, p)
            kwargs (dict): Additional internal arguments (see examples)
        Returns: dt.Frame, np.ndarray or pd.DataFrame, containing predictions (target values or class probabilities)
            Shape: (K, c) where c = 1 for regression or binary classification, and c>=3 for multi-class problems.
        """

        # training = os.environ.get("training", "Yes")
        model, features, importances, iterations = self.get_model_properties()
        n_jobs = max(1, physical_cores_count)

        if isinstance(X, dt.Frame) and len(self.params["cat_features"]) == 0:
            X = X.to_numpy()
            X = np.ascontiguousarray(
                X,
                dtype=np.float32 if config.data_precision == "float32" else np.float64,
            )

        X, _ = self.process_cats(X, None, self.feature_names_fitted)

        # preds = model.predict(
        #   X, ntree_start=0, ntree_end=iterations, thread_count=n_jobs
        # )

        # Can try other values for virtual_ensembles_count
        preds = model.virtual_ensembles_predict(
            X, prediction_type="TotalUncertainty", virtual_ensembles_count=10
        )

        df_pred = pd.DataFrame(
            {"mean_preds": preds[:, 0], "knowledge": preds[:, 1], "data": preds[:, 2]}
        )

        msg = df_pred.to_markdown(index=False)
        loggerinfo(self.get_logger(**kwargs), msg)

        # Write the predictions to experiment logs
        now = datetime.now()
        file_desc_json = os.path.join(
            self.context.experiment_tmp_dir, "model_predictions.json"
        )
        # Write to predictions to experiment folder
        with open(file_desc_json, "w") as f:
            f.write(df_pred.to_json(orient="split"))

        # Uncomment below section in case if you would like to push the model predictions to notifications
        # Push the predictions to GUI notifications tab in experiment
        # task = kwargs.get("task")
        # if task is not None:
        #   task.sync(
        #      progress=dict(
        #         type="warning",
        #        key=self.context.experiment_id,
        #       title="Model Uncertanity Predictions",
        #      # level=AutoDLNotificationLevel.HIGH,
        #     data=df_pred.head().to_markdown(),
        # )
        # )
        # task.flush()

        return preds[:, 0]  # Returning only mean values predicted by a virtual ensemble

    def process_cats(self, X, eval_set, orig_cols):
        # Ref: https://catboost.ai/en/docs/features/categorical-features#:~:text=CatBoost-,supports,-numerical%2C%20categorical%20and
        # Ensure catboost treats as categorical values by making them str
        if len(self.params["cat_features"]) > 0:
            X = X.to_pandas()
            if eval_set is not None:
                valid_X = eval_set[0][0]
                valid_y = eval_set[0][1]
                valid_X = valid_X.to_pandas()
                eval_set = [(valid_X, valid_y)]

            for coli in self.params["cat_features"]:
                col = orig_cols[coli]
                if "CatOrig:" in col:
                    cattype = str
                    # must be string for catboost
                elif "Cat:" in col:
                    cattype = int
                else:
                    cattype = str
                if cattype is not None:
                    if cattype == int:
                        # otherwise would hit: ValueError: Cannot convert non-finite values (NA or inf) to integer
                        X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                        X[col] = X[col].fillna(value=0)
                    X[col] = X[col].astype(cattype)
                    if eval_set is not None:
                        valid_X = eval_set[0][0]
                        valid_y = eval_set[0][1]
                        if cattype == int:
                            # otherwise would hit: ValueError: Cannot convert non-finite values (NA or inf) to integer
                            valid_X[col] = valid_X[col].replace(
                                [np.inf, -np.inf], np.nan
                            )
                            valid_X[col] = valid_X[col].fillna(value=0)
                        valid_X[col] = valid_X[col].astype(cattype)
                        eval_set = [(valid_X, valid_y)]
        return X, eval_set

    def transcribe_params(self, params=None, **kwargs):
        """Transcribe Params method is used to make sure to parameters have acceptable values
        Args:
           params: dictionary of model parameters to be used during `fit()` and `predict()`.
           kwargs (dict): Additional internal arguments (see examples)
        Returns: Params
        """
        uses_gpus, n_gpus = self.get_uses_gpus(self.params)

        if params is None:
            params = self.params.copy()  # reference

        has_eval_set = self.have_eval_set(
            kwargs
        )  # only needs (and does) operate at fit-time

        # Get the allowed parameters for regression in Catboost
        from catboost import CatBoostRegressor

        fullspec_regression = inspect.getfullargspec(CatBoostRegressor)
        allowed_params = {
            k: v for k, v in zip(fullspec_regression.args, fullspec_regression.defaults)
        }
        # If parameter doesnt exist in allowed param dict remove it
        params_copy = copy.deepcopy(params)
        for k, v in params_copy.items():
            if k not in allowed_params.keys():
                del params[k]

        k = "grow_policy"
        if k in params:
            if params[k] == "depthwise":
                params[k] = "Depthwise"
            elif params[k] == "symmetrictree":
                params[k] = "SymmetricTree"
            else:
                params[k] = "Lossguide"

        params.pop("verbose", None)
        params.pop("verbose_eval", None)
        params.pop("logging_level", None)

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=CPU%20and%20GPU-,grow_policy,-Command%2Dline%3A%20%2D%2Dgrow
        if "grow_policy" in params and params["grow_policy"] in [
            "Depthwise",
            "SymmetricTree",
        ]:
            if "max_depth" in params and params["max_depth"] in [0, -1]:
                params["max_depth"] = max(
                    2, int(np.log(params.get("num_leaves", 2 ** 6)))
                )
        else:
            params.pop("max_depth", None)
            params.pop("depth", None)
        if "grow_policy" in params and params["grow_policy"] == "Lossguide":
            if "max_leaves" in params and params["max_leaves"] in [0, -1]:
                params["max_leaves"] = 2 ** params.get("max_depth", 6)
        else:
            params.pop("max_leaves", None)

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=line%3A%20%2D%2Dmax%2Dleaves-,Alias%3Anum_leaves,-Description
        if "num_leaves" in params and "max_leaves" in params:
            params.pop("num_leaves", None)

        # Apply Limits
        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=CPU%20and%20GPU-,max_leaves,-Command%2Dline%3A%20%2D%2Dmax
        if "max_leaves" in params:
            params["max_leaves"] = min(params["max_leaves"], 64)

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=CPU%C2%A0%E2%80%94%20Any%20integer%20up%20to%C2%A0%2016.
        if "max_depth" in params:
            params["max_depth"] = min(params["max_depth"], 16)

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=l2%2Dleaf%2Dregularizer-,Alias%3A%20reg_lambda,-Description
        if "reg_lambda" in params and params["reg_lambda"] <= 0.0:
            params["reg_lambda"] = 3.0

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=about%20Pairwise%20scoring-,255,-if%20training%20is
        if self._can_handle_categorical:
            if "max_cat_to_onehot" in params:
                params["one_hot_max_size"] = params["max_cat_to_onehot"]
                params.pop("max_cat_to_onehot", None)
            if uses_gpus:
                params["one_hot_max_size"] = min(
                    params.get("one_hot_max_size", 255), 255
                )
            else:
                params["one_hot_max_size"] = min(
                    params.get("one_hot_max_size", 65535), 65535
                )

        # Assume during fit self.params_base could have been updated
        # Adjusting the learning rate to honor the DAI settings and local file settings
        assert "n_estimators" in params
        assert "learning_rate" in params
        params["n_estimators"] = self.params_base.get("n_estimators", 100)
        params["learning_rate"] = self.params_base.get(
            "learning_rate", config.min_learning_rate
        )

        # Clip the learning rate between 0.5 & 0.005
        params["learning_rate"] = min(
            params["learning_rate"], 0.5
        )  # 1.0 leads to illegal access on GPUs
        params["learning_rate"] = max(
            self._min_learning_rate_catboost, self.params["learning_rate"]
        )

        if "early_stopping_rounds" not in params and has_eval_set:
            params["early_stopping_rounds"] = 150  # temp fix
            # assert 'early_stopping_rounds' in params

        # Ref: https://catboost.ai/en/docs/concepts/algorithm-main-stages_bootstrap-options#:~:text=Refer%20to%20the%20paper%20for%20details%3B%20supported%20only%20on%20GPU)
        if not uses_gpus and params["bootstrap_type"] == "Poisson":
            params["bootstrap_type"] = "Bayesian"  # revert to default

        # Ref: https://catboost.ai/en/docs/concepts/algorithm-main-stages_bootstrap-options#:~:text=Supported%20only%20on%C2%A0CPU.
        if uses_gpus and params["bootstrap_type"] == "MVS":
            params["bootstrap_type"] = "Bayesian"  # revert to default

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=following%20bootstrap%20types
        if "bootstrap_type" not in params or params["bootstrap_type"] not in [
            "Poisson",
            "Bernoulli",
        ]:
            params.pop(
                "subsample", None
            )  # only allowed for those 2 bootstrap_type settings

        # Ref: https://catboost.ai/en/docs/references/training-parameters/common#:~:text=bootstrap%20type%20is%20Bayesian
        if params["bootstrap_type"] not in ["Bayesian"]:
            params.pop("bagging_temperature", None)

        #########
        # Default value eval_metric for loss=RMSEWithUncertanity is "RMSEWithUncertainty"
        # Can check for alternatives
        #########
        params["eval_metric"] = "RMSEWithUncertainty"

        # Catboost can sometimes write files to local dir even when allow_writing_files is False;
        # Thats why we also specify user directory
        params.update({"train_dir": user_dir(), "allow_writing_files": False})

        # set system stuff here
        params["silent"] = self.params_base.get("silent", True)
        if config.debug_daimodel_level >= 1:
            params[
                "silent"
            ] = False  # Can enable for tracking improvement in console/dai.log if have access

        params["thread_count"] = self.params_base.get(
            "n_jobs", max(1, physical_cores_count)
        )  # -1 is not supported

        return params

    def get_uses_gpus(self, params):
        # For the first version to be simple, skipping support for GPU
        return False, 0
