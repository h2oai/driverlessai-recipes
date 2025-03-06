"""AutoTheta for TimeSeries Forecasting 

Uses AutoTheta implemented in https://github.com/Nixtla/statsforecast


Current limitations:
(1) No handling for prediction "gap"
(2) Enable user to pass in seasonality length to run AutoTheta with seasonality.  NOTE: The seasonality will be infered based on the temporal column.
"""

import importlib
import datatable as dt
import numpy as np
from h2oaicore.models import CustomTimeSeriesModel
from h2oaicore.systemutils import (
    make_experiment_logger,
    loggerinfo,
    loggerwarning,
    loggerdebug,
)
from h2oaicore.systemutils import (
    small_job_pool,
    save_obj,
    load_obj,
    user_dir,
    remove,
    config,
    max_threads,
)
from h2oaicore.systemutils_more import arch_type
import os
import pandas as pd
import shutil
import random
import uuid
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


FREQS = {"MS": 12, "M": 12, "D": 7, "Q": 4, "QS": 4, "H": 24}

class suppress_stdout_stderr(object):
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


# Parallel implementation requires methods being called from different processes
# Global methods support this feature
# We use global methods as a wrapper for member methods of the transformer
def MyParallelAutoThetaTransformer_fit_async(*args, **kwargs):
    return AutoThetaParallelModel._fit_async(*args, **kwargs)


def MyParallelAutoThetaTransformer_transform_async(*args, **kwargs):
    return AutoThetaParallelModel._transform_async(*args, **kwargs)


class AutoThetaParallelModel(CustomTimeSeriesModel):
    _regression = True
    _binary = False
    _multiclass = False
    _display_name = "Auto_Theta_Parallel"
    _description = "Auto Theta TimeSeries forecasting with multi process support"
    _parallel_task = True
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    _modules_needed_by_name = ['statsforecast==1.7.4', 'dask==2024.12.1',]
    
    @staticmethod
    def is_enabled():
        return not (arch_type == "ppc64le")

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return True  # by default too slow unless only enabled

    @staticmethod
    def do_acceptance_test():
        return True

    def set_default_params(
            self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs
    ):

        #only main parameter to set by user is the seasonality period for Theta.
        #if not set defaults to non-seasonal Theta
        self.params = dict(
            season_length=kwargs.get("season_length", 1)
        )

    def mutate_params(self, accuracy, time_tolerance, interpretability, **kwargs):
        # No mutation required for any parameters for AutoTheta
        pass
        

    @staticmethod
    def _get_n_jobs(logger, **kwargs):
        if "n_jobs_theta" in config.recipe_dict:
            return min(config.recipe_dict["n_jobs_theta"], max_threads())
        try:
            if config.fixed_num_folds <= 0:
                n_jobs = max(
                    1,
                    int(
                        int(
                            max_threads() / min(config.num_folds, kwargs["max_workers"])
                        )
                    ),
                )
            else:
                n_jobs = max(
                    1,
                    int(
                        int(
                            max_threads()
                            / min(
                                config.fixed_num_folds,
                                config.num_folds,
                                kwargs["max_workers"],
                            )
                        )
                    ),
                )
        except KeyError:
            loggerinfo(logger, "autotheta No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs if n_jobs > 1 else 1

    def _clean_tmp_folder(self, logger, tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            loggerinfo(logger, "autotheta cleaned up temporary file folder.")
        except:
            loggerwarning(
                logger, "autotheta could not delete the temporary file folder."
            )

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files used during multi processing experiment
        # This temp folder will be removed at the end of the process
        # Set the default value without context available (required to pass acceptance test
        tmp_folder = os.path.join(
            user_dir(), "%s_autotheta_model_folder" % uuid.uuid4()
        )
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(
                self.context.experiment_tmp_dir,
                "%s_autotheta_model_folder" % uuid.uuid4(),
            )

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This should not occur so log a warning
            loggerwarning(logger, "autotheta was denied temp folder creation rights")
            tmp_folder = os.path.join(
                user_dir(), "%s_autotheta_model_folder" % uuid.uuid4()
            )
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "autotheta temp folder already exists")
            tmp_folder = os.path.join(
                self.context.experiment_tmp_dir,
                "%s_autotheta_model_folder" % uuid.uuid4(),
            )
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(
                user_dir(), "%s_autotheta_model_folder" % uuid.uuid4()
            )
            os.mkdir(tmp_folder)

        loggerinfo(logger, "autotheta temp folder {}".format(tmp_folder))
        return tmp_folder

    @staticmethod
    def _fit_async(X_path, grp_hash, tmp_folder, exogenous_vars, params): #, cap):
        """
        Fits a autotheta model for a particular time group
        :param X_path: Path to the data used to fit the autotheta model
        :param grp_hash: Time group identifier
        :exogenous_vars: column names of additional predictors, AutoTheta does not use them
        :return: time group identifier and path to the pickled model
        """
        np.random.seed(1234)
        random.seed(1234)
        X = load_obj(X_path)
        # Commented for performance, uncomment for debug
        # print("autotheta - fitting on data of shape: %s for group: %s" % (str(X.shape), grp_hash))
        #if X.shape[0] < 20:
        #    return grp_hash, None

        # Import autotheta model
        mod = importlib.import_module("statsforecast.models")        
        model = mod.AutoTheta(season_length = params["season_length"])  
        
        with suppress_stdout_stderr():
            try:
                model.fit(X['y'].values)
            except:
                model = None # model fit failed for AutoTheta
                
        model_path = os.path.join(tmp_folder, "autotheta_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(X_path)  # remove to indicate success
        return grp_hash, model_path

    def get_hash(self, key):
        # Create dict key to store the min max scaler
        if isinstance(key, tuple):
            key = list(key)
        elif isinstance(key, list):
            pass
        else:
            # Not tuple, not list
            key = [key]
        grp_hash = "_".join(map(str, key))
        return grp_hash

    def fit(
            self,
            X,
            y,
            sample_weight=None,
            eval_set=None,
            sample_weight_eval_set=None,
            **kwargs,
    ):

        # Get TGC and time column
        self.tgc = self.params_base.get("tgc", None)
        self.time_column = self.params_base.get("time_column", None)
        self.nan_value = np.mean(y)
        self.prior = np.mean(y)

        if self.time_column is None:
            self.time_column = self.tgc[0]

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
            )

        loggerinfo(
            logger, "Start Fitting autotheta Model with params : {}".format(self.params)
        )

        try:
            # Add value of autotheta_top_n in recipe_dict variable inside of config.toml file
            # eg1: recipe_dict="{'autotheta_top_n': 200}"
            # eg2: recipe_dict="{'autotheta_top_n':10}"
            self.top_n = config.recipe_dict["autotheta_top_n"]
        except KeyError:
            self.top_n = 50

        loggerinfo(
            logger,
            f"autotheta will use {self.top_n} groups as well as average target data.",
        )

        # Get temporary folders for multi process communication
        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        X = X.to_pandas()

        #collect the column names of the non TGC columns used as predictors
        exogenous_vars = list( np.setdiff1d(X.columns, self.tgc + [self.time_column] ) )


        # Fill NaNs or None
        X = X.replace([None, np.nan], 0)

        # Add target, Label encoder is only used for Classif. which we don't support...
        if self.labels is not None:
            y = LabelEncoder().fit(self.labels).transform(y)
        X["y"] = np.array(y)

        self.nan_value = X["y"].mean()

        # Change date feature name to match autotheta requirements
        X.rename(columns={self.time_column: "ds"}, inplace=True)
        
        #Sort the X dataframe expects target values to be in sorted order
        if len(tgc_wo_time) > 0:
            cols_to_sort = tgc_wo_time + ["ds"]
        else:
            cols_to_sort = ["ds"]
        X.sort_values(cols_to_sort, inplace=True, ignore_index = True)

        # transform to datetime to try to infer the frequency
        X["ds"] = pd.to_datetime(X["ds"])
        infered_freq = pd.infer_freq(X["ds"].iloc[:5])
        if infered_freq is not None:
            # update season length
            self.params["season_length"] = FREQS.get(infered_freq, 1)
        
            loggerinfo(
                logger, "Updated season_length based on inference"
            )

        # Create a general scale now that will be used for unknown groups at prediction time
        # Can we do smarter than that ?
        general_scaler = MinMaxScaler().fit(
            X[["y", "ds"]].groupby("ds").median().values
        )

        # Go through groups and standard scale them
        if len(tgc_wo_time) > 0:
            X_groups = X.groupby(tgc_wo_time)
        else:
            X_groups = [([None], X)]

        scalers = {}
        scaled_ys = []

        print("Number of groups : ", len(X_groups))
        for g in tgc_wo_time:
            print(f"Number of groups in {g} groups : {X[g].unique().shape}") #improve message >>>>>>

        for key, X_grp in X_groups:
            # Create dict key to store the min max scaler
            grp_hash = self.get_hash(key)
            # Scale target for current group
            scalers[grp_hash] = MinMaxScaler()
            y_skl = scalers[grp_hash].fit_transform(X_grp[["y"]].values)
            # Put back in a DataFrame to keep track of original index
            y_skl_df = pd.DataFrame(y_skl, columns=["y"])

            y_skl_df.index = X_grp.index
            scaled_ys.append(y_skl_df)

        # Set target back in original frame but keep original
        X["y_orig"] = X["y"]
        X["y"] = pd.concat(tuple(scaled_ys), axis=0)

        # Now Average groups
        X_avg = X[["ds", "y"]].groupby("ds").mean().reset_index()

        # Send that to autotheta
        mod = importlib.import_module("statsforecast.models")
        model = mod.AutoTheta(season_length=self.params["season_length"])  

        with suppress_stdout_stderr():
            try:
                model.fit(X_avg['y'].values) #AutoTheta accepts only numpy arrays
            except:
                #try a fallback model i.e. SeasonalNaive
                model = mod.SeasonalNaive(season_length=self.params["season_length"])
                model.fit(X_avg['y'].values)

        top_groups = None
        if len(tgc_wo_time) > 0:
            if self.top_n > 0:
                top_n_grp = (
                    X.groupby(tgc_wo_time)
                        .size()
                        .sort_values()
                        .reset_index()[tgc_wo_time]
                        .iloc[-self.top_n:]
                        .values
                )
                top_groups = ["_".join(map(str, key)) for key in top_n_grp]

        grp_models = {}
        priors = {}
        if top_groups:
            # Prepare for multi processing
            num_tasks = len(top_groups)

            def processor(out, res):
                out[res[0]] = res[1]

            pool_to_use = small_job_pool
            loggerinfo(logger, f"autotheta will use {n_jobs} workers for fitting.")

            pool = pool_to_use(
                logger=None,
                processor=processor,
                num_tasks=num_tasks,
                max_workers=n_jobs,
            )
            
            # Fit 1 autotheta model per time group column
            nb_groups = len(X_groups)

            # Put y back to its unscaled value for top groups
            X["y"] = X["y_orig"]

            for _i_g, (key, X) in enumerate(X_groups):
                # Just log where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(
                        logger,
                        "autotheta : %d%% of groups fitted"
                        % (100 * (_i_g + 1) // nb_groups),
                    )


                grp_hash = self.get_hash(key)

                if grp_hash not in top_groups:
                    continue

                X_path = os.path.join(tmp_folder, "autotheta" + str(uuid.uuid4()))
                X = X.reset_index(drop=True)
                save_obj(X, X_path)
                priors[grp_hash] = X["y"].mean()

                args = (X_path, grp_hash, tmp_folder, exogenous_vars, self.params) 
                kwargs = {}
                pool.submit_tryget(
                    None,
                    MyParallelAutoThetaTransformer_fit_async,
                    args=args,
                    kwargs=kwargs,
                    out=grp_models,
                )
            pool.finish()

            for k, v in grp_models.items():
                grp_models[k] = load_obj(v) if v is not None else None
                remove(v)

        self._clean_tmp_folder(logger, tmp_folder)

        self.set_model_properties(
            model={
                "avg": model,
                "group": grp_models,
                "priors": priors,
                "topgroups": top_groups,
                "skl": scalers,
                "gen_scaler": general_scaler,
                "exogenous_vars" : exogenous_vars
            },
            
            features=self.tgc + exogenous_vars,
            importances=np.ones(len(self.tgc + exogenous_vars)),
            
            iterations=-1,  # Does not have iterations
        )

        return None

    @staticmethod
    def _transform_async(model_path, X_path, nan_value, exogenous_vars, tmp_folder):
        """
        Predicts target for a particular time group
        :param model_path: path to the stored model
        :param X_path: Path to the data used to fit the autotheta model
        :param nan_value: Value of target prior, used when no fitted model has been found
        :return: self
        """
        model = load_obj(model_path)
        XX_path = os.path.join(tmp_folder, "autotheta_XXt" + str(uuid.uuid4()))
        X = load_obj(X_path)
        # autotheta returns the predictions ordered by time
        # So we should keep track of the time order for each group so that
        # predictions are ordered the same as the imput frame
        # Keep track of the order
        order = np.argsort(pd.to_datetime(X["ds"]))
        if model is not None:
            try:
                # Run AutoTheta
                yhat = model.predict(h = X.shape[0])['mean']   
                
                XX = pd.DataFrame(yhat, columns=["yhat"])
            except:
                XX = pd.DataFrame(
                np.full((X.shape[0], 1), nan_value), columns=["yhat"]
                ) # AutoTheta generated error while predicting
                
        else:
            XX = pd.DataFrame(
                np.full((X.shape[0], 1), nan_value), columns=["yhat"]
            )  # invalid models
            
        XX.index = X.index[order]
        assert XX.shape[1] == 1
        save_obj(XX, XX_path)
        remove(model_path)  # indicates success, no longer need
        remove(X_path)  # indicates success, no longer need
        return XX_path

    def predict(self, X: dt.Frame, **kwargs):
        """
        Uses fitted models (1 per time group) to predict the target
        :param X: Datatable Frame containing the features
        :return: autotheta predictions
        """
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
            )

        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            loggerdebug(logger, "Return 0 predictions")
            return np.ones(X.shape[0]) * self.nan_value

        models, _, _, _ = self.get_model_properties()

        avg_model = models["avg"]
        grp_models = models["group"]
        priors = models["priors"]
        top_groups = models["topgroups"]
        scalers = models["skl"]
        general_scaler = models["gen_scaler"]
        exogenous_vars = models["exogenous_vars"]

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        X = X.to_pandas()

        # Fill NaNs or None
        X = X.replace([None, np.nan], 0)

        # Change date feature name to match autotheta requirements
        X.rename(columns={self.time_column: "ds"}, inplace=True)

        # Predict y using unique dates
        X_time = X[["ds"]].groupby("ds").first().reset_index()
        with suppress_stdout_stderr():
            y_avg = avg_model.predict(h = X_time.shape[0])['mean']
            

        ### AutoTheta returns the results in sorted time order so allow for that
        X_time.sort_values("ds", inplace=True)
        X_time["yhat"] = y_avg
        X_time.sort_index(inplace=True)

        # Merge back into original frame on 'ds'
        # pd.merge wipes the index ... so keep it to provide it again
        indices = X.index
        X = pd.merge(left=X, right=X_time[["ds", "yhat"]], on="ds", how="left")
        X.index = indices

        # Go through groups and recover the scaled target for knowed groups
        if len(tgc_wo_time) > 0:
            X_groups = X.groupby(tgc_wo_time)
        else:
            X_groups = [([None], X)]

        inverted_ys = []
        for key, X_grp in X_groups:
            grp_hash = self.get_hash(key)

            # Scale target for current group
            if grp_hash in scalers.keys():
                inverted_y = scalers[grp_hash].inverse_transform(X_grp[["yhat"]])
            else:
                inverted_y = general_scaler.inverse_transform(X_grp[["yhat"]])

            # Put back in a DataFrame to keep track of original index
            inverted_df = pd.DataFrame(inverted_y, columns=["yhat"])
            inverted_df.index = X_grp.index
            inverted_ys.append(inverted_df)

        XX_general = pd.concat(tuple(inverted_ys), axis=0).sort_index()

        if top_groups:
            # Go though the groups and predict only top
            XX_paths = []
            model_paths = []

            def processor(out, res):
                out.append(res)

            num_tasks = len(top_groups)
            pool_to_use = small_job_pool
            pool = pool_to_use(
                logger=None,
                processor=processor,
                num_tasks=num_tasks,
                max_workers=n_jobs,
            )


            nb_groups = len(X_groups)
            for _i_g, (key, X_grp) in enumerate(X_groups):

                # Just log where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(
                        logger,
                        "autotheta : %d%% of groups predicted"
                        % (100 * (_i_g + 1) // nb_groups),
                    )

                # Create dict key to store the min max scaler
                grp_hash = self.get_hash(key)
                X_path = os.path.join(tmp_folder, "autotheta_Xt" + str(uuid.uuid4()))

                if grp_hash not in top_groups:
                    XX = pd.DataFrame(
                        np.full((X_grp.shape[0], 1), np.nan), columns=["yhat"]
                    )  # unseen groups
                    XX.index = X_grp.index
                    save_obj(XX, X_path)
                    XX_paths.append(X_path)
                    continue

                if grp_models[grp_hash] is None:
                    XX = pd.DataFrame(
                        np.full((X_grp.shape[0], 1), np.nan), columns=["yhat"]
                    )  # unseen groups
                    XX.index = X_grp.index
                    save_obj(XX, X_path)
                    XX_paths.append(X_path)
                    continue

                model = grp_models[grp_hash]
                model_path = os.path.join(
                    tmp_folder, "autotheta_modelt" + str(uuid.uuid4())
                )
                save_obj(model, model_path)
                save_obj(X_grp, X_path)
                model_paths.append(model_path)

                args = (model_path, X_path, priors[grp_hash], exogenous_vars, tmp_folder)                
                kwargs = {}
                pool.submit_tryget(
                    None,
                    MyParallelAutoThetaTransformer_transform_async,
                    args=args,
                    kwargs=kwargs,
                    out=XX_paths,
                )

            pool.finish()
            XX_top_groups = pd.concat(
                (load_obj(XX_path) for XX_path in XX_paths), axis=0
            ).sort_index()
            for p in XX_paths + model_paths:
                remove(p)

        self._clean_tmp_folder(logger, tmp_folder)

        features_df = pd.DataFrame()
        features_df["GrpAvg"] = XX_general["yhat"]

        if top_groups:
            features_df[f"_Top{self.top_n}Grp"] = XX_top_groups["yhat"]
            features_df.loc[
                features_df[f"_Top{self.top_n}Grp"].notnull(), "GrpAvg"
            ] = features_df.loc[
                features_df[f"_Top{self.top_n}Grp"].notnull(), f"_Top{self.top_n}Grp"
            ]

        # Models have to return a numpy array
        return features_df["GrpAvg"].values
    
