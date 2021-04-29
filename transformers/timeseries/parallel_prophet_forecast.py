"""Parallel FB Prophet transformer is a time series transformer that predicts target using FBProphet models."""

"""
This transformer fits one FBProphet model per time group and therefore may take time. Before using this transformer
we suggest you check FBProphet prediction significance by running an experiment with
parallel_prophet_forecast_using_individual_groups. Then enable parallel prophet forecast to get even better predictions."""

"""
In this implementation, Time Group Models are fitted in parallel

The recipe outputs 2 predictors:
 - The first one is trained on the average of the target over the time column
 - The second one is trained on TopN groups, where TopN is defined by recipe_dict in config.toml.
   These groups are those with the highest number of data points.
   
If TopN is not defined in config.toml set using the toml override in the expert settings,
 TopN group defaults to 1. Setting TopN is done with recipe_dict="{'prophet_top_n': 200}"
 
You may also want to modify the parameters explored line 99 to 103 to fit your needs. 
"""
import importlib
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
from h2oaicore.systemutils import (
    small_job_pool, save_obj, load_obj, remove, max_threads, config,
    user_dir)
import datatable as dt
import numpy as np
import os
import uuid
import shutil
import random
import importlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from datetime import datetime


# For more information about FB prophet please visit :

# This parallel implementation is faster than the serial implementation
# available in the repository.
# Standard implementation is therefore disabled.

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
def MyParallelProphetTransformer_fit_async(*args, **kwargs):
    return MyParallelProphetTransformer._fit_async(*args, **kwargs)


def MyParallelProphetTransformer_transform_async(*args, **kwargs):
    return MyParallelProphetTransformer._transform_async(*args, **kwargs)


class MyParallelProphetTransformer(CustomTimeSeriesTransformer):
    """Implementation of the FB Prophet transformer using a pool of processes to fit models in parallel"""
    _is_reproducible = True
    _binary = False
    _multiclass = False
    # some package dependencies are best sequential to overcome known issues
    _modules_needed_by_name = ['convertdate', 'pystan==3.0.2', 'fbprophet==0.7.1']
    _included_model_classes = None  # ["gblinear"] for strong trends - can extrapolate
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(
            self,
            country_holidays=None,
            monthly_seasonality=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.country_holidays = country_holidays
        self.monthly_seasonality = monthly_seasonality

    @property
    def display_name(self):
        name = "FBProphet"
        if self.country_holidays is not None:
            name += "_Holiday_{}".format(self.country_holidays)
        if self.monthly_seasonality:
            name += "_Month"
        return name

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {
            "country_holidays": [None, "US"],
            "monthly_seasonality": [False, True],
        }

    @staticmethod
    def acceptance_test_timeout():
        return 30  # allow for 20 minutes to do acceptance test

    @staticmethod
    def is_enabled():
        return False

    @staticmethod
    def _fit_async(X_path, grp_hash, tmp_folder, params):
        """
        Fits a FB Prophet model for a particular time group
        :param X_path: Path to the data used to fit the FB Prophet model
        :param grp_hash: Time group identifier
        :return: time group identifier and path to the pickled model
        """
        np.random.seed(1234)
        random.seed(1234)
        X = load_obj(X_path)
        # Commented for performance, uncomment for debug
        # print("prophet - fitting on data of shape: %s for group: %s" % (str(X.shape), grp_hash))
        if X.shape[0] < 20:
            # print("prophet - small data work-around for group: %s" % grp_hash)
            return grp_hash, None
        # Import FB Prophet package
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

        if params["country_holidays"] is not None:
            model.add_country_holidays(country_name=params["country_holidays"])
        if params["monthly_seasonality"]:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        with suppress_stdout_stderr():
            model.fit(X[['ds', 'y']])
        model_path = os.path.join(tmp_folder, "fbprophet_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(X_path)  # remove to indicate success
        return grp_hash, model_path

    def _get_n_jobs(self, logger, **kwargs):
        try:
            if config.fixed_num_folds <= 0:
                n_jobs = max(1, int(int(max_threads() / min(config.num_folds, kwargs['max_workers']))))
            else:
                n_jobs = max(1, int(
                    int(max_threads() / min(config.fixed_num_folds, config.num_folds, kwargs['max_workers']))))
        except KeyError:
            loggerinfo(logger, "Prophet No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs if n_jobs > 1 else 1

    def _clean_tmp_folder(self, logger, tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            loggerinfo(logger, "Prophet cleaned up temporary file folder.")
        except:
            loggerwarning(logger, "Prophet could not delete the temporary file folder.")

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files used during multi processing experiment
        # This temp folder will be removed at the end of the process
        # Set the default value without context available (required to pass acceptance test
        tmp_folder = os.path.join(user_dir(), "%s_prophet_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_prophet_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "Prophet was denied temp folder creation rights")
            tmp_folder = os.path.join(user_dir(), "%s_prophet_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "Prophet temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_prophet_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(user_dir(), "%s_prophet_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "Prophet temp folder {}".format(tmp_folder))
        return tmp_folder

    def fit(self, X: dt.Frame, y: np.array = None, **kwargs):
        """
        Fits FB Prophet models (1 per time group) using historical target values contained in y
        Model fitting is distributed over a pool of processes and uses file storage to share the data with workers
        :param X: Datatable frame containing the features
        :param y: numpy array containing the historical values of the target
        :return: self
        """

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir,
                username=self.context.username,
            )

        try:
            # Add value of prophet_top_n in recipe_dict variable inside of config.toml file
            # eg1: recipe_dict="{'prophet_top_n': 200}"
            # eg2: recipe_dict="{'prophet_top_n':10}"
            self.top_n = config.recipe_dict['prophet_top_n']
        except KeyError:
            self.top_n = 50

        loggerinfo(logger, f"Prophet will use {self.top_n} groups as well as average target data.")

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Reduce X to TGC
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        X = X[:, self.tgc].to_pandas()

        # Fill NaNs or None
        X = X.replace([None, np.nan], 0)

        # Add target, Label encoder is only used for Classif. which we don't support...
        if self.labels is not None:
            y = LabelEncoder().fit(self.labels).transform(y)
        X['y'] = np.array(y)

        self.nan_value = X['y'].mean()

        # Change date feature name to match Prophet requirements
        X.rename(columns={self.time_column: "ds"}, inplace=True)

        # Create a general scale now that will be used for unknown groups at prediction time
        # Can we do smarter than that ?
        self.general_scaler = MinMaxScaler().fit(X[['y', 'ds']].groupby('ds').median().values)

        # Go through groups and standard scale them
        if len(tgc_wo_time) > 0:
            X_groups = X.groupby(tgc_wo_time)
        else:
            X_groups = [([None], X)]

        self.scalers = {}
        scaled_ys = []
        print(f'{datetime.now()} Start of group scaling')

        for key, X_grp in X_groups:
            # Create dict key to store the min max scaler
            grp_hash = self.get_hash(key)
            # Scale target for current group
            self.scalers[grp_hash] = MinMaxScaler()
            y_skl = self.scalers[grp_hash].fit_transform(X_grp[['y']].values)
            # Put back in a DataFrame to keep track of original index
            y_skl_df = pd.DataFrame(y_skl, columns=['y'])
            # (0, 'A') (1, 4) (100, 1) (100, 1)
            # print(grp_hash, X_grp.shape, y_skl.shape, y_skl_df.shape)

            y_skl_df.index = X_grp.index
            scaled_ys.append(y_skl_df)

        print(f'{datetime.now()} End of group scaling')
        # Set target back in original frame but keep original
        X['y_orig'] = X['y']
        X['y'] = pd.concat(tuple(scaled_ys), axis=0)

        # Now Average groups
        X_avg = X[['ds', 'y']].groupby('ds').mean().reset_index()

        # Send that to Prophet
        params = {
            "country_holidays": self.country_holidays,
            "monthly_seasonality": self.monthly_seasonality
        }
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)

        if params["country_holidays"] is not None:
            self.model.add_country_holidays(country_name=params["country_holidays"])
        if params["monthly_seasonality"]:
            self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        with suppress_stdout_stderr():
            self.model.fit(X[['ds', 'y']])

        print(f'{datetime.now()} General Model Fitted')

        self.top_groups = None
        if len(tgc_wo_time) > 0:
            if self.top_n > 0:
                top_n_grp = X.groupby(tgc_wo_time).size().sort_values().reset_index()[tgc_wo_time].iloc[
                            -self.top_n:].values
                self.top_groups = [
                    '_'.join(map(str, key))
                    for key in top_n_grp
                ]

        if self.top_groups:
            self.grp_models = {}
            self.priors = {}

            # Prepare for multi processing
            num_tasks = len(self.top_groups)

            def processor(out, res):
                out[res[0]] = res[1]

            pool_to_use = small_job_pool
            loggerinfo(logger, f"Prophet will use {n_jobs} workers for fitting.")
            loggerinfo(logger, "Prophet parameters holidays {} / monthly {}".format(self.country_holidays,
                                                                                    self.monthly_seasonality))
            pool = pool_to_use(
                logger=None, processor=processor,
                num_tasks=num_tasks, max_workers=n_jobs
            )
            #
            # Fit 1 FB Prophet model per time group columns
            nb_groups = len(X_groups)

            # Put y back to its unscaled value for top groups
            X['y'] = X['y_orig']

            for _i_g, (key, X) in enumerate(X_groups):
                # Just log where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(logger, "FB Prophet : %d%% of groups fitted" % (100 * (_i_g + 1) // nb_groups))

                X_path = os.path.join(tmp_folder, "fbprophet_X" + str(uuid.uuid4()))
                X = X.reset_index(drop=True)
                save_obj(X, X_path)

                grp_hash = self.get_hash(key)

                if grp_hash not in self.top_groups:
                    continue

                self.priors[grp_hash] = X['y'].mean()

                params = {
                    "country_holidays": self.country_holidays,
                    "monthly_seasonality": self.monthly_seasonality
                }

                args = (X_path, grp_hash, tmp_folder, params)
                kwargs = {}
                pool.submit_tryget(None, MyParallelProphetTransformer_fit_async,
                                   args=args, kwargs=kwargs, out=self.grp_models)
            pool.finish()

            for k, v in self.grp_models.items():
                self.grp_models[k] = load_obj(v) if v is not None else None
                remove(v)

        self._clean_tmp_folder(logger, tmp_folder)

        return self

    @staticmethod
    def _transform_async(model_path, X_path, nan_value, tmp_folder):
        """
        Predicts target for a particular time group
        :param model_path: path to the stored model
        :param X_path: Path to the data used to fit the FB Prophet model
        :param nan_value: Value of target prior, used when no fitted model has been found
        :return: self
        """
        model = load_obj(model_path)
        XX_path = os.path.join(tmp_folder, "fbprophet_XX" + str(uuid.uuid4()))
        X = load_obj(X_path)
        # Facebook Prophet returns the predictions ordered by time
        # So we should keep track of the time order for each group so that
        # predictions are ordered the same as the imput frame
        # Keep track of the order
        order = np.argsort(pd.to_datetime(X["ds"]))
        if model is not None:
            # Run prophet
            yhat = model.predict(X)['yhat'].values
            XX = pd.DataFrame(yhat, columns=['yhat'])
        else:
            XX = pd.DataFrame(np.full((X.shape[0], 1), nan_value), columns=['yhat'])  # invalid models
        XX.index = X.index[order]
        assert XX.shape[1] == 1
        save_obj(XX, XX_path)
        remove(model_path)  # indicates success, no longer need
        remove(X_path)  # indicates success, no longer need
        return XX_path

    def transform(self, X: dt.Frame, **kwargs):
        """
        Uses fitted models (1 per time group) to predict the target
        :param X: Datatable Frame containing the features
        :return: FB Prophet predictions
        """
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Reduce X to TGC
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        X = X[:, self.tgc].to_pandas()

        # Fill NaNs or None
        X = X.replace([None, np.nan], 0)

        # Change date feature name to match Prophet requirements
        X.rename(columns={self.time_column: "ds"}, inplace=True)

        # Predict y using unique dates
        X_time = X[['ds']].groupby('ds').first().reset_index()
        with suppress_stdout_stderr():
            y_avg = self.model.predict(X_time)[['ds', 'yhat']]

        # Prophet transforms the date column to datetime so we need to transfrom that to merge back
        X_time.sort_values('ds', inplace=True)
        X_time['yhat'] = y_avg['yhat']
        X_time.sort_index(inplace=True)

        # Merge back into original frame on 'ds'
        # pd.merge wipes the index ... so keep it to provide it again
        indices = X.index
        X = pd.merge(
            left=X,
            right=X_time[['ds', 'yhat']],
            on='ds',
            how='left'
        )
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
            if grp_hash in self.scalers.keys():
                inverted_y = self.scalers[grp_hash].inverse_transform(X_grp[['yhat']])
            else:
                inverted_y = self.general_scaler.inverse_transform(X_grp[['yhat']])

            # Put back in a DataFrame to keep track of original index
            inverted_df = pd.DataFrame(inverted_y, columns=['yhat'])
            inverted_df.index = X_grp.index
            inverted_ys.append(inverted_df)

        XX_general = pd.concat(tuple(inverted_ys), axis=0).sort_index()

        if self.top_groups:
            # Go though the groups and predict only top
            XX_paths = []
            model_paths = []

            def processor(out, res):
                out.append(res)

            num_tasks = len(self.top_groups)
            pool_to_use = small_job_pool
            pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks, max_workers=n_jobs)

            nb_groups = len(X_groups)
            for _i_g, (key, X_grp) in enumerate(X_groups):

                # Just log where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(logger, "FB Prophet : %d%% of groups predicted" % (100 * (_i_g + 1) // nb_groups))

                # Create dict key to store the min max scaler
                grp_hash = self.get_hash(key)
                X_path = os.path.join(tmp_folder, "fbprophet_Xt" + str(uuid.uuid4()))

                if grp_hash not in self.top_groups:
                    XX = pd.DataFrame(np.full((X_grp.shape[0], 1), np.nan), columns=['yhat'])  # unseen groups
                    XX.index = X_grp.index
                    save_obj(XX, X_path)
                    XX_paths.append(X_path)
                    continue

                if self.grp_models[grp_hash] is None:
                    XX = pd.DataFrame(np.full((X_grp.shape[0], 1), np.nan), columns=['yhat'])  # unseen groups
                    XX.index = X_grp.index
                    save_obj(XX, X_path)
                    XX_paths.append(X_path)
                    continue

                model = self.grp_models[grp_hash]
                model_path = os.path.join(tmp_folder, "fbprophet_modelt" + str(uuid.uuid4()))
                save_obj(model, model_path)
                save_obj(X_grp, X_path)
                model_paths.append(model_path)

                args = (model_path, X_path, self.priors[grp_hash], tmp_folder)
                kwargs = {}
                pool.submit_tryget(None, MyParallelProphetTransformer_transform_async, args=args, kwargs=kwargs,
                                   out=XX_paths)

            pool.finish()
            XX_top_groups = pd.concat((load_obj(XX_path) for XX_path in XX_paths), axis=0).sort_index()
            for p in XX_paths + model_paths:
                remove(p)

        self._clean_tmp_folder(logger, tmp_folder)

        features_df = pd.DataFrame()
        features_df[self.display_name + '_GrpAvg'] = XX_general['yhat']

        if self.top_groups:
            features_df[self.display_name + f'_Top{self.top_n}Grp'] = XX_top_groups['yhat']

        self._output_feature_names = list(features_df.columns)
        self._feature_desc = list(features_df.columns)

        return features_df

    def get_hash(self, key):
        # Create dict key to store the min max scaler
        if isinstance(key, tuple):
            key = list(key)
        elif isinstance(key, list):
            pass
        else:
            # Not tuple, not list
            key = [key]
        grp_hash = '_'.join(map(str, key))
        return grp_hash

    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):
        """
        Fits the FB Prophet models (1 per time group) and outputs the corresponding predictions
        :param X: Datatable Frame
        :param y: Target to be used to fit FB Prophet models
        :return: FB Prophet predictions
        """
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
