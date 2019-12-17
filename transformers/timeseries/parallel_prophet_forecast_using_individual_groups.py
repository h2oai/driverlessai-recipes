"""Parallel FB Prophet transformer is a time series transformer that predicts target using FBProphet models.
In this implementation, Time Group Models are fitted in parallel

The recipe outputs following predictors:
 - First predictor is trained on average target over time column
 - Other predictors are trained on each individual time group.

If the dataset has 2 groups like department and stores:
  - one predictor will be trained on the target averaged by departments and time
  - a second predictor will be trained on the target averaged by stores and time

This implementation is faster than the standard parallel implementation, which trains one model
per time group and is able to bring similar performance.
"""

import importlib
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
from h2oaicore.systemutils import (
    small_job_pool, save_obj, load_obj, temporary_files_path, remove, max_threads, config
)
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
def MyProphetOnSingleGroupsTransformer_fit_async(*args, **kwargs):
    return MyProphetOnSingleGroupsTransformer._fit_async(*args, **kwargs)


def MyProphetOnSingleGroupsTransformer_transform_async(*args, **kwargs):
    return MyProphetOnSingleGroupsTransformer._transform_async(*args, **kwargs)


def fit_prophet_model(Prophet, X_avg, params):
    avg_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
    if params["country_holidays"] is not None:
        avg_model.add_country_holidays(country_name=params["country_holidays"])
    if params["monthly_seasonality"]:
        avg_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    with suppress_stdout_stderr():
        avg_model.fit(X_avg[['ds', 'y']])
    return avg_model


class MyProphetOnSingleGroupsTransformer(CustomTimeSeriesTransformer):
    """Implementation of the FB Prophet transformer using a pool of processes to fit models in parallel"""
    _is_reproducible = True
    _binary = False
    _multiclass = False
    # some package dependencies are best sequential to overcome known issues
    _modules_needed_by_name = ['convertdate', 'pystan==2.18', 'fbprophet==0.4.post2']
    _included_model_classes = None  # ["gblinear"] for strong trends - can extrapolate

    def __init__(
            self,
            country_holidays=None,
            monthly_seasonality=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.country_holidays = country_holidays
        self.monthly_seasonality = monthly_seasonality
        self.general_scaler = None
        self.scalers = None
        self.avg_model = None
        self.grp_models = None
        self.priors = None

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
        return 20  # allow for 20 minutes to do acceptance test

    @staticmethod
    def _fit_async(data_path, grp_hash, tmp_folder, params):
        """
        Fits a FB Prophet model for a particular time group
        :param data_path: Path to the data used to fit the FB Prophet model
        :param grp_hash: Time group identifier
        :return: time group identifier and path to the pickled model
        """
        np.random.seed(1234)
        random.seed(1234)
        X = load_obj(data_path)

        if X.shape[0] < 20:
            return grp_hash, None
        # Import FB Prophet package
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")
        model = fit_prophet_model(Prophet, X, params)
        model_path = os.path.join(tmp_folder, "fbprophet_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(data_path)  # remove to indicate success
        return grp_hash, model_path

    @staticmethod
    def _get_n_jobs(logger, **kwargs):
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

    @staticmethod
    def _clean_tmp_folder(logger, tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            loggerinfo(logger, "Prophet cleaned up temporary file folder.")
        except:
            loggerwarning(logger, "Prophet could not delete the temporary file folder.")

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files used during multi processing experiment
        # This temp folder will be removed at the end of the process
        # Set the default value without context available (required to pass acceptance test
        tmp_folder = os.path.join(temporary_files_path, "%s_prophet_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_prophet_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "Prophet was denied temp folder creation rights")
            tmp_folder = os.path.join(temporary_files_path, "%s_prophet_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "Prophet temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_prophet_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(temporary_files_path, "%s_prophet_folder" % uuid.uuid4())
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
        logger = self.get_experiment_logger()

        loggerinfo(logger, f"Prophet will use individual groups as well as average target data.")

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Reduce X to TGC
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))

        X = self.convert_to_prophet(X)

        # Add target, Label encoder is only used for Classif. which we don't support...
        if self.labels is not None:
            y = LabelEncoder().fit(self.labels).transform(y)
        X['y'] = np.array(y)

        self.prior_value = X['y'].mean()

        self.general_scaler = self.fit_scaler_to_median_target(X)

        X = self.scale_target_for_each_time_group(X, tgc_wo_time)

        self.avg_model = self.fit_prophet_model_on_average_target(X)

        # Go through individual time group columns and create avg models
        self.grp_models = {}
        self.priors = {}
        for grp_col in tgc_wo_time:
            self.grp_models[grp_col] = {}
            self.priors[grp_col] = {}

            X_groups = X[['ds', 'y', grp_col]].groupby(grp_col)

            nb_groups = len(X_groups)

            def processor(out, res):
                out[res[0]] = res[1]

            pool_to_use = small_job_pool
            loggerinfo(logger, f"Prophet will use {n_jobs} workers for fitting.")
            loggerinfo(logger, "Prophet parameters holidays {} / monthly {}".format(self.country_holidays,
                                                                                    self.monthly_seasonality))
            pool = pool_to_use(logger=None, processor=processor, num_tasks=nb_groups, max_workers=n_jobs)

            for _i_g, (key, X_grp) in enumerate(X_groups):
                # Just log where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(logger, "FB Prophet : %d%% of groups fitted" % (100 * (_i_g + 1) // nb_groups))

                X_path = os.path.join(tmp_folder, "fbprophet_X" + str(uuid.uuid4()))

                # Save target average for current group
                grp_hash = self.get_hash(key)
                self.priors[grp_col][grp_hash] = X_grp['y'].mean()

                # Average by date
                X_grp_avg = X_grp.groupby('ds')['y'].mean().reset_index()

                save_obj(X_grp_avg, X_path)

                params = {
                    "country_holidays": self.country_holidays,
                    "monthly_seasonality": self.monthly_seasonality
                }

                args = (X_path, grp_hash, tmp_folder, params)
                kwargs = {}
                pool.submit_tryget(None, MyProphetOnSingleGroupsTransformer_fit_async,
                                   args=args, kwargs=kwargs, out=self.grp_models[grp_col])
            pool.finish()

            for k, v in self.grp_models[grp_col].items():
                self.grp_models[grp_col][k] = load_obj(v) if v is not None else None
                remove(v)

        self._clean_tmp_folder(logger, tmp_folder)

        return self

    def fit_prophet_model_on_average_target(self, X):
        # Now Average groups
        X_avg = X[['ds', 'y']].groupby('ds').mean().reset_index()

        # Send that to Prophet
        params = {
            "country_holidays": self.country_holidays,
            "monthly_seasonality": self.monthly_seasonality
        }
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")
        avg_model = fit_prophet_model(Prophet, X_avg, params)

        return avg_model

    def scale_target_for_each_time_group(self, X, tgc_wo_time):
        # Go through groups and standard scale them
        if len(tgc_wo_time) > 0:
            X_groups = X.groupby(tgc_wo_time)
        else:
            X_groups = [([None], X)]

        self.scalers = {}
        scaled_ys = []
        for key, X_grp in X_groups:
            # Create dict key to store the min max scaler
            grp_hash = self.get_hash(key)
            # Scale target for current group
            self.scalers[grp_hash] = MinMaxScaler()
            y_skl = self.scalers[grp_hash].fit_transform(X_grp[['y']].values)
            # Put back in a DataFrame to keep track of original index
            y_skl_df = pd.DataFrame(y_skl, columns=['y'])
            y_skl_df.index = X_grp.index
            scaled_ys.append(y_skl_df)
        # Set target back in original frame but keep original
        X['y_orig'] = X['y']
        X['y'] = pd.concat(tuple(scaled_ys), axis=0)
        return X

    def get_experiment_logger(self):
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )
        return logger

    @staticmethod
    def fit_scaler_to_median_target(X):
        # Create a general scale now that will be used for unknown groups at prediction time
        # Can we do smarter than that ?
        median = X[['y', 'ds']].groupby('ds').median()
        median = median.fillna(median.values.mean())
        return MinMaxScaler().fit(median.values)

    def convert_to_prophet(self, X):
        # Change date feature name to match Prophet requirement
        return X[:, self.tgc].to_pandas().rename(columns={self.time_column: "ds"})

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

        X_time = X[['ds']].groupby('ds').first().reset_index()
        with suppress_stdout_stderr():
            y_avg = model.predict(X_time)[['ds', 'yhat']]

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

        save_obj(X[['yhat']], XX_path)
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
        logger = self.get_experiment_logger()

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Reduce X to TGC
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))

        # Change date feature name to match Prophet requirements
        X = self.convert_to_prophet(X)

        y_predictions = self.predict_with_average_model(X, tgc_wo_time)
        y_predictions.columns = ['average_pred']

        # Go through groups
        for grp_col in tgc_wo_time:
            # Get the unique dates to be predicted
            X_groups = X[['ds', grp_col]].groupby(grp_col)

            # Go though the groups and predict only top
            XX_paths = []
            model_paths = []

            def processor(out, res):
                out.append(res)
            num_tasks = len(X_groups)
            pool_to_use = small_job_pool
            pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks, max_workers=n_jobs)

            for _i_g, (key, X_grp) in enumerate(X_groups):

                # Just log where we are in the fitting process
                if (_i_g + 1) % max(1, num_tasks // 20) == 0:
                    loggerinfo(logger, "FB Prophet : %d%% of groups predicted" % (100 * (_i_g + 1) // num_tasks))

                # Create dict key to store the min max scaler
                grp_hash = self.get_hash(key)
                X_path = os.path.join(tmp_folder, "fbprophet_Xt" + str(uuid.uuid4()))

                if grp_hash not in self.grp_models[grp_col]:
                    # unseen groups
                    XX = pd.DataFrame(np.full((X_grp.shape[0], 1), np.nan), columns=['yhat'])
                    XX.index = X_grp.index
                    save_obj(XX, X_path)
                    XX_paths.append(X_path)
                    continue

                if self.grp_models[grp_col][grp_hash] is None:
                    # known groups but not enough train data
                    XX = pd.DataFrame(np.full((X_grp.shape[0], 1), np.nan), columns=['yhat'])
                    XX.index = X_grp.index
                    save_obj(XX, X_path)
                    XX_paths.append(X_path)
                    continue

                model = self.grp_models[grp_col][grp_hash]
                model_path = os.path.join(tmp_folder, "fbprophet_modelt" + str(uuid.uuid4()))
                save_obj(model, model_path)
                save_obj(X_grp, X_path)
                model_paths.append(model_path)

                args = (model_path, X_path, self.priors[grp_col][grp_hash], tmp_folder)
                kwargs = {}
                pool.submit_tryget(None, MyProphetOnSingleGroupsTransformer_transform_async, args=args, kwargs=kwargs,
                                   out=XX_paths)

            pool.finish()
            y_predictions[f'{grp_col}_pred'] = pd.concat((load_obj(XX_path) for XX_path in XX_paths), axis=0).sort_index()
            for p in XX_paths + model_paths:
                remove(p)

        # Now we can invert scale
        # But first get rid of NaNs
        for grp_col in tgc_wo_time:
            # Add time group to the predictions, will be used to invert scaling
            y_predictions[grp_col] = X[grp_col]
            # Fill NaN
            y_predictions[f'{grp_col}_pred'] = y_predictions[f'{grp_col}_pred'].fillna(y_predictions['average_pred'])

        # Go through groups and recover the scaled target for knowed groups
        if len(tgc_wo_time) > 0:
            X_groups = y_predictions.groupby(tgc_wo_time)
        else:
            X_groups = [([None], y_predictions)]

        for _f in [f'{grp_col}_pred' for grp_col in tgc_wo_time] + ['average_pred']:
            inverted_ys = []
            for key, X_grp in X_groups:
                grp_hash = self.get_hash(key)
                # Scale target for current group
                if grp_hash in self.scalers.keys():
                    inverted_y = self.scalers[grp_hash].inverse_transform(X_grp[[_f]])
                else:
                    inverted_y = self.general_scaler.inverse_transform(X_grp[[_f]])

                # Put back in a DataFrame to keep track of original index
                inverted_df = pd.DataFrame(inverted_y, columns=[_f])
                inverted_df.index = X_grp.index
                inverted_ys.append(inverted_df)
            y_predictions[_f] = pd.concat(tuple(inverted_ys), axis=0).sort_index()[_f]

        self._clean_tmp_folder(logger, tmp_folder)

        y_predictions.drop(tgc_wo_time, axis=1, inplace=True)

        self._output_feature_names = [f'{self.display_name}_{_f}' for _f in y_predictions]
        self._feature_desc = [f'{self.display_name}_{_f}' for _f in y_predictions]

        return y_predictions

    def predict_with_average_model(self, X, tgc_wo_time):
        # Predict y using unique dates
        X_time = X[['ds']].groupby('ds').first().reset_index()
        with suppress_stdout_stderr():
            y_avg = self.avg_model.predict(X_time)[['ds', 'yhat']]
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

        return X[['yhat']]

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
