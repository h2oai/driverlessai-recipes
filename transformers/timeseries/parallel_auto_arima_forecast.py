"""Parallel Auto ARIMA transformer is a time series transformer that predicts target using ARIMA models.
In this implementation, Time Group Models are fitted in parallel"""

# For more information about the python ARIMA package
# please visit https://www.alkaline-ml.com/pmdarima/index.html

# Please note that depending on your server setup, the parallel implementation may not be faster

import importlib
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
from h2oaicore.systemutils import (
    small_job_pool, save_obj, load_obj, user_dir, remove, config, max_threads
)
import datatable as dt
import numpy as np
import pandas as pd
import random
import os
import uuid
import shutil
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
def MyParallelAutoArimaTransformer_fit_async(*args, **kwargs):
    return MyParallelAutoArimaTransformer._fit_async(*args, **kwargs)


def MyParallelAutoArimaTransformer_transform_async(*args, **kwargs):
    return MyParallelAutoArimaTransformer._transform_async(*args, **kwargs)


class MyParallelAutoArimaTransformer(CustomTimeSeriesTransformer):
    """Implementation of the ARIMA transformer using a pool of processes to fit models in parallel"""
    _binary = False
    _multiclass = False
    _modules_needed_by_name = ['pmdarima==1.8.3']
    _included_model_classes = None
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _lag_recipe_allowed = True
    _causal_recipe_allowed = False

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    # Disable parallel AutoArima since current implementation is not faster than auto_arima_forecast.py
    @staticmethod
    def is_enabled():
        return False

    @staticmethod
    def _fit_async(X_path, grp_hash, time_column, tmp_folder):
        """
        Fits an ARIMA model for a particular time group
        :param X_path: Path to the data used to fit the ARIMA model
        :param grp_hash: Time group identifier
        :param time_column: Name of the time column in the input data
        :return: time group identifier and path to the pickled model
        """
        np.random.seed(1234)
        random.seed(1234)
        X = load_obj(X_path)

        pm = importlib.import_module('pmdarima')
        with suppress_stdout_stderr():
            try:
                order = order = np.argsort(X[time_column])
                model = pm.auto_arima(X['y'].values[order], error_action='ignore')
            except:
                model = None
        model_path = os.path.join(tmp_folder, "autoarima_model" + str(uuid.uuid4()))
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
            loggerinfo(logger, "Arima No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs

    def _clean_tmp_folder(self, logger, tmp_folder):
        try:
            shutil.rmtree(tmp_folder)
            loggerinfo(logger, "Arima cleaned up temporary file folder.")
        except:
            loggerwarning(logger, "Arima could not delete the temporary file folder.")

    def _create_tmp_folder(self, logger):
        # Create a temp folder to store files used during multi processing experiment
        # This temp folder will be removed at the end of the process
        # Set the default value without context available (required to pass acceptance test
        tmp_folder = os.path.join(user_dir(), "%s_arima_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_arima_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "Arima was denied temp folder creation rights")
            tmp_folder = os.path.join(user_dir(), "%s_arima_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "Arima temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_arima_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(user_dir(), "%s_arima_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "Arima temp folder {}".format(tmp_folder))
        return tmp_folder

    def fit(self, X: dt.Frame, y: np.array = None, **kwargs):
        """
        Fits ARIMA models (1 per time group) using historical target values contained in y
        Model fitting is distributed over a pool of processes and uses file storage to share the data with workers
        :param X: Datatable frame containing the features
        :param y: numpy array containing the historical values of the target
        :return: self
        """
        # Get the logger if it exists
        logger = None
        tmp_folder = str(uuid.uuid4()) + "_arima_folder/"
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Import the ARIMA python module
        pm = importlib.import_module('pmdarima')
        # Init models
        self.models = {}
        # Convert to pandas
        X = X.to_pandas()
        XX = X[self.tgc].copy()
        XX['y'] = np.array(y)
        self.nan_value = np.mean(y)
        self.ntrain = X.shape[0]

        # Group the input by TGC (Time group column) excluding the time column itself
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]

        # Prepare for multi processing
        num_tasks = len(XX_grp)

        def processor(out, res):
            out[res[0]] = res[1]

        pool_to_use = small_job_pool
        loggerinfo(logger, "Arima will use {} workers for parallel processing".format(n_jobs))
        pool = pool_to_use(
            logger=None, processor=processor,
            num_tasks=num_tasks, max_workers=n_jobs
        )

        # Build 1 ARIMA model per time group columns
        nb_groups = len(XX_grp)
        for _i_g, (key, X) in enumerate(XX_grp):
            # Just say where we are in the fitting process
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                loggerinfo(logger, "Auto ARIMA : %d%% of groups fitted" % (100 * (_i_g + 1) // nb_groups))

            X_path = os.path.join(tmp_folder, "autoarima_X" + str(uuid.uuid4()))
            X = X.reset_index(drop=True)
            save_obj(X, X_path)
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            args = (X_path, grp_hash, self.time_column, tmp_folder)
            kwargs = {}
            pool.submit_tryget(None, MyParallelAutoArimaTransformer_fit_async, args=args, kwargs=kwargs,
                               out=self.models)
        pool.finish()

        for k, v in self.models.items():
            self.models[k] = load_obj(v) if v is not None else None
            remove(v)

        self._clean_tmp_folder(logger, tmp_folder)

        return self

    @staticmethod
    def _transform_async(model_path, X_path, nan_value, has_is_train_attr, time_column, pred_gap, tmp_folder):
        """
        Predicts target for a particular time group
        :param model_path: path to the stored model
        :param X_path: Path to the data used to fit the ARIMA model
        :param nan_value: Value of target prior, used when no fitted model has been found
        :param has_is_train_attr: indicates if we predict in-sample or out-of-sample
        :param time_column: Name of the time column in the input data
        :return: self
        """
        model = load_obj(model_path)
        XX_path = os.path.join(tmp_folder, "autoarima_XXt" + str(uuid.uuid4()))
        X = load_obj(X_path)
        # Arima returns the predictions ordered by time
        # So we should keep track of the time order for each group so that
        # predictions are ordered the same as the imput frame
        # Keep track of the order

        order = np.argsort(X[time_column])
        if model is not None:
            if has_is_train_attr:
                yhat = model.predict_in_sample()
            else:
                yhat = model.predict(n_periods=pred_gap + X.shape[0])
                yhat = yhat[pred_gap:]
            XX = pd.DataFrame(yhat, columns=['yhat'])

        else:
            XX = pd.DataFrame(np.full((X.shape[0], 1), nan_value), columns=['yhat'])  # invalid model

        # Sync index
        XX.index = X.index
        assert XX.shape[1] == 1
        save_obj(XX, XX_path)
        remove(model_path)  # indicates success, no longer need
        remove(X_path)  # indicates success, no longer need
        return XX_path

    def transform(self, X: dt.Frame, **kwargs):
        """
        Uses fitted models (1 per time group) to predict the target
        If self.is_train exists, it means we are doing in-sample predictions
        if it does not then we Arima is used to predict the future
        :param X: Datatable Frame containing the features
        :return: ARIMA predictions
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

        X = X.to_pandas()
        XX = X[self.tgc].copy()
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]

        assert len(XX_grp) > 0
        num_tasks = len(XX_grp)

        def processor(out, res):
            out.append(res)

        pool_to_use = small_job_pool
        loggerinfo(logger, "Arima will use {} workers for transform".format(n_jobs))
        pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks, max_workers=n_jobs)

        XX_paths = []
        model_paths = []
        nb_groups = len(XX_grp)
        for _i_g, (key, X) in enumerate(XX_grp):
            # Just print where we are in the process of fitting models
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                loggerinfo(logger, "Auto ARIMA : %d%% of groups transformed" % (100 * (_i_g + 1) // nb_groups))

            # Create time group key to store and retrieve fitted models
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            # Create file path to store data and pass it to the fitting pool
            X_path = os.path.join(tmp_folder, "autoarima_Xt" + str(uuid.uuid4()))

            # Commented for performance, uncomment for debug
            # print("ARIMA - transforming data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            if grp_hash in self.models:
                model = self.models[grp_hash]
                model_path = os.path.join(tmp_folder, "autoarima_modelt" + str(uuid.uuid4()))
                save_obj(model, model_path)
                save_obj(X, X_path)
                model_paths.append(model_path)

                args = (
                    model_path, X_path,
                    self.nan_value, hasattr(self, 'is_train'),
                    self.time_column, self.pred_gap, tmp_folder
                )
                kwargs = {}
                pool.submit_tryget(None, MyParallelAutoArimaTransformer_transform_async, args=args, kwargs=kwargs,
                                   out=XX_paths)
            else:
                # Don't go through pools
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups
                # Sync indices
                XX.index = X.index
                save_obj(XX, X_path)
                XX_paths.append(X_path)
        pool.finish()
        XX = pd.concat((load_obj(XX_path) for XX_path in XX_paths), axis=0).sort_index()
        for p in XX_paths + model_paths:
            remove(p)

        self._clean_tmp_folder(logger, tmp_folder)

        return XX

    def fit_transform(self, X: dt.Frame, y: np.array = None, **kwargs):
        """
        Fits the ARIMA models (1 per time group) and outputs the corresponding predictions
        :param X: Datatable Frame
        :param y: Target to be used to fit the ARIMA model and perdict in-sample
        :return: in-sample ARIMA predictions
        """
        self.is_train = True
        ret = self.fit(X, y, **kwargs).transform(X, **kwargs)
        del self.is_train
        return ret

    def update_history(self, X: dt.Frame, y: np.array = None):
        """
        Update the model fit with additional observed endog/exog values.
        Updating an ARIMA adds new observations to the model, updating the MLE of the parameters
        accordingly by performing several new iterations (maxiter) from the existing model parameters.
        :param X: Datatable Frame containing input features
        :param y: Numpy array containing new observations to update the ARIMA model
        :return: self
        """
        print("auto arima - update history")
        X = X.to_pandas()
        XX = X[self.tgc].copy()
        XX['y'] = np.array(y)
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]
        for key, X in XX_grp:
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            print("auto arima - update history with data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            order = np.argsort(X[self.time_column])
            if grp_hash in self.models:
                model = self.models[grp_hash]
                if model is not None:
                    model.update(X['y'].values[order])
        return self
