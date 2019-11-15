"""Parallel FB Prophet transformer is a time series transformer that predicts target using FBProphet models.
In this implementation, Time Group Models are fitted in parallel"""
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
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


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
    def do_acceptance_test():
        return True

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
            if config.fixed_num_folds == 0:
                n_jobs = max(1, int(int(max_threads() / min(config.num_folds, kwargs['max_workers']))))
            else:
                n_jobs = max(1, int(
                    int(max_threads() / min(config.fixed_num_folds, config.num_folds, kwargs['max_workers']))))
        except KeyError:
            loggerinfo(logger, "Prophet No Max Worker in kwargs. Set n_jobs to 1")
            n_jobs = 1

        return n_jobs if n_jobs > 1 else 2

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
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Convert to pandas
        XX = X[:, self.tgc].to_pandas()
        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
        # Make sure labales are numeric
        if self.labels is not None:
            y = LabelEncoder().fit(self.labels).transform(y)
        XX['y'] = np.array(y)
        # Set target prior
        self.nan_value = np.mean(y)

        # Group the input by TGC (Time group column) excluding the time column itself
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]
        self.models = {}
        self.priors = {}

        # Prepare for multi processing
        num_tasks = len(XX_grp)

        def processor(out, res):
            out[res[0]] = res[1]

        pool_to_use = small_job_pool
        loggerinfo(logger, f"Prophet will use {n_jobs} workers for fitting.")
        loggerinfo(logger, "Prophet parameters holidays {} / monthly {}".format(self.country_holidays, self.monthly_seasonality))
        pool = pool_to_use(
            logger=None, processor=processor,
            num_tasks=num_tasks, max_workers=n_jobs
        )

        # Fit 1 FB Prophet model per time group columns
        nb_groups = len(XX_grp)
        for _i_g, (key, X) in enumerate(XX_grp):
            # Just log where we are in the fitting process
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                loggerinfo(logger, "FB Prophet : %d%% of groups fitted" % (100 * (_i_g + 1) // nb_groups))

            X_path = os.path.join(tmp_folder, "fbprophet_X" + str(uuid.uuid4()))
            X = X.reset_index(drop=True)
            save_obj(X, X_path)
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))

            self.priors[grp_hash] = X['y'].mean()

            params = {
                "country_holidays": self.country_holidays,
                "monthly_seasonality": self.monthly_seasonality
            }

            args = (X_path, grp_hash, tmp_folder, params)
            kwargs = {}
            pool.submit_tryget(None, MyParallelProphetTransformer_fit_async, args=args, kwargs=kwargs, out=self.models)
        pool.finish()
        for k, v in self.models.items():
            self.models[k] = load_obj(v) if v is not None else None
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
        XX_path = os.path.join(tmp_folder, "fbprophet_XXt" + str(uuid.uuid4()))
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

        XX = X[:, self.tgc].to_pandas()
        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
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
        loggerinfo(logger, "Prophet will use {} workers for transform".format(n_jobs))
        pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks, max_workers=n_jobs)
        XX_paths = []
        model_paths = []
        nb_groups = len(XX_grp)
        print("Nb Groups = ", nb_groups)
        for _i_g, (key, X) in enumerate(XX_grp):
            # Log where we are in the transformation of the dataset
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                loggerinfo(logger, "FB Prophet : %d%% of groups transformed" % (100 * (_i_g + 1) // nb_groups))

            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            X_path = os.path.join(tmp_folder, "fbprophet_Xt" + str(uuid.uuid4()))
            # Commented for performance, uncomment for debug
            # print("prophet - transforming data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            if grp_hash in self.models:
                model = self.models[grp_hash]
                model_path = os.path.join(tmp_folder, "fbprophet_modelt" + str(uuid.uuid4()))
                save_obj(model, model_path)
                save_obj(X, X_path)
                model_paths.append(model_path)

                args = (model_path, X_path, self.priors[grp_hash], tmp_folder)
                kwargs = {}
                pool.submit_tryget(None, MyParallelProphetTransformer_transform_async, args=args, kwargs=kwargs,
                                   out=XX_paths)
            else:
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups
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
        Fits the FB Prophet models (1 per time group) and outputs the corresponding predictions
        :param X: Datatable Frame
        :param y: Target to be used to fit FB Prophet models
        :return: FB Prophet predictions
        """
        return self.fit(X, y, **kwargs).transform(X, **kwargs)
