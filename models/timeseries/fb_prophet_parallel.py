"""Prophet by Facebook for TimeSeries with an example of parameter mutation."""
import importlib
import datatable as dt
import numpy as np
from h2oaicore.models import CustomTimeSeriesModel
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdebug
from h2oaicore.systemutils import (
    arch_type, small_job_pool, save_obj, load_obj, temporary_files_path, remove, max_threads, config
)
import os
import pandas as pd
import shutil
import random
import uuid


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
    return FBProphetParallelModel._fit_async(*args, **kwargs)


def MyParallelProphetTransformer_transform_async(*args, **kwargs):
    return FBProphetParallelModel._transform_async(*args, **kwargs)


class FBProphetParallelModel(CustomTimeSeriesModel):
    _regression = True
    _binary = False
    _multiclass = False
    _display_name = "FB_Prophet_Parallel"
    _description = "Facebook Prophet TimeSeries forecasting with multi process support"
    _parallel_task = True

    @staticmethod
    def is_enabled():
        return not (arch_type == "ppc64le")

    @staticmethod
    def do_acceptance_test():
        return False

    _modules_needed_by_name = ['convertdate', 'pystan==2.18', 'fbprophet==0.4.post2']

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):

        """
        Parameters available for the model :
          - growth : available market growth strategy in Prophet are linear and logistic
            logistic growth require a cap that saturates the predictions output
            See : https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth

          - country_holidays : allows Prophet to use built in Holidays
            See mutate_params to check the available countries in the model
            https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#built-in-country-holidays

          We can change the way seasonality affects the predictions
          - seasonality_mode : 'additive' (default) or 'multiplicative'

          We can override Fourier Order for seasonality calculation
          https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#fourier-order-for-seasonalities
          - weekly_seasonality : default is 'auto'
            Can be False or any number that gives the Fourier Order for the seasonality calculation
          - yearly_seasonality : default is 'auto'
            Can be False or any number that gives the Fourier Order for the seasonality calculation

          By default only weekly and yearly seasonality are calculated
          However one can ask Prophet to calculate other/specific seasonality
          https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#specifying-custom-seasonalities
          - monthly_seasonality : Either False (no monthly seasonality) or a number which will be the Fourier Order
            for monthly seasonality.

          - quarterly_seasonality : Either False (no quarterly seasonality) or a number which will be the Fourier Order
            for quarterly seasonality.
        """
        self.params = dict(
            growth=kwargs.get("growth", "linear"),
            seasonality_mode=kwargs.get("seasonality_mode", "additive"),
            country_holidays=kwargs.get("country_holidays", None),
            weekly_seasonality=kwargs.get("weekly_seasonality", 'auto'),
            monthly_seasonality=kwargs.get("monthly_seasonality", False),
            quarterly_seasonality=kwargs.get("quarterly_seasonality", False),
            yearly_seasonality=kwargs.get("yearly_seasonality", 'auto'),
        )

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):

        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)

        # Default version is do no mutation
        # Otherwise, change self.params for this model
        holiday_choice = [None, "US", "UK", "DE", "FRA"]
        if accuracy >= 8:
            weekly_choice = [False, 'auto', 5, 7, 10, 15]
            yearly_choice = [False, 'auto', 5, 10, 15, 20, 30]
            monthly_choice = [False, 3, 5, 7, 10]
            quarterly_choice = [False, 3, 5, 7, 10]
        elif accuracy >= 5:
            weekly_choice = [False, 'auto', 10, 20]
            yearly_choice = [False, 'auto', 10, 20]
            monthly_choice = [False, 5]
            quarterly_choice = [False, 5]
        else:
            # No alternative seasonality, and no seasonality override for weekly and yearly
            weekly_choice = [False, 'auto']
            yearly_choice = [False, 'auto']
            monthly_choice = [False]
            quarterly_choice = [False]

        self.params["country_holidays"] = np.random.choice(holiday_choice)
        self.params["seasonality_mode"] = np.random.choice(["additive", "multiplicative"])
        self.params["weekly_seasonality"] = np.random.choice(weekly_choice)
        self.params["monthly_seasonality"] = np.random.choice(monthly_choice)
        self.params["quarterly_seasonality"] = np.random.choice(quarterly_choice)
        self.params["yearly_seasonality"] = np.random.choice(yearly_choice)
        self.params["growth"] = np.random.choice(["linear", "logistic"])

    @staticmethod
    def _fit_async(X_path, grp_hash, tmp_folder):
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
        model = Prophet()

        with suppress_stdout_stderr():
            model.fit(X[['ds', 'y']])
        model_path = os.path.join(tmp_folder, "fbprophet_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(X_path)  # remove to indicate success
        return grp_hash, model_path

    def _get_n_jobs(self, logger, **kwargs):
        return 4  # self.params_base['n_jobs']

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
        tmp_folder = os.path.join(temporary_files_path, "%s_prophet_model_folder" % uuid.uuid4())
        # Make a real tmp folder when experiment is available
        if self.context and self.context.experiment_id:
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_prophet_model_folder" % uuid.uuid4())

        # Now let's try to create that folder
        try:
            os.mkdir(tmp_folder)
        except PermissionError:
            # This not occur so log a warning
            loggerwarning(logger, "Prophet was denied temp folder creation rights")
            tmp_folder = os.path.join(temporary_files_path, "%s_prophet_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except FileExistsError:
            # We should never be here since temp dir name is expected to be unique
            loggerwarning(logger, "Prophet temp folder already exists")
            tmp_folder = os.path.join(self.context.experiment_tmp_dir, "%s_prophet_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)
        except:
            # Revert to temporary file path
            tmp_folder = os.path.join(temporary_files_path, "%s_prophet_model_folder" % uuid.uuid4())
            os.mkdir(tmp_folder)

        loggerinfo(logger, "Prophet temp folder {}".format(tmp_folder))
        return tmp_folder

    @staticmethod
    def _fit_async(X_path, grp_hash, tmp_folder, params, cap):
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
            print("prophet - small data work-around for group: %s" % grp_hash)
            return grp_hash, None

        # Import FB Prophet package
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")

        # Fit current model and prior
        model = Prophet(growth=params["growth"])
        # Add params
        if params["country_holidays"] is not None:
            model.add_country_holidays(country_name=params["country_holidays"])
        if params["monthly_seasonality"]:
            model.add_seasonality(name='monthly', period=30.5, fourier_order=params["monthly_seasonality"])
        if params["quarterly_seasonality"]:
            model.add_seasonality(name='quarterly', period=92, fourier_order=params["quarterly_seasonality"])

        with suppress_stdout_stderr():
            if params["growth"] == "logistic":
                X["cap"] = cap
                model.fit(X[['ds', 'y', 'cap']])
            else:
                model.fit(X[['ds', 'y']])

        model_path = os.path.join(tmp_folder, "fbprophet_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(X_path)  # remove to indicate success
        return grp_hash, model_path

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        # Get TGC and time column
        self.tgc = self.params_base.get('tgc', None)
        self.time_column = self.params_base.get('time_column', None)
        self.nan_value = np.mean(y)
        self.cap = np.max(y) * 1.5  # TODO Don't like this we should compute a cap from average yearly growth
        self.prior = np.mean(y)

        if self.time_column is None:
            self.time_column = self.tgc[0]

        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )
        loggerinfo(logger, "Start Fitting Prophet Model with params : {}".format(self.params))

        # Get temporary folders for multi process communication
        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        # Convert to pandas
        XX = X[:, self.tgc].to_pandas()
        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
        # Make target available in the Frame
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
        loggerdebug(logger, "Prophet will use {} workers for fitting".format(n_jobs))
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

            args = (X_path, grp_hash, tmp_folder, self.params, self.cap)
            kwargs = {}
            pool.submit_tryget(None, MyParallelProphetTransformer_fit_async, args=args, kwargs=kwargs, out=self.models)
        pool.finish()
        for k, v in self.models.items():
            self.models[k] = load_obj(v) if v is not None else None
            remove(v)

        self._clean_tmp_folder(logger, tmp_folder)

        return None

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

    def predict(self, X: dt.Frame, **kwargs):
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

        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            loggerdebug(logger, "Return 0 predictions")
            return np.ones(X.shape[0]) * self.nan_value

        tmp_folder = self._create_tmp_folder(logger)

        n_jobs = self._get_n_jobs(logger, **kwargs)

        XX = X[:, self.tgc].to_pandas()
        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)

        if self.params["growth"] == "logistic":
            XX["cap"] = self.cap

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
        loggerdebug(logger, "Prophet will use {} workers for transform".format(n_jobs))
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

        return XX['yhat'].values
