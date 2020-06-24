"""Prophet by Facebook for TimeSeries with an example of parameter mutation."""

"""
This recipe is disabled, please use the multi-task version named fb_prophet_parallel.py
that is faster
"""
import importlib
import datatable as dt
import numpy as np
from h2oaicore.models import CustomTimeSeriesModel
from h2oaicore.systemutils import config, arch_type, physical_cores_count
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
import os
import pandas as pd


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


class FBProphetModel(CustomTimeSeriesModel):
    _regression = True
    _binary = False
    _multiclass = False
    _display_name = "FB_Prophet"
    _description = "Facebook Prophet TimeSeries forecasting"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def is_enabled():
        # Please use Prophet recipe in parallel mode : fb_prophet_parallel.py
        return False

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return False  # by default too slow unless only enabled

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
                      accuracy=10,
                      **kwargs):

        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
        loggerinfo(logger, "Mutate is called")

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

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        # Get TGC and time column
        self.tgc = self.params_base.get('tgc')
        self.time_column = self.params_base.get('time_column')
        nan_value = np.mean(y)
        cap = np.max(y) * 1.5  # TODO Don't like this we should compute a cap from average yearly growth
        prior = np.mean(y)

        if self.time_column is None:
            self.time_column = self.tgc[0]

        # Example use of logger, with required import of:
        #  from h2oaicore.systemutils import make_experiment_logger, loggerinfo
        # Can use loggerwarning, loggererror, etc. for different levels
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
        loggerinfo(logger, "Start Fitting Prophet with params : {}".format(self.params))

        # Reduce to TimeGroupColumns
        if isinstance(X, dt.Frame):
            # Convert to pandas
            XX = X[:, self.tgc].to_pandas()
        else:
            XX = X[:, self.tgc].copy()

        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
        XX['y'] = np.array(y)

        # Compute groups
        # Group the input by TGC (Time group column) excluding the time column itself
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]

        # Go Through groups
        priors = {}
        models = {}

        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")

        # Fit 1 FB Prophet model per time group columns
        nb_groups = len(XX_grp)
        for _i_g, (key, X) in enumerate(XX_grp):
            # Just log where we are in the fitting process
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                loggerinfo(logger, "FB Prophet Model : %d%% of groups fitted" % (100 * (_i_g + 1) // nb_groups))

            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))

            # Fit current model and prior
            model = Prophet(growth=self.params["growth"])
            # Add params
            if self.params["country_holidays"] is not None:
                model.add_country_holidays(country_name=self.params["country_holidays"])
            if self.params["monthly_seasonality"]:
                model.add_seasonality(name='monthly', period=30.5, fourier_order=self.params["monthly_seasonality"])
            if self.params["quarterly_seasonality"]:
                model.add_seasonality(name='quarterly', period=92, fourier_order=self.params["quarterly_seasonality"])

            with suppress_stdout_stderr():
                if X.shape[0] < 20:
                    model = None
                else:
                    if self.params["growth"] == "logistic":
                        X["cap"] = cap
                        model.fit(X[['ds', 'y', 'cap']])
                    else:
                        model.fit(X[['ds', 'y']])

            models[grp_hash] = model
            priors[grp_hash] = X['y'].mean()

        self.set_model_properties(
            model={
                'models': models,
                'cap': cap,
                'priors': priors,
                'prior': prior
            },
            features=self.tgc,  # Prophet uses time and timegroups
            importances=np.ones(len(self.tgc)),
            iterations=-1  # Does not have iterations
        )

    def predict(self, X, **kwargs):

        model_config, _, _, _ = self.get_model_properties()

        models = model_config['models']
        cap = model_config['cap']
        priors = model_config['priors']
        prior = model_config['prior']

        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            return np.ones(X.shape[0]) * self.nan_value

        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
        loggerinfo(logger, "Start Predicting with Prophet")

        # Reduce to TimeGroupColumns
        if isinstance(X, dt.Frame):
            # Convert to pandas
            XX = X[:, self.tgc].to_pandas()
        else:
            XX = X[:, self.tgc].copy()

        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)

        if self.params["growth"] == "logistic":
            XX["cap"] = cap

        # Compute groups
        # Group the input by TGC (Time group column) excluding the time column itself
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]

        # Go Through groups and predict
        #
        nb_groups = len(XX_grp)
        preds = []
        for _i_g, (key, X) in enumerate(XX_grp):
            # Just log where we are in the fitting process
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                loggerinfo(logger, "FB Prophet Model : %d%% of groups transformed" % (100 * (_i_g + 1) // nb_groups))

            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))

            # Facebook Prophet returns the predictions ordered by time
            # So we should keep track of the time order for each group so that
            # predictions are ordered the same as the imput frame
            # Keep track of the order
            order = np.argsort(pd.to_datetime(X["ds"]))
            if grp_hash in models.keys():
                model = models[grp_hash]
                if model is not None:
                    # Run prophet
                    yhat = model.predict(X)
                    XX = yhat
                else:
                    if grp_hash in priors.keys():
                        XX = pd.DataFrame(np.full((X.shape[0], 1), priors[grp_hash]), columns=['yhat'])
                    else:
                        # This should not happen
                        loggerinfo(logger, "Group in models but not in priors")
                        XX = pd.DataFrame(np.full((X.shape[0], 1), prior), columns=['yhat'])
            else:
                # print("No Group")
                XX = pd.DataFrame(np.full((X.shape[0], 1), prior), columns=['yhat'])  # unseen groups

            # Reorder the index like prophet re-ordered the predictions
            XX.index = X.index[order]
            # print("Transformed Output for Group")
            # print(XX.sort_index().head(20), flush=True)
            preds.append(XX[['yhat']])

        XX = pd.concat(tuple(preds), axis=0).sort_index()

        return XX['yhat'].values
