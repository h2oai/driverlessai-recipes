"""Auto ARIMA transformer is a time series transformer that predicts target using ARIMA models"""

# For more information about the python ARIMA package
# please visit https://www.alkaline-ml.com/pmdarima/index.html

import importlib
import numpy as np
import pandas as pd
import datatable as dt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer


class MyAutoArimaTransformer(CustomTimeSeriesTransformer):
    _binary = False
    _multiclass = False
    _modules_needed_by_name = ['pmdarima==1.5']
    _included_model_classes = None
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _lag_recipe_allowed = True
    _causal_recipe_allowed = False

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        return False  # by default auto arima is too slow, but if the only model selected this will still allow use

    def fit(self, X: dt.Frame, y: np.array = None):
        """
        Fits ARIMA models (1 per time group) using historical target values contained in y
        :param X: Datatable frame containing the features
        :param y: numpy array containing the historical values of the target
        :return: self
        """
        # Import the ARIMA python module
        pm = importlib.import_module('pmdarima')

        self.scalers = None

        logger = self._get_experiment_logger()

        # 0. Preliminary steps
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        X = X[:, self.tgc].to_pandas()

        # Fill NaNs or None
        X = X.replace([None, np.nan], 0)

        # Add target, Label encoder is only used for Classif. which we don't support...
        if self.labels is not None:
            y = LabelEncoder().fit(self.labels).transform(y)
        X['y'] = np.array(y)

        # 0. Fit general scaler to make predictions for unknown groups
        X.rename(columns={self.time_column: "ds"}, inplace=True)
        self.general_scaler = MinMaxScaler(feature_range=(1, 2)).fit(X[['y', 'ds']].groupby('ds').median().values)

        # 1. Scale target for each individual group
        # Go through groups and standard scale them
        X['y_skl'] = self.scale_target_per_time_group(X, tgc_wo_time, logger)

        # 2. Make time a pandas datetime series so that we can order it
        X['ds'] = pd.to_datetime(X['ds'], format=self.datetime_formats[self.time_column])

        # 3. Fit a model on averages
        X_avg = X[['ds', 'y_skl']].groupby('ds').mean().reset_index()
        order = np.argsort(X_avg['ds'])
        try:
            self.avg_model = pm.auto_arima(X_avg['y_skl'].values[order], error_action='ignore', seasonal=False)
        except Exception as e:
            loggerinfo(logger, "ARIMA: Average model error : {}".format(e))
            self.avg_model = None

        # 4. Fit model for Average Groups
        self.models = {}
        # Go through groups
        for grp_col in tgc_wo_time:
            print(f'fitting {grp_col}')
            # Get the unique dates to be predicted
            X_groups = X[['ds', 'y_skl', grp_col]].groupby(grp_col)
            print(X.shape)

            nb_groups = len(X_groups)
            for _i_g, (key, X_grp) in enumerate(X_groups):
                # Just say where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(logger, "Auto ARIMA : %d%% of groups fitted" % (100 * (_i_g + 1) // nb_groups))

                # Average over dates
                X_grp = X_grp.groupby('ds')['y_skl'].mean().reset_index()

                grp_hash = self.get_hash(grp_col, key)
                # print("auto arima - fitting on data of shape: %s for group: %s" % (str(X.shape), grp_hash))

                X_grp['ds'] = pd.to_datetime(X_grp['ds'], format=self.datetime_formats[self.time_column])
                order = np.argsort(X_grp['ds'])

                try:
                    model = pm.auto_arima(X_grp['y_skl'].values[order], error_action='ignore', seasonal=False)
                except Exception as e:
                    loggerinfo(logger, "Auto ARIMA warning: {}".format(e))
                    model = None

                self.models[grp_hash] = model

        return self

    def get_hash(self, col='', key=None):
        # Create dict key to store the min max scaler
        if isinstance(key, tuple):
            key = [col] + list(key)
        elif isinstance(key, list):
            pass
        else:
            # Not tuple, not list
            key = [col, key]

        grp_hash = '_'.join(map(str, key))
        return grp_hash

    def scale_target_per_time_group(self, X, tgc_wo_time, logger):
        loggerinfo(logger, 'Start of group scaling')
        if len(tgc_wo_time) > 0:
            X_groups = X.groupby(tgc_wo_time)
        else:
            X_groups = [([None], X)]

        if self.scalers is None:
            self.scalers = {}

            scaled_ys = []
            for key, X_grp in X_groups:
                # Create dict key to store the min max scaler
                grp_hash = self.get_hash(key)
                # Scale target for current group
                self.scalers[grp_hash] = MinMaxScaler(feature_range=(1, 2))
                y_skl = self.scalers[grp_hash].fit_transform(X_grp[['y']].values)
                # Put back in a DataFrame to keep track of original index
                y_skl_df = pd.DataFrame(y_skl, columns=['y'])
                # (0, 'A') (1, 4) (100, 1) (100, 1)
                # print(grp_hash, X_grp.shape, y_skl.shape, y_skl_df.shape)

                y_skl_df.index = X_grp.index
                scaled_ys.append(y_skl_df)
        else:
            scaled_ys = []
            for key, X_grp in X_groups:
                # Create dict key to store the min max scaler
                grp_hash = self.get_hash(key)
                # Scale target for current group
                y_skl = self.scalers[grp_hash].transform(X_grp[['y']].values)
                # Put back in a DataFrame to keep track of original index
                y_skl_df = pd.DataFrame(y_skl, columns=['y'])
                # (0, 'A') (1, 4) (100, 1) (100, 1)
                # print(grp_hash, X_grp.shape, y_skl.shape, y_skl_df.shape)

                y_skl_df.index = X_grp.index
                scaled_ys.append(y_skl_df)
        loggerinfo(logger, 'End of group scaling')

        return pd.concat(tuple(scaled_ys), axis=0)

    def transform(self, X: dt.Frame):
        """
        Uses fitted models (1 per time group) to predict the target
        If self.is_train exists, it means we are doing in-sample predictions
        if it does not then we Arima is used to predict the future
        :param X: Datatable Frame containing the features
        :return: ARIMA predictions
        """
        logger = self._get_experiment_logger()

        # 0. Preliminary steps
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        X = X[:, self.tgc].to_pandas()

        # Fill NaNs or None
        X = X.replace([None, np.nan], 0)

        X.rename(columns={self.time_column: "ds"}, inplace=True)
        X['ds'] = pd.to_datetime(X['ds'], format=self.datetime_formats[self.time_column])

        # 1. Predict with average model
        if self.avg_model is not None:
            X_time = X[['ds']].groupby('ds').first().reset_index()
            if hasattr(self, 'is_train'):
                yhat = self.avg_model.predict_in_sample()
            else:
                yhat = self.avg_model.predict(n_periods=self.pred_gap + X_time.shape[0])
                # Assign predictions the same order the dates had
                yhat = yhat[self.pred_gap:]

            X_time.sort_values('ds', inplace=True)
            X_time['yhat'] = yhat
            X_time.sort_index(inplace=True)
            # Merge back the average prediction to all similar timestamps
            indices = X.index
            X = pd.merge(
                left=X,
                right=X_time[['ds', 'yhat']],
                on='ds',
                how='left'
            )
            X.index = indices
        else:
            X['yhat'] = np.nan

        y_avg_model = X['yhat'].values
        y_predictions = pd.DataFrame(y_avg_model, columns=['average_pred'])

        # 2. Predict for individual group
        # Go through groups
        for i_tgc, grp_col in enumerate(tgc_wo_time):
            y_hat_tgc = np.zeros(X.shape[0])

            # Get the unique dates to be predicted
            X_groups = X[['ds', grp_col]].groupby(grp_col)

            nb_groups = len(X_groups)
            dfs = []
            for _i_g, (key, X_grp) in enumerate(X_groups):
                # Just say where we are in the fitting process
                if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                    loggerinfo(logger, "Auto ARIMA : %d%% of groups transformed" % (100 * (_i_g + 1) // nb_groups))

                grp_hash = self.get_hash(grp_col, key)
                model = self.models[grp_hash]

                # Find unique datetime
                X_time = X_grp [['ds']].groupby('ds').first().reset_index()
                X_time['ds'] = pd.to_datetime(X_time['ds'], format=self.datetime_formats[self.time_column])
                X_time = X_time.sort_values('ds')

                if model is not None:
                    # Get predictions from ARIMA model, make sure we include prediction gaps
                    if hasattr(self, 'is_train'):
                        print(X_grp.shape, model.predict_in_sample().shape)
                        # It can happen that in_sample predictions are smaller than the training set used
                        pred = model.predict_in_sample()
                        tmp = np.zeros(X_time.shape[0])
                        tmp[:len(pred)] = pred
                        X_time['yhat'] = tmp
                    else:
                        # In ARIMA, you provide the number of periods you predict on
                        # So you have to
                        yhat = model.predict(n_periods=self.pred_gap + X_time.shape[0])
                        X_time['yhat'] = yhat[self.pred_gap:]

                    # Now merge back the predictions into X_grp
                    indices = X_grp.index
                    X_grp = pd.merge(
                        left=X_grp,
                        right=X_time[['ds', 'yhat']],
                        on='ds',
                        how='left'
                    )
                    X_grp.index = indices
                else:
                    X_grp = X_grp.copy()
                    X_grp['yhat'] = np.nan

                dfs.append(X_grp['yhat'])

            y_predictions[f'{grp_col}_pred'] = pd.concat(dfs, axis=0)

        # Now we have to invert scale all this
        for grp_col in tgc_wo_time:
            # Add time group to the predictions, will be used to invert scaling
            y_predictions[grp_col] = X[grp_col].copy()
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

        y_predictions.drop(tgc_wo_time, axis=1, inplace=True)

        self._output_feature_names = [f'{self.display_name}_{_f}' for _f in y_predictions]
        self._feature_desc = [f'{self.display_name}_{_f}' for _f in y_predictions]

        return y_predictions

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        """
        Fits the ARIMA models (1 per time group) and outputs the corresponding predictions
        :param X: Datatable Frame
        :param y: Target to be used to fit the ARIMA model and perdict in-sample
        :return: in-sample ARIMA predictions
        """

        # Flag the fact we are doing in-sample predictions
        self.is_train = True
        ret = self.fit(X, y).transform(X)
        del self.is_train
        return ret

    def update_history(self, X: dt.Frame, y: np.array = None):
        """
        Update the model fit with additional observed endog/exog values.
        Updating an ARIMA adds new observations to the model, updating the MLE of the parameters
        accordingly by performing several new iterations (maxiter) from the existing model parameters.
        :param X: Datatable Frame containing input features
        :param y: Numpy array containing new observations to update the ARIMA model
        :return:
        """
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
            # print("auto arima - update history with data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            order = np.argsort(X[self.time_column])
            if grp_hash in self.models:
                model = self.models[grp_hash]
                if model is not None:
                    model.update(X['y'].values[order])
        return self

    def _get_experiment_logger(self):
        # Get the logger if it exists
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )

        return logger

