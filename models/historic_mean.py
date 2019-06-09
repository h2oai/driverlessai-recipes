import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.models import CustomTimeSeriesModel


class HistoricMeanModel(CustomTimeSeriesModel):
    _can_handle_non_numeric = True
    _boosters = ['historic_mean']
    _regression = True
    _display_name = "HistoricMean"
    _description = "Historic Mean"

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def is_enabled():
        return True

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        self.tgc = self.params_base.get('tgc')
        self.time_column = self.params_base.get('time_column')
        self.encoder = self.params_base.get('encoder')
        self.nan_value = y.mean()
        self.means = {}
        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            return

        if self.time_column is None:
            self.time_column = self.tgc[0]

        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))

        # Datatable code

        if len(tgc_wo_time) > 0:
            self.nan_value = np.mean(y)
            self.ntrain = X.shape[0]
            X_dt = X.copy()
            X_dt.cbind(dt.Frame({"y": y}))
            self.group_means = X_dt[:, dt.mean(dt.f.y), dt.by(*tgc_wo_time)]
            # Have meaningful column names
            self.group_means.names = tgc_wo_time + ["yhat"]
        else:
            self.group_means = np.mean(y)

        # # Pandas code
        # XX = X[:, self.tgc].to_pandas()
        # XX['y'] = np.array(y)
        # if len(tgc_wo_time) > 0:
        #     self.nan_value = np.mean(y)
        #     self.ntrain = X.shape[0]
        #     self.group_means = XX.groupby(tgc_wo_time)["y"].mean().reset_index()
        #     # Have meaningful column names
        #     self.group_means.columns = tgc_wo_time + ["yhat"]
        # else:
        #     self.group_means = np.mean(y)

    def predict(self, X, **kwargs):
        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            return np.ones(X.shape[0]) * self.nan_value

        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))

        # Datatable code
        if len(tgc_wo_time) > 0:
            # Join the average per group to the input datafrane
            self.group_means.key = tgc_wo_time
            # Predictions for unknown tgc will be None in DT
            yhat_dt = X[:, :, dt.join(self.group_means)][:, "yhat"]
            # In DT missing values after the join are None
            # Need to cast to float64 to replace None or np.nan
            yhat_dt.replace(None, np.float64(self.nan_value))

            return yhat_dt.to_numpy()[:, 0]
        else:
            # if no Groups are avaible then just return the target average
            return np.full((X.shape[0], 1), self.nan_value)

        # # Pandas code
        # XX = X[:, self.tgc].to_pandas()
        # if len(tgc_wo_time) > 0:
        #     # Join the average per group to the input datafrane
        #     return XX[tgc_wo_time].merge(
        #         right=self.group_means,
        #         on=tgc_wo_time,
        #         how='left'
        #     )["yhat"].fillna(self.nan_value).values
        #
        # else:
        #     # if no Groups are avaible then just return the target average
        #     return np.full((X.shape[0], 1), self.nan_value)
