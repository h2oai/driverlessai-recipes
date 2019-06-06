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
    _included_transformers = ['OriginalTransformer', 'CatOriginalTransformer',
                             'DateOriginalTransformer', 'DateTimeOriginalTransformer']

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def is_enabled():
        return False

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        self.tgc = self.params_base.get('tgc')
        self.time_column = self.params_base.get('time_column')
        self.encoder = self.params_base.get('encoder')
        self.nan_value = y.mean()
        self.means = {}
        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            return
        XX = X[:, self.tgc].to_pandas()
        XX['y'] = np.array(y)
        self.nan_value = np.mean(y)
        self.ntrain = X.shape[0]
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]
        for key, X in XX_grp:
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            self.means[grp_hash] = X['y'].mean()

    def predict(self, X, **kwargs):
        if self.tgc is None or not all([x in X.names for x in self.tgc]):
            return np.ones(X.shape[0]) * self.nan_value
        XX = X[:, self.tgc].to_pandas()
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]
        preds = []
        for key, X in XX_grp:
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            if grp_hash in self.means:
                mean = self.means[grp_hash]
                if mean is not None:
                    yhat = np.ones(X.shape[0]) * mean
                    XX = pd.DataFrame(yhat, columns=['yhat'])
                else:
                    XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # invalid model
            else:
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups
            XX.index = X.index
            preds.append(XX)
        XX = pd.concat(tuple(preds), axis=0).sort_index()
        return XX.values
