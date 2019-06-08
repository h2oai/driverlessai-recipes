import importlib

from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np
import pandas as pd


class MyAutoArimaTransformer(CustomTimeSeriesTransformer):
    _binary = False
    _multiclass = False
    _modules_needed_by_name = ['pmdarima']
    _included_boosters = None

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    def fit(self, X: dt.Frame, y: np.array = None):
        pm = importlib.import_module('pmdarima')
        self.models = {}
        X = X.to_pandas()
        XX = X[self.tgc].copy()
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
            # print("auto arima - fitting on data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            order = np.argsort(X[self.time_column])
            try:
                model = pm.auto_arima(X['y'].values[order], error_action='ignore')
            except:
                model = None
            self.models[grp_hash] = model
        return self

    def transform(self, X: dt.Frame):
        X = X.to_pandas()
        XX = X[self.tgc].copy()
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]
        preds = []
        for key, X in XX_grp:
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            # print("auto arima - transforming data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            order = np.argsort(X[self.time_column])
            if grp_hash in self.models:
                model = self.models[grp_hash]
                if model is not None:
                    yhat = model.predict_in_sample() \
                        if hasattr(self, 'is_train') else model.predict(n_periods=X.shape[0])
                    yhat = yhat[order]
                    XX = pd.DataFrame(yhat, columns=['yhat'])
                else:
                    XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # invalid model
            else:
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups
            XX.index = X.index
            preds.append(XX)
        XX = pd.concat(tuple(preds), axis=0).sort_index()

        return XX

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.is_train = True
        ret = self.fit(X, y).transform(X)
        del self.is_train
        return ret
        
    def update_history(self, X: dt.Frame, y: np.array = None):
        X = X.to_pandas()
        XX = X[self.tgc].copy
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
