"""Transformer that uses FB Prophet for time series prediction.
Please see the parallel implementation for more information"""

import importlib

from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np
import pandas as pd
import gc
import os


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


class MySerialProphetTransformer(CustomTimeSeriesTransformer):
    _binary = False
    _multiclass = False
    # some package dependencies are best sequential to overcome known issues
    _modules_needed_by_name = ['pystan==2.18', 'fbprophet==0.4.post2']
    # _modules_needed_by_name = ['fbprophet']
    _included_boosters = None  # ["gblinear"] for strong trends - can extrapolate

    @staticmethod
    def is_enabled():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    def fit(self, X: dt.Frame, y: np.array = None):
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")
        # from fbprophet import Prophet
        self.models = {}
        XX = X[:, self.tgc].to_pandas()
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
        XX['y'] = np.array(y)
        self.nan_value = np.mean(y)
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]

        nb_groups = len(XX_grp)
        for _i_g, (key, X) in enumerate(XX_grp):
            if (100 * (_i_g + 1) // nb_groups) % 5 == 0:
                print("FB Prophet - ", 100 * (_i_g + 1) // nb_groups, "%% of Groups Fitted")
            model = Prophet()
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            # print("prophet - fitting on data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            if X.shape[0] < 20:
                print("prophet - small data work-around for group: %s" % grp_hash)
                model = None
            else:
                with suppress_stdout_stderr():
                    model.fit(X[['ds', 'y']])
                gc.collect()
            self.models[grp_hash] = model
        return self

    def transform(self, X: dt.Frame):
        XX = X[:, self.tgc].to_pandas()
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
        # print("Input Frame")
        # print(XX.head(20), flush=True)
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]

        preds = []
        nb_groups = len(XX_grp)
        for _i_g, (key, X) in enumerate(XX_grp):
            if (_i_g + 1) % max(1, nb_groups // 20) == 0:
                print("FB Prophet - ", 100 * (_i_g + 1) // nb_groups, "%% of Groups Transformed")
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            # print("prophet - transforming data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            # Facebook Prophet returns the predictions ordered by time
            # So we should keep track of the time order for each group so that
            # predictions are ordered the same as the imput frame
            # Keep track of the order
            order = np.argsort(pd.to_datetime(X["ds"]))
            # print("DateTime To Be transformed")
            # print(X['ds'].head(20), flush=True)
            if grp_hash in self.models:
                model = self.models[grp_hash]
                if model is not None:
                    # Run prophet
                    yhat = model.predict(X)
                    # print("Model Output")
                    # print(yhat.head(20), flush=True)
                    XX = yhat  # pd.DataFrame(yhat['yat'], columns=['yhat'])
                else:
                    # print("No Model")
                    XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # invalid model
            else:
                # print("No Group")
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups

            # Reorder the index like prophet re-ordered the predictions
            XX.index = X.index[order]
            # print("Transformed Output for Group")
            # print(XX.sort_index().head(20), flush=True)
            preds.append(XX[['yhat']])

        XX = pd.concat(tuple(preds), axis=0).sort_index()
        # print("Transformed results")
        # print(XX)

        return XX

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.fit(X, y).transform(X)
