import importlib
from h2oaicore.systemutils import small_job_pool, save_obj, load_obj, temporary_files_path, remove
from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
import datatable as dt
import numpy as np
import pandas as pd
import random
import os
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


def MyParallelAutoArimaTransformer_fit_async(*args, **kwargs):
    return MyParallelAutoArimaTransformer._fit_async(*args, **kwargs)


def MyParallelAutoArimaTransformer_transform_async(*args, **kwargs):
    return MyParallelAutoArimaTransformer._transform_async(*args, **kwargs)


class MyParallelAutoArimaTransformer(CustomTimeSeriesTransformer):
    _binary = False
    _multiclass = False
    _modules_needed_by_name = ['pmdarima']
    _included_boosters = None

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def _fit_async(X_path, grp_hash, order):
        np.random.seed(1234)
        random.seed(1234)
        X = load_obj(X_path)

        pm = importlib.import_module('pmdarima')
        with suppress_stdout_stderr():
            model = pm.auto_arima(X['y'].values[order], error_action='ignore')

        model_path = os.path.join(temporary_files_path, "autoarima_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(X_path)  # remove to indicate success
        return grp_hash, model_path

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

        num_tasks = len(XX_grp)

        def processor(out, res):
            out[res[0]] = res[1]

        pool_to_use = small_job_pool
        pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks)
        for key, X in XX_grp:
            X_path = os.path.join(temporary_files_path, "autoarima_X" + str(uuid.uuid4()))
            order = np.argsort(X[self.time_column])
            X = X.reset_index(drop=True)
            save_obj(X, X_path)
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            args = (X_path, grp_hash, order,)
            kwargs = {}
            pool.submit_tryget(None, MyParallelAutoArimaTransformer_fit_async, args=args, kwargs=kwargs, out=self.models)
        pool.finish()

        for k, v in self.models.items():
            self.models[k] = load_obj(v) if v is not None else None
            remove(v)
        return self

    @staticmethod
    def _transform_async(model_path, X_path, nan_value, has_is_train_attr, order):
        model = load_obj(model_path)
        XX_path = os.path.join(temporary_files_path, "autoarima_XXt" + str(uuid.uuid4()))
        X = load_obj(X_path)
        # Facebook Prophet returns the predictions ordered by time
        # So we should keep track of the time order for each group so that
        # predictions are ordered the same as the imput frame
        # Keep track of the order

        if model is not None:
            yhat = model.predict_in_sample() \
                if has_is_train_attr else model.predict(n_periods=X.shape[0])
            yhat = yhat[order]
            XX = pd.DataFrame(yhat, columns=['yhat'])
        else:
            XX = pd.DataFrame(np.full((X.shape[0], 1), nan_value), columns=['yhat'])  # invalid model

        assert XX.shape[1] == 1
        save_obj(XX, XX_path)
        remove(model_path)  # indicates success, no longer need
        remove(X_path)  # indicates success, no longer need
        return XX_path

    def transform(self, X: dt.Frame):
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
        pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks)
        XX_paths = []
        model_paths = []
        for key, X in XX_grp:
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            X_path = os.path.join(temporary_files_path, "autoarima_Xt" + str(uuid.uuid4()))
            order = np.argsort(X[self.time_column])
            # Commented for performance, uncomment for debug
            # print("prophet - transforming data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            if grp_hash in self.models:
                model = self.models[grp_hash]
                model_path = os.path.join(temporary_files_path, "autoarima_modelt" + str(uuid.uuid4()))
                save_obj(model, model_path)
                save_obj(X, X_path)
                model_paths.append(model_path)

                args = (model_path, X_path, self.nan_value, hasattr(self, 'is_train'), order,)
                kwargs = {}
                pool.submit_tryget(None, MyParallelAutoArimaTransformer_transform_async, args=args, kwargs=kwargs,
                                   out=XX_paths)
            else:
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups
                save_obj(XX, X_path)
                XX_paths.append(X_path)
        pool.finish()
        XX = pd.concat((load_obj(XX_path) for XX_path in XX_paths), axis=0).sort_index()
        for p in XX_paths + model_paths:
            remove(p)
        print(XX, flush=True)
        return XX

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.is_train = True
        ret = self.fit(X, y).transform(X)
        del self.is_train
        return ret
