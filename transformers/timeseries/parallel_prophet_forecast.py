from h2oaicore.transformer_utils import CustomTimeSeriesTransformer
from h2oaicore.systemutils import small_job_pool, save_obj, load_obj, temporary_files_path, remove
import datatable as dt
import numpy as np
import os
import uuid
import random
import importlib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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


def MyParallelProphetTransformer_fit_async(*args, **kwargs):
    return MyParallelProphetTransformer._fit_async(*args, **kwargs)


def MyParallelProphetTransformer_transform_async(*args, **kwargs):
    return MyParallelProphetTransformer._transform_async(*args, **kwargs)


class MyParallelProphetTransformer(CustomTimeSeriesTransformer):
    _is_reproducible = False
    _binary = False
    _multiclass = False
    # some package dependencies are best sequential to overcome known issues
    _modules_needed_by_name = ['convertdate', 'pystan==2.18', 'fbprophet==0.4.post2']
    # _modules_needed_by_name = ['fbprophet']
    _allowed_boosters = None  # ["gblinear"] for strong trends - can extrapolate

    @staticmethod
    def get_default_properties():
        return dict(col_type="time_column", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def _fit_async(X_path, grp_hash):
        np.random.seed(1234)
        random.seed(1234)
        X = load_obj(X_path)
        print("prophet - fitting on data of shape: %s for group: %s" % (str(X.shape), grp_hash))
        if X.shape[0] < 20:
            print("prophet - small data work-around for group: %s" % grp_hash)
            return grp_hash, None
        mod = importlib.import_module('fbprophet')
        Prophet = getattr(mod, "Prophet")
        model = Prophet()
        with suppress_stdout_stderr():
            model.fit(X[['ds', 'y']])
        model_path = os.path.join(temporary_files_path, "fbprophet_model" + str(uuid.uuid4()))
        save_obj(model, model_path)
        remove(X_path)  # remove to indicate success
        return grp_hash, model_path

    def fit(self, X: dt.Frame, y: np.array = None):
        XX = X[:, self.tgc].to_pandas()
        XX = XX.replace([None, np.nan], 0)
        XX.rename(columns={self.time_column: "ds"}, inplace=True)
        if self.labels is not None:
            y = LabelEncoder().fit(self.labels).transform(y)
        XX['y'] = np.array(y)
        self.nan_value = np.mean(y)  # TODO - store mean per group, not just global
        tgc_wo_time = list(np.setdiff1d(self.tgc, self.time_column))
        if len(tgc_wo_time) > 0:
            XX_grp = XX.groupby(tgc_wo_time)
        else:
            XX_grp = [([None], XX)]
        self.models = {}
        num_tasks = len(XX_grp)

        def processor(out, res):
            out[res[0]] = res[1]

        pool_to_use = small_job_pool
        pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks)
        for key, X in XX_grp:
            X_path = os.path.join(temporary_files_path, "fbprophet_X" + str(uuid.uuid4()))
            X = X.reset_index(drop=True)
            save_obj(X, X_path)
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            args = (X_path, grp_hash,)
            kwargs = {}
            pool.submit_tryget(None, MyParallelProphetTransformer_fit_async, args=args, kwargs=kwargs, out=self.models)
        pool.finish()
        for k, v in self.models.items():
            self.models[k] = load_obj(v) if v is not None else None
            remove(v)
        return self

    @staticmethod
    def _transform_async(model_path, X_path, nan_value):
        model = load_obj(model_path)
        XX_path = os.path.join(temporary_files_path, "fbprophet_XXt" + str(uuid.uuid4()))
        X = load_obj(X_path)
        if model is not None:
            XX = model.predict(X[['ds']])[['yhat']]
        else:
            XX = pd.DataFrame(np.full((X.shape[0], 1), nan_value), columns=['yhat'])  # invalid models
        XX.index = X.index
        assert XX.shape[1] == 1
        save_obj(XX, XX_path)
        remove(model_path)  # indicates success, no longer need
        remove(X_path)  # indicates success, no longer need
        return XX_path

    def transform(self, X: dt.Frame):
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
        pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks)
        XX_paths = []
        model_paths = []
        for key, X in XX_grp:
            key = key if isinstance(key, list) else [key]
            grp_hash = '_'.join(map(str, key))
            X_path = os.path.join(temporary_files_path, "fbprophet_Xt" + str(uuid.uuid4()))
            print("prophet - transforming data of shape: %s for group: %s" % (str(X.shape), grp_hash))
            if grp_hash in self.models:
                model = self.models[grp_hash]
                model_path = os.path.join(temporary_files_path, "fbprophet_modelt" + str(uuid.uuid4()))
                save_obj(model, model_path)
                save_obj(X, X_path)
                model_paths.append(model_path)

                args = (model_path, X_path, self.nan_value)
                kwargs = {}
                pool.submit_tryget(None, MyParallelProphetTransformer_transform_async, args=args, kwargs=kwargs,
                                   out=XX_paths)
            else:
                XX = pd.DataFrame(np.full((X.shape[0], 1), self.nan_value), columns=['yhat'])  # unseen groups
                save_obj(XX, X_path)
                XX_paths.append(X_path)
        pool.finish()
        XX = pd.concat((load_obj(XX_path) for XX_path in XX_paths), axis=0).sort_index()
        for p in XX_paths + model_paths:
            remove(p)
        return XX

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.fit(X, y).transform(X)
