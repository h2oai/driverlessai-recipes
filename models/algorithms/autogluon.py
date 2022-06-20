"""AutoGluon + RAPIDS"""
from h2oaicore.models_custom import CustomModel


class AutoGluonModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = True
    _display_name = "AutoGluon"
    _description = "AutoGluon Model"
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _mojo = False
    _mutate_all = 'auto'
    _can_handle_non_numeric = False
    _datatable_in_out = True
    _force_no_fork_isolation = False
    _can_use_gpu = True
    _can_use_multi_gpu = False
    _get_gpu_lock_vis = True
    _must_use_gpu = False
    _booster_str = 'autogluon'
    _supports_sample_weight = True
    _supports_eval_set = True
    _supports_missings = True
    _used_return_params = True  # so Optuna (non-GA) results passed back to individual scored params tables etc.
    _average_return_params = True

    # autogluon depends upon slightly different package versions than DAI has, or could work with xgboost but needs official xgboost <1.5
    # so use isolated env
    isolate_env = dict(pyversion="3.8", install_h2oaicore=False, install_datatable=True, cache_env=True,
                       cache_by_full_module_name=False, install_pip="latest",
                       modules_needed_by_name=['autogluon==0.3.1'])

    @staticmethod
    def is_enabled():
        import os
        # avoid testing until speed-up when used
        return 'GIT_HASH' not in os.environ

    @staticmethod
    def acceptance_test_coverage_fraction():
        import os
        return 0.05 if 'GIT_HASH' in os.environ else 1.0

    @staticmethod
    def fit_static(X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        from autogluon.tabular import TabularDataset, TabularPredictor
        from autogluon.tabular.models.knn.knn_rapids_model import KNNRapidsModel
        from autogluon.tabular.models.lr.lr_rapids_model import LinearRapidsModel

        num_classes = kwargs['num_classes']
        if kwargs['verbose'] is not None and kwargs['verbose'] is True:
            verbosity = 2
        else:
            verbosity = 0
        labels = kwargs['labels']
        num_gpus = kwargs['n_gpus']
        accuracy = kwargs.get('accuracy', 10)
        interpretability = kwargs.get('interpretability', 1)
        is_acceptance = kwargs.get('IS_ACCEPTANCE', False)
        is_backend_tuning = kwargs.get('IS_BACKEND_TUNING', False)

        lb = None
        if num_classes >= 2:
            from sklearn.preprocessing import LabelEncoder
            lb = LabelEncoder()
            lb.fit(labels)
            y = lb.transform(y)

        label = '____TARGET_____'
        import datatable as dt
        y_dt = dt.Frame(y, names=[label])

        if eval_set is not None:
            valid_X = eval_set[0][0]
            valid_y = eval_set[0][1]
            if num_classes >= 2:
                valid_y = lb.transform(valid_y)
            valid_y_dt = dt.Frame(valid_y, names=[label])

            assert X.shape[1] == valid_X.shape[1], "Bad shape to rbind: %s %s : %s %s" % (
                X.shape, X.names, valid_X.shape, valid_X.names)
            X = dt.rbind([X, valid_X])
            y_dt = dt.rbind([y_dt, valid_y_dt])

        sw = None
        if sample_weight is not None:
            sw = '____SAMPLE_WEIGHT_____'
            sw_dt = dt.Frame(sample_weight, names=[sw])
            if sample_weight_eval_set is not None:
                swes_dt = dt.Frame(sample_weight_eval_set[0], names=[sw])
                sw_dt = dt.rbind([sw_dt, swes_dt])
            X = dt.cbind([X, y_dt, sw_dt])
        else:
            X = dt.cbind([X, y_dt])

        X = X.to_pandas()  # AutoGluon needs pandas, not numpy

        eval_metric = AutoGluonModel.get_eval_metric(**kwargs)
        time_limit = AutoGluonModel.get_time_limit(accuracy)
        presets = AutoGluonModel.get_presets(accuracy, interpretability, is_acceptance, is_backend_tuning)

        model = TabularPredictor(
            label=label,
            sample_weight=sw,
            eval_metric=eval_metric,
            verbosity=verbosity,
            # learner_kwargs={'ignored_columns': ['id']}
        )
        n_jobs = kwargs.get('n_jobs', 4) or 4
        hyperparameters = {
            KNNRapidsModel: {},
            LinearRapidsModel: {},
            'RF': {},
            'XGB': {'n_jobs': n_jobs, 'ag_args_fit': {'num_gpus': num_gpus, 'num_cpus': n_jobs}},
            'CAT': {'thread_count': n_jobs, 'ag_args_fit': {'num_gpus': num_gpus, 'num_cpus': n_jobs}},
            'GBM': [{}, {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, 'GBMLarge'],
            'NN': {'ag_args_fit': {'num_gpus': num_gpus, 'num_cpus': n_jobs}},
            'FASTAI': {'ag_args_fit': {'num_gpus': num_gpus, 'num_cpus': n_jobs}},
        }
        kwargs_fit = dict(hyperparameters=hyperparameters)
        if accuracy >= 5:
            kwargs_fit.update(dict(presets=presets, time_limit=time_limit))
        model.fit(X, **kwargs_fit)

        print(model.leaderboard(silent=True))

        return model

    @staticmethod
    def get_presets(accuracy, interpretability, is_acceptance, is_backend_tuning):
        if is_acceptance or is_backend_tuning:
            return 'medium_quality_faster_train'
        if accuracy >= 8:
            return 'best_quality'
        elif accuracy >= 5:
            return 'high_quality_fast_inference_only_refit'
        elif accuracy >= 3:
            return 'good_quality_faster_inference_only_refit'
        elif accuracy >= 2:
            return 'medium_quality_faster_train'
        elif accuracy >= 1:
            return 300
        if interpretability >= 9:
            return 'optimize_for_deployment'
        return 'best_quality'

    @staticmethod
    def get_time_limit(accuracy):
        if accuracy >= 8:
            return None
        elif accuracy >= 5:
            return 7200
        elif accuracy >= 3:
            return 3600
        elif accuracy >= 2:
            return 1000
        elif accuracy >= 1:
            return 300
        return None

    @staticmethod
    def get_eval_metric(**kwargs):
        num_classes = kwargs['num_classes']
        if kwargs['score_f_name'] is None:
            if num_classes >= 2:
                eval_metric = 'log_loss'
            else:
                eval_metric = 'root_mean_squared_error'
        elif kwargs['score_f_name'].lower() == 'accuracy':
            eval_metric = 'accuracy'
        elif kwargs['score_f_name'].lower() == 'f1':
            eval_metric = 'f1'
        elif kwargs['score_f_name'].lower() == 'auc':
            if num_classes == 2:
                eval_metric = 'roc_auc'
            else:
                # roc_auc would hit: multiclass format is not supported
                eval_metric = 'log_loss'
        elif kwargs['score_f_name'].lower() == 'precision':
            eval_metric = 'precision'
        elif kwargs['score_f_name'].lower() == 'recall':
            eval_metric = 'recall'
        elif kwargs['score_f_name'].lower() == 'logloss':
            eval_metric = 'log_loss'
        elif kwargs['score_f_name'].lower() == 'macrof1':
            eval_metric = 'f1_macro'
        elif kwargs['score_f_name'].lower() == 'aucpr':
            eval_metric = 'average_precision'
        elif kwargs['score_f_name'].lower() == 'rmse':
            eval_metric = 'root_mean_squared_error'
        elif kwargs['score_f_name'].lower() == 'mae':
            eval_metric = 'mean_absolute_error'
        elif kwargs['score_f_name'].lower() == 'mse':
            eval_metric = 'mean_squared_error'
        elif kwargs['score_f_name'].lower() == 'r2':
            eval_metric = 'r2'
        else:
            if num_classes >= 2:
                eval_metric = 'log_loss'
            else:
                eval_metric = 'root_mean_squared_error'
        return eval_metric

    @staticmethod
    def predict_static(model, X, **kwargs):
        import datatable as dt
        import pandas as pd
        X = dt.Frame(X)
        X = X.to_pandas()
        num_classes = kwargs['num_classes']
        if num_classes == 1:
            preds = model.predict(X)
        else:
            preds = model.predict_proba(X)
        return dt.Frame(pd.DataFrame(preds))
