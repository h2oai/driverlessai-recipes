"""CatBoost gradient boosting by Yandex. Currently supports regression and binary classification."""
import datatable as dt
import numpy as np
import _pickle as pickle
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import CustomModel
from h2oaicore.systemutils import config, arch_type, physical_cores_count
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


# https://github.com/KwokHing/YandexCatBoost-Python-Demo
class CatBoostModel(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # WIP
    _display_name = "CatBoost"
    _description = "Yandex CatBoost GBM"

    @staticmethod
    def is_enabled():
        return not (arch_type == "ppc64le")

    @staticmethod
    def do_acceptance_test():
        return True

    _modules_needed_by_name = ['catboost']

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        n_jobs = max(1, physical_cores_count)
        max_iterations = min(kwargs['n_estimators'],
                             config.max_nestimators) if 'n_estimators' in kwargs else config.max_nestimators
        max_iterations = min(kwargs['iterations'], max_iterations) if 'iterations' in kwargs else max_iterations
        self.params = {'iterations': max_iterations,
                       'learning_rate': config.min_learning_rate,
                       'train_dir': config.data_directory,
                       'allow_writing_files': False,
                       'thread_count': self.params_base.get('n_jobs', n_jobs),  # -1 is not supported
                       'early_stopping_rounds': 20
                       }

    def mutate_params(self,
                      accuracy=None, time_tolerance=None, interpretability=None,
                      **kwargs):
        # Default version is do no mutation
        # Otherwise, change self.params for this model
        pass

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        # Example use of logger, with required import of:
        #  from h2oaicore.systemutils import make_experiment_logger, loggerinfo
        # Can use loggerwarning, loggererror, etc. for different levels
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                            experiment_tmp_dir=self.context.experiment_tmp_dir)
        loggerinfo(logger, "TestLOGGER: Fit CatBoost")

        # Example task sync operations
        if hasattr(self, 'testcount'):
            self.test_count += 1
        else:
            self.test_count = 0

        # The below generates a message in the GUI notifications panel
        if self.test_count == 0 and self.context and self.context.experiment_id:
            warning = "TestWarning: First CatBoost fit for this model instance"
            loggerwarning(logger, warning)
            task = kwargs.get('task')
            if task:
                task.sync(key=self.context.experiment_id, progress=dict(type='warning', data=warning))
                task.flush()

        # The below generates a message in the GUI top-middle panel above the progress wheel
        if self.test_count == 0 and self.context and self.context.experiment_id:
            message = "TestMessage: CatBoost"
            loggerinfo(logger, message)
            task = kwargs.get('task')
            if task:
                task.sync(key=self.context.experiment_id, progress=dict(type='update', message=message))
                task.flush()

        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        lb = LabelEncoder()
        if self.num_classes >= 2:
            lb.fit(self.labels)
            y = lb.transform(y)

        if isinstance(X, dt.Frame):
            orig_cols = list(X.names)
            # dt -> lightgbm internally using buffer leaks, so convert here
            # assume predict is after pipeline collection or in subprocess so needs no protection
            X = X.to_numpy()  # don't assign back to X so don't damage during predict
            X = np.ascontiguousarray(X, dtype=np.float32 if config.data_precision == "float32" else np.float64)
            if eval_set is not None:
                valid_X = eval_set[0][0].to_numpy()  # don't assign back to X so don't damage during predict
                valid_X = np.ascontiguousarray(valid_X,
                                               dtype=np.float32 if config.data_precision == "float32" else np.float64)
                valid_y = eval_set[0][1]
                if self.num_classes >= 2:
                    valid_y = lb.transform(valid_y)
                eval_set[0] = (valid_X, valid_y)
        else:
            orig_cols = list(X.columns)

        if self.num_classes == 1:
            model = CatBoostRegressor(**self.params)
        else:
            model = CatBoostClassifier(**self.params)
        # Hit sometimes: Exception: catboost/libs/data_new/quantization.cpp:779: All features are either constant or ignored.
        if self.num_classes == 1:
            # assume not mae, which would use median
            # baseline = [np.mean(y)] * len(y)
            baseline = None
        else:
            baseline = None

        model.fit(X, y=y,
                  sample_weight=sample_weight,
                  baseline=baseline,
                  eval_set=eval_set,
                  early_stopping_rounds=kwargs.get('early_stopping_rounds', None),
                  verbose=self.params.get('verbose', False)
                  )

        # need to move to wrapper
        if model.get_best_iteration() is not None:
            iterations = model.get_best_iteration() + 1
        else:
            iterations = self.params['iterations'] + 1
        # must always set best_iterations
        self.set_model_properties(model=model,
                                  features=orig_cols,
                                  importances=model.feature_importances_,
                                  iterations=iterations)

    def predict(self, X, **kwargs):
        model, features, importances, iterations = self.get_model_properties()
        # FIXME: Do equivalent throttling of predict size like def _predict_internal(self, X, **kwargs), wrap-up.
        if isinstance(X, dt.Frame):
            # dt -> lightgbm internally using buffer leaks, so convert here
            # assume predict is after pipeline collection or in subprocess so needs no protection
            X = X.to_numpy()  # don't assign back to X so don't damage during predict
            X = np.ascontiguousarray(X, dtype=np.float32 if config.data_precision == "float32" else np.float64)

        pred_contribs = kwargs.get('pred_contribs', None)
        output_margin = kwargs.get('output_margin', None)
        fast_approx = kwargs.pop('fast_approx', False)
        if fast_approx:
            kwargs['ntree_limit'] = min(config.fast_approx_num_trees, iterations - 1)
            kwargs['approx_contribs'] = pred_contribs
        else:
            kwargs['ntree_limit'] = iterations - 1

        # implicit import
        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        if not pred_contribs:
            if self.num_classes >= 2:
                preds = model.predict_proba(X,
                                            ntree_start=iterations - 1,
                                            thread_count=self.params['thread_count'])
                if preds.shape[1] == 2:
                    return preds[:, 1]
                else:
                    return preds
            else:
                return model.predict(X,
                                     ntree_start=iterations - 1,
                                     thread_count=self.params['thread_count'])
        else:
            # For Shapley, doesn't come from predict, instead:
            return model.get_feature_importance(data=X,
                                                ntree_start=iterations - 1,
                                                thread_count=self.params['thread_count'],
                                                type=EFstrType.ShapValues)
            # FIXME: Do equivalent of preds = self._predict_internal_fixup(preds, **mykwargs) or wrap-up
