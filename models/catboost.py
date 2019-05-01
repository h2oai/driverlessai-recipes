import sys
import datatable as dt
import numpy as np
import _pickle as pickle
from sklearn.preprocessing import LabelEncoder

from h2oaicore.models import CustomModel
from h2oaicore.systemutils import config, arch_type


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

    _boosters = ['catboost']
    _modules_needed_by_name = ['catboost']

    def set_default_params(self,
                           params,
                           accuracy, time_tolerance, interpretability,
                           **kwargs):
        params.update(dict(
            iterations=config.max_nestimators,
            learning_rate=config.min_learning_rate,
            early_stopping_rounds=20
        ))

    def get_random_specialparams(self,
                                 params,
                                 accuracy, time_tolerance, interpretability,
                                 **kwargs):
        # DUMMY version is do nothing
        if params is not None:
            return params
        else:
            return {}

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
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
                valid_y = lb.transform(valid_y)
                eval_set[0] = (valid_X, valid_y)
        else:
            orig_cols = list(X.columns)

        params = {'iterations': config.max_nestimators,
                  'learning_rate': config.min_learning_rate,
                  'thread_count': self.params.get('n_jobs', None)}  # -1 is not supported
        if self.num_classes == 1:
            self.model = CatBoostRegressor(**params)
        else:
            self.model = CatBoostClassifier(**params)
        # Hit sometimes: Exception: catboost/libs/data_new/quantization.cpp:779: All features are either constant or ignored.
        if 'param' in kwargs:
            baseline = kwargs['param'].get('base_score', None)
        else:
            baseline = None
        self.model.fit(X, y=y,
                       sample_weight=sample_weight,
                       baseline=baseline,
                       eval_set=eval_set,
                       early_stopping_rounds=kwargs.get('early_stopping_rounds', None),
                       verbose=self.params.get('verbose', False)
                       )

        # need to move to wrapper
        self.feature_names_fitted = orig_cols
        self.transformed_features = self.feature_names_fitted
        if self.model.get_best_iteration() is not None:
            self.best_ntree_limit = self.model.get_best_iteration()
        else:
            self.best_ntree_limit = params['iterations']
        # must always set best_iterations
        self.best_iterations = self.best_ntree_limit + 1
        self.set_feature_importances(self.model.feature_importances_)
        self.model_bytes = pickle.dumps(self.model, protocol=4)
        del self.model
        self.model = None
        return self

    def predict(self, X, **kwargs):
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
            kwargs['ntree_limit'] = min(config.fast_approx_num_trees, self.best_ntree_limit)
            kwargs['approx_contribs'] = pred_contribs
        else:
            kwargs['ntree_limit'] = self.best_ntree_limit

        # implicit import
        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        if not pred_contribs:
            self.get_model()
            if self.num_classes >= 2:
                preds = self.model.predict_proba(X,
                                                 ntree_start=self.best_ntree_limit,
                                                 thread_count=self.params.get('n_jobs', -1))  # None is not supported
                if preds.shape[1] == 2:
                    return preds[:, 1]
                else:
                    return preds
            else:
                return self.model.predict(X,
                                          ntree_start=self.best_ntree_limit,
                                          thread_count=self.params.get('n_jobs', -1))
        else:
            # For Shapley, doesn't come from predict, instead:
            return self.model.get_feature_importance(data=X,
                                                     ntree_start=self.best_ntree_limit,
                                                     thread_count=self.params.get('n_jobs', -1),
                                                     type=EFstrType.ShapValues)
            # FIXME: Do equivalent of preds = self._predict_internal_fixup(preds, **mykwargs) or wrap-up
