import sys
import datatable as dt
import numpy as np
import _pickle as pickle
import pandas as pd

from h2oaicore.models import CustomModel
from h2oaicore.systemutils import config, arch_type


# https://github.com/KwokHing/YandexCatBoost-Python-Demo
class MyCatBoostModel(CustomModel):
    @staticmethod
    def is_enabled():
        return not (arch_type == "ppc64le")

    _boosters = ['catboost']
    _modules_needed_by_name = ['catboost']

    def set_default_params(self, params, logger=None, num_classes=None, seed=1234, disable_gpus=False,
                           score_f_name: str = None,
                           lossguide=False,
                           monotonicity_constraints=False, silent=True, dummy=False, accuracy=None,
                           time_tolerance=None,
                           interpretability=None, min_child_weight=1.0, params_orig=None, ensemble_level=0,
                           train_shape=None, valid_shape=None, labels=None):

        params_new = {'iterations': config.max_nestimators, 'learning_rate': config.min_learning_rate}
        params.update(params_new)
        params['early_stopping_rounds'] = 20

    def get_random_specialparams(self, get_best=False, time_tolerance=None, accuracy=None,
                                 imbalance_ratio=None,
                                 train_shape=None, ncol_effective=None,
                                 params_orig=None, time_series=False, ensemble_level=None,
                                 score_f_name: str = None):
        # DUMMY version is do nothing
        if params_orig is not None:
            return params_orig
        else:
            return {}

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType

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
                eval_set[0] = (valid_X, valid_y)
        else:
            orig_cols = list(X.columns)

        params = {'iterations': config.max_nestimators, 'learning_rate': config.min_learning_rate}
        if self.num_classes == 1:
            self.model = CatBoostRegressor(**params)
        else:
            self.model = CatBoostClassifier(**params)
        # Hit sometimes: Exception: catboost/libs/data_new/quantization.cpp:779: All features are either constant or ignored.
        print("CATBOOST")
        sys.stdout.flush()
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
                                                 thread_count=self.params.get('n_jobs', -1))
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

