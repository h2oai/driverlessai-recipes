"""CatBoost gradient boosting by Yandex. Currently supports regression and binary classification."""
import copy

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
    _multiclass = True
    _display_name = "CatBoost"
    _description = "Yandex CatBoost GBM"
    _can_handle_categorical = True
    _can_handle_non_numeric = True
    _fit_by_iteration = True
    _fit_iteration_name = 'n_estimators'
    _predict_by_iteration = True
    _predict_iteration_name = 'ntree_end'

    _show_logger_test = True  # set to True to see how to send information to experiment logger
    _show_task_test = False  # set to True to see how task is used to send message to GUI

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
        #  https://catboost.ai/docs/concepts/python-reference_catboostclassifier.html
        # optimize for final model as transcribed from best lightgbm model
        n_estimators = self.params_base.get('n_estimators', 100)
        self.params = {'train_dir': config.data_directory,
                       'allow_writing_files': False,
                       'thread_count': self.params_base.get('n_jobs', n_jobs),  # -1 is not supported

                       'n_estimators': n_estimators,
                       'learning_rate': 0.018,
                       'random_seed': self.params_base.get('seed', 1234),
                       'metric_period': 500,
                       'od_wait': min(500, n_estimators),
                       'task_type': 'CPU' if self.params_base.get('n_gpus', 0) == 0 else 'GPU',
                       'depth': 8,
                       # 'one_hot_max_size': 10000,
                       # 'colsample_bylevel': 0.35, # kyak commented out
                       }

        lgbm_params_example = {'booster': 'lightgbm', 'model_class_name': 'LightGBMModel', 'n_gpus': 0, 'gpu_id': 0,
                               'n_jobs': 5, 'num_classes': 2, 'num_class': 1, 'score_f_name': 'AUC',
                               'random_state': 889688271,
                               'pred_gap': None, 'pred_periods': None, 'n_estimators': 6000,
                               'learning_rate': 0.018000000000000002, 'early_stopping_rounds': 100, 'reg_alpha': 0.0,
                               'reg_lambda': 0.5, 'gamma': 0, 'max_bin': 64, 'scale_pos_weight': 1, 'max_delta_step': 0,
                               'min_child_weight': 1, 'subsample': 1, 'colsample_bytree': 0.35, 'tree_method': 'hist',
                               'grow_policy': 'lossguide', 'max_leaves': 64, 'max_depth': 0, 'min_data_in_bin': 1,
                               'min_child_samples': 1, 'boosting_type': 'gbdt', 'objective': 'xentropy',
                               'eval_metric': 'auc',
                               'monotonicity_constraints': False, 'max_cat_to_onehot': 10000, 'silent': True,
                               'seed': 889688271,
                               'disable_gpus': True, 'lossguide': False, 'dummy': False, 'accuracy': 7,
                               'time_tolerance': 10,
                               'interpretability': 1, 'ensemble_level': 2, 'train_shape': (590540, 1182),
                               'valid_shape': None,
                               'model_origin': 'DefaultIndiv: do_te:True,interp:1,depth:6,num_as_cat:False',
                               'subsample_freq': 1, 'min_data_per_group': 10, 'max_cat_threshold': 50,
                               'cat_smooth': 1.0,
                               'cat_l2': 1.0}

    def mutate_params(self,
                      **kwargs):
        # Default version is do no mutation
        # Otherwise, change self.params for this model
        pass

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):

        if self._show_logger_test:
            # Example use of logger, with required import of:
            #  from h2oaicore.systemutils import make_experiment_logger, loggerinfo
            # Can use loggerwarning, loggererror, etc. for different levels
            logger = None
            if self.context and self.context.experiment_id:
                logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
                                                experiment_tmp_dir=self.context.experiment_tmp_dir)
            loggerinfo(logger, "TestLOGGER: Fit CatBoost")

        if self._show_task_test:
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
                message = "Tuning CatBoost"
                loggerinfo(logger, message)
                task = kwargs.get('task')
                if task:
                    task.sync(key=self.context.experiment_id, progress=dict(type='update', message=message))
                    task.flush()

        from catboost import CatBoostClassifier, CatBoostRegressor, EFstrType
        params = self.params.copy()
        lb = LabelEncoder()
        if self.num_classes >= 2:
            lb.fit(self.labels)
            y = lb.transform(y)
            params.update({'eval_metric': 'AUC', 'loss_function': 'Logloss'})
        if self.num_classes >= 2:
            params.update({'eval_metric': 'AUC', 'loss_function': 'MultiClass'})

        if isinstance(X, dt.Frame):
            orig_cols = list(X.names)
            # dt -> catboost internally using buffer leaks, so convert here
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
                eval_set = [(valid_X, valid_y)]
        else:
            orig_cols = list(X.columns)

        # unlike lightgbm that needs label encoded categoricals, catboots can take raw strings etc.
        params['cat_features'] = [i for i, x in enumerate(orig_cols) if 'CatOrig:' in x or 'Cat:' in x]
        params = self.filter_params(params)
        if self._show_logger_test:
            loggerinfo(logger, "CatBoost parameters: %s" % str(params))

        if self.num_classes == 1:
            model = CatBoostRegressor(**params)
        else:
            model = CatBoostClassifier(**params)
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
            iterations = self.params['n_estimators'] + 1
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
                preds = model.predict_proba(
                    data=X,
                    ntree_start=0,
                    ntree_end=iterations - 1,
                    thread_count=self.params['thread_count']
                )

                if preds.shape[1] == 2:
                    return preds[:, 1]
                else:
                    return preds
            else:
                return model.predict(
                    data=X,
                    ntree_start=0,
                    ntree_end=iterations - 1,
                    thread_count=self.params['thread_count']
                )
        else:
            # For Shapley, doesn't come from predict, instead:
            return model.get_feature_importance(
                data=X,
                ntree_start=0,
                ntree_end=iterations - 1,
                thread_count=self.params['thread_count'],
                type=EFstrType.ShapValues
            )
            # FIXME: Do equivalent of preds = self._predict_internal_fixup(preds, **mykwargs) or wrap-up

    def filter_params(self, params):
        regression = dict(iterations=None,
                          learning_rate=None,
                          depth=None,
                          l2_leaf_reg=None,
                          model_size_reg=None,
                          rsm=None,
                          loss_function='RMSE',
                          border_count=None,
                          feature_border_type=None,
                          per_float_feature_quantization=None,
                          input_borders=None,
                          output_borders=None,
                          fold_permutation_block=None,
                          od_pval=None,
                          od_wait=None,
                          od_type=None,
                          nan_mode=None,
                          counter_calc_method=None,
                          leaf_estimation_iterations=None,
                          leaf_estimation_method=None,
                          thread_count=None,
                          random_seed=None,
                          use_best_model=None,
                          best_model_min_trees=None,
                          verbose=None,
                          silent=None,
                          logging_level=None,
                          metric_period=None,
                          ctr_leaf_count_limit=None,
                          store_all_simple_ctr=None,
                          max_ctr_complexity=None,
                          has_time=None,
                          allow_const_label=None,
                          one_hot_max_size=None,
                          random_strength=None,
                          name=None,
                          ignored_features=None,
                          train_dir=None,
                          custom_metric=None,
                          eval_metric=None,
                          bagging_temperature=None,
                          save_snapshot=None,
                          snapshot_file=None,
                          snapshot_interval=None,
                          fold_len_multiplier=None,
                          used_ram_limit=None,
                          gpu_ram_part=None,
                          pinned_memory_size=None,
                          allow_writing_files=None,
                          final_ctr_computation_mode=None,
                          approx_on_full_history=None,
                          boosting_type=None,
                          simple_ctr=None,
                          combinations_ctr=None,
                          per_feature_ctr=None,
                          ctr_target_border_count=None,
                          task_type=None,
                          device_config=None,
                          devices=None,
                          bootstrap_type=None,
                          subsample=None,
                          sampling_unit=None,
                          dev_score_calc_obj_block_size=None,
                          max_depth=None,
                          n_estimators=None,
                          num_boost_round=None,
                          num_trees=None,
                          colsample_bylevel=None,
                          random_state=None,
                          reg_lambda=None,
                          objective=None,
                          eta=None,
                          max_bin=None,
                          gpu_cat_features_storage=None,
                          data_partition=None,
                          metadata=None,
                          early_stopping_rounds=None,
                          cat_features=None,
                          grow_policy=None,
                          min_data_in_leaf=None,
                          min_child_samples=None,
                          max_leaves=None,
                          num_leaves=None,
                          score_function=None,
                          leaf_estimation_backtracking=None,
                          ctr_history_unit=None,
                          monotone_constraints=None)

        classification = dict(iterations=None,
                              learning_rate=None,
                              depth=None,
                              l2_leaf_reg=None,
                              model_size_reg=None,
                              rsm=None,
                              loss_function=None,
                              border_count=None,
                              feature_border_type=None,
                              per_float_feature_quantization=None,
                              input_borders=None,
                              output_borders=None,
                              fold_permutation_block=None,
                              od_pval=None,
                              od_wait=None,
                              od_type=None,
                              nan_mode=None,
                              counter_calc_method=None,
                              leaf_estimation_iterations=None,
                              leaf_estimation_method=None,
                              thread_count=None,
                              random_seed=None,
                              use_best_model=None,
                              verbose=None,
                              logging_level=None,
                              metric_period=None,
                              ctr_leaf_count_limit=None,
                              store_all_simple_ctr=None,
                              max_ctr_complexity=None,
                              has_time=None,
                              allow_const_label=None,
                              classes_count=None,
                              class_weights=None,
                              one_hot_max_size=None,
                              random_strength=None,
                              name=None,
                              ignored_features=None,
                              train_dir=None,
                              custom_loss=None,
                              custom_metric=None,
                              eval_metric=None,
                              bagging_temperature=None,
                              save_snapshot=None,
                              snapshot_file=None,
                              snapshot_interval=None,
                              fold_len_multiplier=None,
                              used_ram_limit=None,
                              gpu_ram_part=None,
                              allow_writing_files=None,
                              final_ctr_computation_mode=None,
                              approx_on_full_history=None,
                              boosting_type=None,
                              simple_ctr=None,
                              combinations_ctr=None,
                              per_feature_ctr=None,
                              task_type=None,
                              device_config=None,
                              devices=None,
                              bootstrap_type=None,
                              subsample=None,
                              sampling_unit=None,
                              dev_score_calc_obj_block_size=None,
                              max_depth=None,
                              n_estimators=None,
                              num_boost_round=None,
                              num_trees=None,
                              colsample_bylevel=None,
                              random_state=None,
                              reg_lambda=None,
                              objective=None,
                              eta=None,
                              max_bin=None,
                              scale_pos_weight=None,
                              gpu_cat_features_storage=None,
                              data_partition=None,
                              metadata=None,
                              early_stopping_rounds=None,
                              cat_features=None,
                              grow_policy=None,
                              min_data_in_leaf=None,
                              min_child_samples=None,
                              max_leaves=None,
                              num_leaves=None,
                              score_function=None,
                              leaf_estimation_backtracking=None,
                              ctr_history_unit=None,
                              monotone_constraints=None)

        if self.num_classes == 1:
            allowed_params = regression
        else:
            allowed_params = classification

        params_copy = copy.deepcopy(params)
        for k, v in params_copy.items():
            if k not in allowed_params.keys():
                del params[k]
        return params
