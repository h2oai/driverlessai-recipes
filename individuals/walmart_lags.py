"""Custom Individual 0 from Experiment test_time_series_walmart_nolimits_ac642_6d3b_subosahu """
from h2oaicore.ga import CustomIndividual
class Indivtesttimeserieswalmartnolimitsac6426d3bsubosahu_finalFalse_id0(CustomIndividual):


    def set_params(self):
        # Below block of variables are not required to be set
        # They are only informative

        # Was best in population
        self.final_best = True
        # Was final population
        self.final_pop = True
        # Was in final model
        self.is_final = True

        # Score function's (hashed) name
        self.score_f_name = 'MAE'
        # Score
        self.score = 2333.444580078125
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 20.78180694580078
        # Tournament Score (penalized by features counts or interpretabilty)
        self.tournament_score = 2333.444580078125
        # Score history during tuning and evolution
        self.score_list = [2409.958, 2409.9580078125]
        # Score standard deviation history during tuning and evolution
        self.score_sd_list = [497.51245, 497.512451171875]

        # Number of classes if supervised
        self.num_classes = 1
        # Labels if classification, None for regression
        self.labels = None

        # Shape of training frame (may include target)
        self.train_shape = (421570, 6)
        # Shape of validation frame (may include target)
        self.valid_shape = None
        # Cardinality for each column
        self.cardinality_dict = {'Date': 143, 'Dept': 81, 'IsHoliday': 2, 'Store': 45}

        # Target column
        self.target = 'Weekly_Sales'
        # Label counts for target column
        self.label_counts = None
        # Imbalanced ratio
        self.imbalance_ratio = None

        # Weight column
        self.weight_column = 'sample_weight'
        # Time column
        self.time_column = 'Date'

        # Number of validation splits
        self.num_validation_splits = 3
        # Seed for individual
        self.seed = 71307952
        # factor of extra genes added during activation
        self.default_factor = 1
        # Ensemble level
        self.ensemble_level = 0

        # Parameters set for custom individual
        self.params = {'accuracy': 5,
                       'do_te': True,
                       'explore_anneal_factor': 0.9,
                       'explore_model_anneal_factor': 0.9,
                       'explore_model_prob': 0.5,
                       'explore_model_prob_lowest': 0.1,
                       'explore_prob': 0.5,
                       'explore_prob_lowest': 0.1,
                       'grow_anneal_factor': 0.5,
                       'grow_prob': 0.8,
                       'grow_prob_lowest': 0.05,
                       'interpretability': 5,
                       'nfeatures_max': 300,
                       'nfeatures_min': 1,
                       'ngenes_max': 300,
                       'ngenes_min': 1,
                       'num_as_cat': False,
                       'output_features_to_drop_more': ['1_Date:Date~get_weekday',
                                                        '1_Date:Date~get_weekday'],
                       'random_state': 71307952,
                       'time_tolerance': 1}

    def set_model(self):
        # Display name corresponds to hashed (for custom recipes) display names to ensure exact match
        # One can also provide short names if only one recipe
        self.model_display_name = 'LightGBM'

        # Model parameters
        # Some system-related parameters are overwritten by DAI, e.g. gpu_id, n_jobs for xgboost
        # Monotonicity constraints remain determined by expert toml settings,
        #  e.g. monotonicity_constraints_dict can be used to constrain feature names
        #  To really set own constraints in model parameters for XGBoost and LightGBM, one can set them here,
        #  but then set monotonicity_constraints_interpretability_switch toml to 15 to avoid automatic override
        #  of those monotone_constraints params
        # Some parameters like categorical_feature for LightGBM are specific to that recipe, and automatically
        #  get handled for features that use CatTransformer
        # Some parameters like learning_rate and n_estimators are specific to that recipe, and automatically
        #  are adjusted for setting of accuracy dial.  A custom recipe wrapper could be written and one could set
        #  the static var _gbm = False to avoid such changes to learning rate and n_estimators.
        # Some parameters like disable_gpus are internal to DAI but appear in the model parameters, but they are only
        #  for information purposes and do not affect the model.
        import numpy as np
        nan = np.nan

        self.model_params = {'bagging_seed': 71307954,
                             'booster': 'lightgbm',
                             'boosting_type': 'gbdt',
                             'categorical_feature': '',
                             'class_weight': None,
                             'colsample_bytree': 0.8000000000000002,
                             'deterministic': False,
                             'device_type': 'cpu',
                             'disable_gpus': False,
                             'early_stopping_rounds': 20,
                             'early_stopping_threshold': 0.0,
                             'enable_early_stopping_rounds': True,
                             'eval_metric': 'mae',
                             'feature_fraction_seed': 71307953,
                             'gamma': 0.0,
                             'gpu_device_id': 0,
                             'gpu_platform_id': 0,
                             'gpu_use_dp': True,
                             'grow_policy': 'depthwise',
                             'importance_type': 'gain',
                             'label_counts': None,
                             'labels': None,
                             'learning_rate': 0.15,
                             'max_bin': 251,
                             'max_delta_step': 0.0,
                             'max_depth': 8,
                             'max_leaves': 256,
                             'min_child_samples': 20,
                             'min_child_weight': 0.001,
                             'min_data_in_bin': 1,
                             'min_split_gain': 0.0,
                             'model_class_name': 'LightGBMModel',
                             'model_origin': 'FINAL MODEL',
                             'model_origin_original': 'SEQUENCE',
                             'monotonicity_constraints': False,
                             'n_estimators': 500,
                             'n_gpus': 1,
                             'n_jobs': 8,
                             'num_class': 1,
                             'num_classes': 1,
                             'num_leaves': 256,
                             'num_threads': 8,
                             'objective': 'mse',
                             'random_state': 71307952,
                             'reg_alpha': 0.0,
                             'reg_lambda': 0.0,
                             'scale_pos_weight': 1.0,
                             'score_f_name': 'MAE',
                             'seed': 71307952,
                             'silent': True,
                             'subsample': 0.6999999999999998,
                             'subsample_for_bin': 200000,
                             'subsample_freq': 1,
                             'verbose': -1}

        # model origin is for informative purposes only
        self.model_origin = 'FINAL MODEL'

    def set_target_transformer(self):
        self.target_transformer_name = 'TargetTransformer_standardize'

    def set_genes(self):
        import numpy as np
        nan = np.nan
        from collections import OrderedDict, defaultdict

        # Original variable importances are for reference only, not required to be set
        self.importances_orig = {'CPI': 0.0,
                                 'Date': 0.3388838057640152,
                                 'Dept': 0.32935702060532474,
                                 'Fuel_Price': 0.0,
                                 'IsHoliday': 0.002402153025335408,
                                 'MarkDown1': 0.0,
                                 'MarkDown2': 0.0,
                                 'MarkDown3': 0.0,
                                 'MarkDown4': 0.0,
                                 'MarkDown5': 0.0,
                                 'Store': 0.32935702060532474,
                                 'Temperature': 0.0,
                                 'Unemployment': 0.0}
        # Column types are for reference only, not required to be set
        self.columns = {'all': ['Date', 'Dept', 'IsHoliday', 'Store'],
                        'any': ['Date', 'Dept', 'IsHoliday', 'Store'],
                        'categorical': [],
                        'catlabel': [],
                        'date': ['Date'],
                        'datetime': [],
                        'id': [],
                        'image': [],
                        'numeric': ['IsHoliday'],
                        'ohe_categorical': [],
                        'raw': [],
                        'text': [],
                        'time_column': ['Date']}

        # All valid parameters for genes should be provided, except:
        # output_features_to_drop need not be passed if empty list
        # Mutations need not be provided if want to use default values
        # Mutations or valid parameters are not shown if none, like for OriginalTransformer

        # Gene Normalized Importance: 0.0049551
        # Transformed Feature Names and Importances: {'0_IsHoliday': 0.004955092445015907}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['IsHoliday'], 'random_state': 71307952}
        self.add_transformer('OriginalTransformer', gene_index=0, **params)

        # Gene Normalized Importance: 0.019652
        # Transformed Feature Names and Importances: {'1_Date:Date~get_dayofyear': 0.008648863062262535, '1_Date:Date~get_day': 0.006139116361737251, '1_Date:Date~get_year': 0.0027122267056256533, '1_Date:Date~get_week': 0.0018609896069392562, '1_Date:Date~get_month': 0.00023486147983931005, '1_Date:Date~get_quarter': 5.5522290494991466e-05}
        # Valid parameters: ['dt_cols', 'funcs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        # Allowed parameters and mutations (first mutation in list is default): {'funcs': [['year', 'quarter', 'month', 'week', 'weekday', 'day', 'dayofyear', 'num']]}
        params = {'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dt_cols': ['Date'],
                  'funcs': ['year',
                            'quarter',
                            'month',
                            'week',
                            'weekday',
                            'day',
                            'dayofyear',
                            'num'],
                  'output_features_to_drop': ['1_Date:Date~get_weekday'],
                  'random_state': 71307957}
        self.add_transformer('DatesTransformer', gene_index=1, **params)

        # Gene Normalized Importance: 0.22801
        # Transformed Feature Names and Importances: {'2_TargetLag:Date:Dept:Store.52': 0.14521296322345734, '2_TargetLag:Date:Dept:Store.53': 0.012190841138362885, '2_TargetLag:Date:Dept:Store.39': 0.01149847824126482, '2_TargetLag:Date:Dept:Store.43': 0.011346477083861828, '2_TargetLag:Date:Dept:Store.48': 0.007632155902683735, '2_TargetLag:Date:Dept:Store.47': 0.007067451253533363, '2_TargetLag:Date:Dept:Store.45': 0.005960228852927685, '2_TargetLag:Date:Dept:Store.44': 0.005356497596949339, '2_TargetLag:Date:Dept:Store.42': 0.005342171527445316, '2_TargetLag:Date:Dept:Store.41': 0.004222402349114418, '2_TargetLag:Date:Dept:Store.40': 0.0031587656121701, '2_TargetLag:Date:Dept:Store.46': 0.0029946821741759777, '2_TargetLag:Date:Dept:Store.71': 0.0016180849634110928, '2_TargetLag:Date:Dept:Store.51': 0.0013273628428578377, '2_TargetLag:Date:Dept:Store.50': 0.001012562308460474, '2_TargetLag:Date:Dept:Store.60': 0.0009900506120175123, '2_TargetLag:Date:Dept:Store.49': 0.0005796392215415835, '2_TargetLag:Date:Dept:Store.59': 0.0004957961500622332}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        # 'encoder' parameter is handled internally by DAI
        # 'dropout' options (keys, same order as toml 'lags_dropout'): {'DISABLED': 0, 'DEPENDENT': 1, 'INDEPENDENT': 2}
        params = {'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dropout': 1,
                  'is_ufapt': True,
                  'lag_feature': 'Weekly_Sales',
                  'lag_sizes': [52,
                                53,
                                45,
                                51,
                                44,
                                60,
                                40,
                                50,
                                39,
                                71,
                                59,
                                41,
                                42,
                                43,
                                46,
                                47,
                                48,
                                49],
                  'mfr': False,
                  'n_jobs': 1,
                  'nan_value': nan,
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 71307960,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsTransformer', gene_index=2, **params)

        # Gene Normalized Importance:  1.4256
        # Transformed Feature Names and Importances: {'3_TargetLagsMean:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 1.0, '3_TargetLagsSum:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.3019547164440155, '3_TargetLagsMedian:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.04624990001320839, '3_TargetLagsSkew:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.032831694930791855, '3_TargetLagsMin:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.016932565718889236, '3_TargetLagsStd:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.012253103777766228, '3_TargetLagsKurtosis:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.011466425843536854, '3_TargetLagsMax:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.003885742509737611}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'aggregates', 'nan_value', 'dropout', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        # 'encoder' parameter is handled internally by DAI
        # 'dropout' options (keys, same order as toml 'lags_dropout'): {'DISABLED': 0, 'DEPENDENT': 1, 'INDEPENDENT': 2}
        params = {'aggregates': ['min',
                                 'max',
                                 'mean',
                                 'std',
                                 'skew',
                                 'kurtosis',
                                 'median',
                                 'sum'],
                  'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dropout': 1,
                  'is_ufapt': True,
                  'lag_feature': 'Weekly_Sales',
                  'lag_sizes': [52,
                                53,
                                45,
                                51,
                                44,
                                60,
                                40,
                                50,
                                39,
                                71,
                                59,
                                41,
                                42,
                                43,
                                46,
                                47,
                                48,
                                49],
                  'mfr': False,
                  'n_jobs': 1,
                  'nan_value': nan,
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 2642419254,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsAggregatesTransformer', gene_index=3, **params)

        # Gene Normalized Importance: 0.034736
        # Transformed Feature Names and Importances: {'7_TargetLag:Date:Dept:Store.52': 0.021422937512397766, '7_TargetLag:Date:Dept:Store.56': 0.006232628598809242, '7_TargetLag:Date:Dept:Store.48': 0.004400585312396288, '7_TargetLag:Date:Dept:Store.44': 0.0016071508871391416, '7_TargetLag:Date:Dept:Store.64': 0.0005308681866154075, '7_TargetLag:Date:Dept:Store.40': 0.00041295579285360873, '7_TargetLag:Date:Dept:Store.60': 0.00012920604785904288}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        # 'encoder' parameter is handled internally by DAI
        # 'dropout' options (keys, same order as toml 'lags_dropout'): {'DISABLED': 0, 'DEPENDENT': 1, 'INDEPENDENT': 2}
        params = {'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dropout': 1,
                  'is_ufapt': True,
                  'lag_feature': 'Weekly_Sales',
                  'lag_sizes': [40, 44, 48, 52, 56, 60, 64],
                  'mfr': False,
                  'n_jobs': 1,
                  'nan_value': nan,
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 3023037155,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsTransformer', gene_index=7, **params)

        # Gene Normalized Importance: 0.0008144
        # Transformed Feature Names and Importances: {'9_TargetLagInteraction:Date:Dept:Store.44sub60': 0.000814399856608361}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_feature', 'is_ufapt', 'lag_interactions', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        # 'encoder' parameter is handled internally by DAI
        # 'lag_interactions' op options: ['+', '-', '*', '/']
        # 'dropout' options (keys, same order as toml 'lags_dropout'): {'DISABLED': 0, 'DEPENDENT': 1, 'INDEPENDENT': 2}
        params = {'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dropout': 1,
                  'is_ufapt': True,
                  'lag_feature': 'Weekly_Sales',
                  'lag_interactions': [{'lag1': 44, 'lag2': 60, 'op': '-'}],
                  'mfr': False,
                  'n_jobs': 1,
                  'nan_value': nan,
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 939989390,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsInteractionTransformer', gene_index=9, **params)

        # Gene Normalized Importance: 0.34903
        # Transformed Feature Names and Importances: {'11_EWMA(0.05)(0)TargetLags:Date:Dept:Store.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.3291340470314026, '11_EWMA(0.05)(2)TargetLags:Date:Dept:Store.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.009999675676226616, '11_EWMA(0.05)(1)TargetLags:Date:Dept:Store.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.00989946536719799}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'alpha', 'orders', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        # 'encoder' parameter is handled internally by DAI
        # 'dropout' options (keys, same order as toml 'lags_dropout'): {'DISABLED': 0, 'DEPENDENT': 1, 'INDEPENDENT': 2}
        # Allowed parameters and mutations (first mutation in list is default): {'alpha': [0.05, 0, 0.1, 0.5, 0.9]}
        params = {'alpha': 0.05,
                  'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dropout': 1,
                  'is_ufapt': True,
                  'lag_feature': 'Weekly_Sales',
                  'lag_sizes': [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64],
                  'mfr': False,
                  'n_jobs': 1,
                  'nan_value': nan,
                  'orders': [0, 1, 2],
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 2832822204,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('EwmaLagsTransformer', gene_index=11, **params)

