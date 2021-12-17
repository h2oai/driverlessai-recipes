"""Custom Individual 0 from Experiment test_time_series_walmart_nolimits_0c26b_5c79_bibibebo """

from h2oaicore.ga import CustomIndividual
class Indivtesttimeserieswalmartnolimits0c26b5c79bibibebo_finalFalse_id0(CustomIndividual):
    """ 
    Custom wrapper class used to construct DAI Individual

    _params_valid: dict: items that can be filled for individual-level control of parameters (as opposed to experiment-level)
                         If not set (i.e. not passed in self.params), then new experiment's value is used
                         Many of these parameters match experiment dials or are like expert tomls with a similar name
                         Dict keys are paramters
                         Dict values are the types (or values if list) for each parameter
    _from_exp: dict: parameters that are pulled from experiment-level (if value True)
 """

    def set_params(self):
        """
        
        Function to set individual-level parameters.
        If don't set any parameters, the new experiment's values are used.
        :return:
        
        """

        ###########################################################################
        #
        # BEGIN: VARIABLES ARE INFORMATIVE, NO NEED TO SET

        # Was best in population
        self.final_best = True
        # Was final population
        self.final_pop = True
        # Was in final model
        self.is_final = True

        # Score function's (hashed) name
        self.score_f_name = 'MAE'
        # Score (if is_final=True, then this is the final base model out-of-fold score)
        self.score = 2327.03955078125
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 20.137439727783203
        # Tournament Score (penalized by features counts or interpretabilty)
        self.tournament_score = 2327.03955078125
        # Score history during tuning and evolution
        self.score_list = [2456.51, 2456.510009765625]
        # Score standard deviation history during tuning and evolution
        self.score_sd_list = [549.50366, 549.503662109375]

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
        self.seed = 686428319
        # factor of extra genes added during activation
        self.default_factor = 1
        # Ensemble level
        self.ensemble_level = 0
        #
        # END: VARIABLES ARE INFORMATIVE, NO NEED TO SET
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: PARAMETERS SET FOR CUSTOM INDIVIDUAL, MAY BE SET
        #
        # Explanation of entries in self.params
        self._params_doc = {'accuracy': 'accuracy dial',
                            'config_dict': 'dictionary of config toml items (not currently used)',
                            'do_te': "Whether to support target encoding (TE) (True, False, 'only', "
                                     "'catlabel')\n"
                                     "True means can do TE, False means cannot do TE, 'only' means only "
                                     'have TE\n'
                                     "'catlabel' is special mode for LightGBM categorical handling, to "
                                     'only use that categorical handling',
                            'explore_anneal_factor': 'Explore anneal factor',
                            'explore_model_anneal_factor': 'Explore anneal factor for models',
                            'explore_model_prob': 'Explore Probability for models\n'
                                                  'Exploration vs. Exploitation of Genetic Algorithm '
                                                  'model hyperparameter is controlled via\n'
                                                  'explore_model_prob = max(explore_model_prob_lowest, '
                                                  'explore_model_prob * explore_model_anneal_factor)',
                            'explore_model_prob_lowest': 'Lowest explore probability for models',
                            'explore_prob': 'Explore Probability\n'
                                            'Exploration vs. Exploitation of Genetic Algorithm feature '
                                            'exploration is controlled via\n'
                                            'explore_prob = max(explore_prob_lowest, explore_prob * '
                                            'explore_anneal_factor)',
                            'explore_prob_lowest': 'Lowest explore probability',
                            'grow_anneal_factor': 'Annealing factor for growth',
                            'grow_prob': 'Probability to grow genome\n'
                                         'Fast growth of many genes at once is controlled by chance\n'
                                         'grow_prob = max(grow_prob_lowest, grow_prob * '
                                         'grow_anneal_factor)',
                            'grow_prob_lowest': 'Lowest growth probability',
                            'interpretability': 'interpretability dial',
                            'model_params': 'model parameters, not in self.params but as separate item',
                            'nfeatures_max': 'maximum number of features',
                            'nfeatures_min': 'minimum number of features',
                            'ngenes_max': 'maximum number of genes',
                            'ngenes_min': 'minimum number of genes',
                            'num_as_cat': 'whether to treat numeric as categorical',
                            'output_features_to_drop_more': 'list of features to drop from overall genome '
                                                            'output',
                            'random_state': 'random seed for individual',
                            'target_transformer': 'target transformer, not in self.params but as separate '
                                                  'item',
                            'time_tolerance': 'time dial'}
        #
        # Valid types for self.params
        self._params_valid = {'accuracy': 'int',
                              'config_dict': 'dict',
                              'do_te': "[True, False, 'only', 'catlabel']",
                              'explore_anneal_factor': 'float',
                              'explore_model_anneal_factor': 'float',
                              'explore_model_prob': 'float',
                              'explore_model_prob_lowest': 'float',
                              'explore_prob': 'float',
                              'explore_prob_lowest': 'float',
                              'grow_anneal_factor': 'float',
                              'grow_prob': 'float',
                              'grow_prob_lowest': 'float',
                              'interpretability': 'int',
                              'model_params': 'dict',
                              'nfeatures_max': 'int',
                              'nfeatures_min': 'int',
                              'ngenes_max': 'int',
                              'ngenes_min': 'int',
                              'num_as_cat': 'bool',
                              'output_features_to_drop_more': 'list',
                              'random_state': 'int',
                              'target_transformer': 'None',
                              'time_tolerance': 'int'}
        #
        # Parameters that may be set
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
                       'num_as_cat': True,
                       'output_features_to_drop_more': ['5_Date:Date~get_weekday',
                                                        '5_Date:Date~get_weekday'],
                       'random_state': 686428319,
                       'time_tolerance': 1}
        #
        # END: PARAMETERS SET FOR CUSTOM INDIVIDUAL, MAY BE SET
        #
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: CONTROL IF SOME EXPERIMENT PARAMETERS COME FROM EXPERIMENT (True) OR CustomIndividual (False), MAY BE SET
        #
        self._from_exp_doc = """ 
                    "_from_exp" dictionary have keys as things that will be set from the experiment (True),
                      which then overwrites the custom individual values assigned to self. of False means use custom individual value.
                     Or "_from_exp" values can be forced to come from the self attributes in the CustomIndividual (False).
                     * False is a reasonable possible option for key 'columns', to ensure the exact column types one desires are used
                       regardless of experiment-level column types.
                     * False is default for 'seed' and 'default_factor' to reproduce individual fitting behavior as closely as possible
                       even if reproducible is not set.
                     * False is not currently supported except for 'columns', 'seed', 'default_factor'.
                     One can override the static var value in the constructor or any function call before _from_exp is actually used
                     when calling make_indiv.
 """

        self._from_exp = {'cardinality_dict': True,
                          'columns': True,
                          'default_factor': False,
                          'ensemble_level': True,
                          'imbalance_ratio': True,
                          'label_counts': True,
                          'labels': True,
                          'num_classes': True,
                          'num_validation_splits': True,
                          'score_f': True,
                          'seed': False,
                          'target': True,
                          'target_transformer': True,
                          'time_column': True,
                          'train_shape': True,
                          'tsgi': True,
                          'valid_shape': True,
                          'weight_column': True}
        #
        # END: CONTROL IF SOME EXPERIMENT PARAMETERS COME FROM EXPERIMENT (True) OR CustomIndividual (False), MAY BE SET
        #
        ###########################################################################

    def set_model(self):
        """
        
        Function to set model and its parameters
        :return:
        
        """

        ###########################################################################
        #
        # MODEL TYPE, MUST BE SET
        #
        # Display name corresponds to hashed (for custom recipes) display names to ensure exact match
        # One can also provide short names if only one recipe
        self.model_display_name = 'XGBoostGBM'

        ###########################################################################
        #
        # MODEL PARAMETERS, MUST BE SET
        #
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
        self.model_params = {'base_score': 7697.503255208333,
                             'booster': 'gbtree',
                             'colsample_bytree': 0.8000000000000002,
                             'debug_verbose': 0,
                             'disable_gpus': False,
                             'early_stopping_rounds': 20,
                             'enable_early_stopping_rounds': True,
                             'eval_metric': 'mae',
                             'gamma': 0.0,
                             'gpu_id': 0,
                             'grow_policy': 'depthwise',
                             'importance_type': 'gain',
                             'label_counts': None,
                             'learning_rate': 0.15,
                             'max_bin': 256,
                             'max_delta_step': 0.0,
                             'max_depth': 8,
                             'max_leaves': 256,
                             'min_child_weight': 1,
                             'monotonicity_constraints': False,
                             'n_estimators': 500,
                             'n_jobs': 8,
                             'nthread': 8,
                             'num_class': 1,
                             'objective': 'reg:squarederror',
                             'reg_alpha': 0.0,
                             'reg_lambda': 0.0,
                             'scale_pos_weight': 1.0,
                             'score_f_name': 'MAE',
                             'seed': 686428319,
                             'silent': 1,
                             'subsample': 0.6999999999999998,
                             'tree_method': 'gpu_hist'}

        ###########################################################################
        #
        # ADJUST FINAL GBM PARAMETERS, MAY BE SET
        #
        # A list of model hyperparameters to adjust back to defaults for GA or final model building
        #  If empty list, then no changes to model parameters will be made
        #  For each item in list, set_default_params() will be used to fill those parameters for GA
        #  If _is_gbm=True for the class, then these parameters also will be changed for the final model based upon DAI dails
        #  _is_gbm = True is set for model_classes based upon LightGBM, XGBoost, CatBoost, etc.
        #   E.g. for _is_gbm=True these will be changed:
        #    * learning_rate
        #    * early_stopping_rounds
        #    * n_estimators (_fit_by_iteration in general if not None, if _fit_by_iteration=True),
        self.adjusted_params = ['learning_rate', 'early_stopping_rounds', 'n_estimators']

        ###########################################################################
        #
        # MODEL ORIGIN, VARIABLE IS INFORMATIVE, NO NEED TO SET
        #
        self.model_origin = 'FINAL MODEL'


    def set_target_transformer(self):
        """
        
        Function to set target transformer.
        If don't set any target transformer, the new experiment's values are used.  E.g. this is valid for classification.
        self.target_transformer_name = "None" applies to classification
        self.target_transformer_params = {} applies to non-time-series target transformers, only for informative purposes
        :return:
        
        """

        ###########################################################################
        #
        # TARGET TRANSFORMER, MAY BE SET
        #
        # The target-transformer name is controlled here for non-time-series cases
        # For time-series cases, the config toml choices still control outcome
        self.target_transformer_name = 'TargetTransformer_identity_noclip'

        ###########################################################################
        #
        # TARGET TRANSFORMER PARAMETERS, MAY BE SET
        #
        # Target transformer parameters are only for informative purposes for time-series,
        #  for which the target transformer is re-generated from experiment settings and config toml,
        #  if a time-series-based target transformation
        self.target_transformer_params = {}

    def set_genes(self):
        """
        
        Function to set genes/transformers
        :return:
        
        """

        import numpy as np
        nan = np.nan
        from collections import OrderedDict, defaultdict

        ###########################################################################
        #
        # ORIGINAL VARIABLE IMPORTANCE, VARIABLE IS INFORMATIVE, NO NEED TO SET
        #
        self.importances_orig = {'CPI': 0.0,
                                 'Date': 0.34263766442855115,
                                 'Dept': 0.3163970678319743,
                                 'Fuel_Price': 0.0,
                                 'IsHoliday': 0.024568199907500213,
                                 'MarkDown1': 0.0,
                                 'MarkDown2': 0.0,
                                 'MarkDown3': 0.0,
                                 'MarkDown4': 0.0,
                                 'MarkDown5': 0.0,
                                 'Store': 0.3163970678319743,
                                 'Temperature': 0.0,
                                 'Unemployment': 0.0}

        ###########################################################################
        #
        # COLUMN TYPES, CAN BE SET
        #
        # By default self._from_exp['columns'] = True and so this is only informative
        # If set self._from_exp['columns'] = False, then the below is used
        # This allows one to control the data types for each column
        self.columns = {'all': ['Date', 'Dept', 'IsHoliday', 'Store'],
                        'any': ['Date', 'Dept', 'IsHoliday', 'Store'],
                        'categorical': ['IsHoliday'],
                        'catlabel': ['IsHoliday'],
                        'date': ['Date'],
                        'datetime': [],
                        'id': [],
                        'image': [],
                        'numeric': ['IsHoliday'],
                        'ohe_categorical': ['IsHoliday'],
                        'raw': [],
                        'text': [],
                        'time_column': ['Date']}

        ###########################################################################
        #
        # GENOME, MUST BE SET
        #
        # All valid parameters for genes should be provided, except:
        # * output_features_to_drop need not be passed if empty list
        # * Mutations need not be provided if want to use default values
        # Mutations or valid parameters are not shown if none, like for OriginalTransformer
        # 'gene_index' is optional, except if use:
        # *) transformed feature names in (e.g.) monotonicity_constraints_dict toml
        # *) multiple layers with specific col_dict per layer for layer > 0
        # * 'col_type' argument to add_transformer() is used in some cases to get unique DAI transformer,
        #  and it does not need to be changed or set independently of the transformer in most cases
        # * 'labels' parameter, if present in valid parameters, is handled internally by DAI and does not need to be set
        # NOTE: While some importance variable data is provided, the newly-generated individual has freshly-determined importances

        # Gene Normalized Importance: 0.026241
        # Transformed Feature Names and Importances: {'0_IsHoliday': 0.026240913197398186}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['IsHoliday'], 'random_state': 686428319}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=0, **params)

        # Gene Normalized Importance: 0.024008
        # Transformed Feature Names and Importances: {'1_Freq:IsHoliday': 0.024008311331272125}
        # Valid parameters: ['cat_cols', 'norm', 'random_state', 'output_features_to_drop', 'labels']
        # Allowed parameters and mutations (first mutation in list is default): {'norm': [True, False]}
        params = {'cat_cols': ['IsHoliday'], 'norm': True, 'random_state': 686428320}
        self.add_transformer('FrequentTransformer', col_type='categorical', gene_index=1, **params)

        # Gene Normalized Importance: 0.0016893
        # Transformed Feature Names and Importances: {'2_Date~is_holiday_UnitedStates': 0.0016893190331757069}
        # Valid parameters: ['dt_cols', 'country', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # Allowed parameters and mutations (first mutation in list is default): {'country': ['UnitedStates', 'UnitedKingdom', 'EuropeanCentralBank', 'Germany', 'Mexico', 'Japan']}
        params = {'country': 'UnitedStates',
                  'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dt_cols': ['Date'],
                  'random_state': 686428324}
        self.add_transformer('IsHolidayTransformer', col_type='date', gene_index=2, **params)

        # Gene Normalized Importance: 0.89262
        # Transformed Feature Names and Importances: {'3_TargetLag:Date:Dept:Store.40': 0.28449615836143494, '3_TargetLag:Date:Dept:Store.42': 0.17997677624225616, '3_TargetLag:Date:Dept:Store.52': 0.12446405738592148, '3_TargetLag:Date:Dept:Store.48': 0.061807405203580856, '3_TargetLag:Date:Dept:Store.39': 0.04744958132505417, '3_TargetLag:Date:Dept:Store.47': 0.04050706326961517, '3_TargetLag:Date:Dept:Store.46': 0.02560368739068508, '3_TargetLag:Date:Dept:Store.43': 0.023364415392279625, '3_TargetLag:Date:Dept:Store.53': 0.020446978509426117, '3_TargetLag:Date:Dept:Store.41': 0.017883583903312683, '3_TargetLag:Date:Dept:Store.44': 0.015429142862558365, '3_TargetLag:Date:Dept:Store.51': 0.013844389468431473, '3_TargetLag:Date:Dept:Store.60': 0.011348702944815159, '3_TargetLag:Date:Dept:Store.45': 0.00788742583245039, '3_TargetLag:Date:Dept:Store.49': 0.006239354610443115, '3_TargetLag:Date:Dept:Store.71': 0.0051182946190238, '3_TargetLag:Date:Dept:Store.50': 0.004495012573897839, '3_TargetLag:Date:Dept:Store.59': 0.0022598537616431713}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
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
                  'random_state': 686428327,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsTransformer', col_type='time_column', gene_index=3, **params)

        # Gene Normalized Importance: 0.05198
        # Transformed Feature Names and Importances: {'5_Date:Date~get_year': 0.017476584762334824, '5_Date:Date~get_dayofyear': 0.011599421501159668, '5_Date:Date~get_week': 0.009281093254685402, '5_Date:Date~get_day': 0.006530100479722023, '5_Date:Date~get_month': 0.0058868080377578735, '5_Date:Date~get_quarter': 0.001206442597322166}
        # Valid parameters: ['dt_cols', 'funcs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
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
                  'output_features_to_drop': ['5_Date:Date~get_weekday'],
                  'random_state': 790032779}
        self.add_transformer('DatesTransformer', col_type='date', gene_index=5, **params)

        # Gene Normalized Importance:  1.0488
        # Transformed Feature Names and Importances: {'7_EWMA(0.05)(0)TargetLags:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 1.0, '7_EWMA(0.05)(1)TargetLags:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.024692440405488014, '7_EWMA(0.05)(2)TargetLags:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.024062011390924454}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'alpha', 'orders', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # 'encoder' parameter is handled internally by DAI
        # 'dropout' options (keys, same order as toml 'lags_dropout'): {'DISABLED': 0, 'DEPENDENT': 1, 'INDEPENDENT': 2}
        # Allowed parameters and mutations (first mutation in list is default): {'alpha': [0.05, 0, 0.1, 0.5, 0.9]}
        params = {'alpha': 0.05,
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
                  'orders': [0, 1, 2],
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 888991948,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('EwmaLagsTransformer', col_type='time_column', gene_index=7, **params)


        ###########################################################################
        #
        # TIME SERIES GROUP INFO, VARIABLES ARE FOR ACCEPTANCE TESTING ONLY, NO NEED TO SET
        #
        from h2oaicore.timeseries_support import LagTimeSeriesGeneInfo, NonLagTimeSeriesGeneInfo, \
            NonTimeSeriesGeneInfo, EitherTimeSeriesGeneInfoBase
        from h2oaicore.timeseries_support import DateTimeLabelEncoder
        from h2oaicore.timeseries_support import TimeSeriesProperties

        # Note: tsgi will use tsp and encoder, and tsp will use encoder
        self.tsgi_params = {'_build_info': {'commit': '50922b6', 'version': '1.10.2'},
                           'date_format_strings': {'Date': '%Y-%m-%d'},
                           'encoder': None,
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
                           'non_ufapt_lag_sizes': None,
                           'pred_gap': 0,
                           'pred_periods': 39,
                           'target': 'Weekly_Sales',
                           'target_trafo_lag': None,
                           'tgc': ['Date', 'Dept', 'Store'],
                           'time_column': 'Date',
                           'tsp': None,
                           'ufapt': [],
                           'ufapt_lag_sizes': None}
        self.tsgi = LagTimeSeriesGeneInfo(**self.tsgi_params)

        self.tsp_params = {'_Hz': 1.6534391534391533e-06,
                          '_datetime_format': '%Y-%m-%d',
                          '_dtlabel_enc': None,
                          '_freq_lookup_table': {'EHz': {0: 1e+18},
                                                 'GHz': {0: 1000000000.0},
                                                 'Hz': {0: 1.0},
                                                 'MHz': {0: 1000000.0},
                                                 'PHz': {0: 1000000000000000.0},
                                                 'THz': {0: 1000000000000.0},
                                                 'YHz': {0: 1e+24},
                                                 'ZHz': {0: 1e+21},
                                                 'aHz': {0: 1e-18},
                                                 'fHz': {0: 1e-15},
                                                 'kHz': {0: 1000.0},
                                                 'mHz': {0: 0.001},
                                                 'nHz': {0: 1e-09},
                                                 'pHz': {0: 1e-12},
                                                 'yHz': {0: 1e-24},
                                                 'zHz': {0: 1e-21},
                                                 'µHz': {0: 1e-06}},
                          '_freq_si': 'µHz',
                          '_frequency': 1.6534391534391534e-15,
                          '_lags_ranking': None,
                          '_logger': None,
                          '_n_samples_per_period': 2948.041958041958,
                          '_period': 604800000000000.0,
                          '_period_lookup_table': {'d': {0: 86400.0},
                                                   'h': {0: 3600.0},
                                                   'min': {0: 60.0},
                                                   'mo': {0: 2678000.0},
                                                   'ms': {0: 0.001},
                                                   'ns': {0: 1e-09},
                                                   'qr': {0: 7884000.0},
                                                   's': {0: 1.0},
                                                   'wk': {0: 604800.0},
                                                   'y': {0: 31557600.0},
                                                   'µs': {0: 1e-06}},
                          '_period_si': 'wk',
                          '_resolution_lookup_table': {'d': {0: 86400000000000.0},
                                                       'h': {0: 3600000000000.0},
                                                       'min': {0: 60000000000.0},
                                                       'mo': {0: 2628000000000000.0},
                                                       'ms': {0: 1000000.0},
                                                       'ns': {0: 1.0},
                                                       'qr': {0: 7884000000000000.0},
                                                       's': {0: 1000000000.0},
                                                       'wk': {0: 604800000000000.0},
                                                       'y': {0: 3.15576e+16},
                                                       'µs': {0: 1000.0}},
                          '_target_column': '__TARGET__',
                          '_tdelta_max': 604800000000000,
                          '_tdelta_mean': 604800000000000.0,
                          '_tdelta_median': 604800000000000.0,
                          '_tdelta_min': 604800000000000,
                          '_tdelta_mode': 604800000000000,
                          '_te_periods': 39,
                          '_tgc': ['Date', 'Dept', 'Store'],
                          '_tgc_autodet': False,
                          '_time_column': 'Date',
                          '_time_resolution_si': 'wk',
                          '_tr_periods': 143,
                          '_tr_samples': 421570,
                          '_tr_te_gap': 0,
                          '_tvs': None}
        self.tsp = TimeSeriesProperties(**self.tsp_params)

        self.encoder_params = {'_bad_base_attrs': [],
                              '_base_attrs': ['_bad_base_attrs',
                                              '_build_info',
                                              '_initialized',
                                              '_fitted',
                                              '_input_feature_names',
                                              'target',
                                              'lag_feature',
                                              '_output_feature_names',
                                              '_fitted_output_feature_names',
                                              '_output_feature_names_backup',
                                              '_feature_desc',
                                              '_output_feature_extra_descriptions',
                                              '_full_feature_desc',
                                              '_output_features_to_drop',
                                              '_datetime_formats',
                                              'labels',
                                              'random_state',
                                              'bad_fit',
                                              '_fit_exception',
                                              '_imputations',
                                              '_imputation_transformers',
                                              'dummy_tr'],
                              '_build_info': {'commit': '50922b6', 'version': '1.10.2'},
                              '_datetime_formats': {'Date': '%Y-%m-%d'},
                              '_feature_desc': None,
                              '_fit_exception': None,
                              '_fitted': False,
                              '_fitted_output_feature_names': None,
                              '_full_feature_desc': None,
                              '_imputation_transformers': {},
                              '_imputations': None,
                              '_initialized': True,
                              '_input_feature_names': ['Date'],
                              '_out_types': {},
                              '_output_feature_extra_descriptions': None,
                              '_output_feature_names': None,
                              '_output_feature_names_backup': None,
                              '_output_feature_names_for_estimate': ['Date'],
                              '_output_features_to_drop': [],
                              'bad_fit': False,
                              'copy': True,
                              'dummy_tr': None,
                              'labels': None,
                              'lag_feature': None,
                              'monthly': False,
                              'outer_cv': None,
                              'period': 604800000000000.0,
                              'random_state': 42,
                              'target': None,
                              'te': 1374796800000000000,
                              'time_column': 'Date',
                              'ts': 1265328000000000000,
                              'weekdays_only': False}
        self.encoder = DateTimeLabelEncoder(**self.encoder_params)

    @staticmethod
    def is_enabled():
        """Return whether recipe is enabled. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def do_acceptance_test():
        """
        Return whether to do acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests try to make internal DAI individual out of the python code
        """
        return True

    @staticmethod
    def acceptance_test_timeout():
        """
        Timeout in minutes for each test of a custom recipe.
        """
        return config.acceptance_test_timeout
