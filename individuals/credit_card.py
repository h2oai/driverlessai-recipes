"""Custom Individual 0 from Experiment test_credit_card_c6b98_4367_ficulahe """

from h2oaicore.ga import CustomIndividual
class Indivtestcreditcardc6b984367ficulahe_finalFalse_id0(CustomIndividual):
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
        self.score_f_name = 'AUC'
        # Score (if is_final=True, then this is the final base model out-of-fold score)
        self.score = 0.7179035816201912
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 0.0060064196797443145
        # Tournament Score (penalized by features counts or interpretabilty)
        self.tournament_score = 0.7520820488964624
        # Score history during tuning and evolution
        self.score_list = [0.7520820488964624, 0.7520820488964624]
        # Score standard deviation history during tuning and evolution
        self.score_sd_list = [0.0069880164299224575, 0.0069880164299224575]

        # Number of classes if supervised
        self.num_classes = 2
        # Labels if classification, None for regression
        self.labels = [0, 1]

        # Shape of training frame (may include target)
        self.train_shape = (23999, 27)
        # Shape of validation frame (may include target)
        self.valid_shape = None
        # Cardinality for each column
        self.cardinality_dict = {'AGE': 55,
                                 'EDUCATION': 7,
                                 'LIMIT_BAL': 79,
                                 'MARRIAGE': 4,
                                 'PAY_0': 11,
                                 'PAY_2': 11,
                                 'PAY_3': 11,
                                 'PAY_4': 11,
                                 'PAY_5': 10,
                                 'PAY_6': 10,
                                 'SEX': 2}

        # Target column
        self.target = 'default payment next month'
        # Label counts for target column
        self.label_counts = [18630.0, 5369.0]
        # Imbalanced ratio
        self.imbalance_ratio = 3.469919910597877

        # Weight column
        self.weight_column = None
        # Time column
        self.time_column = None

        # Number of validation splits
        self.num_validation_splits = 1
        # Seed for individual
        self.seed = 159699529
        # factor of extra genes added during activation
        self.default_factor = 1
        # Ensemble level
        self.ensemble_level = 1
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
                       'num_as_cat': False,
                       'output_features_to_drop_more': [],
                       'random_state': 159699529,
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
        self.model_display_name = 'LightGBM'

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
        self.model_params = {'bagging_seed': 159699531,
                             'booster': 'lightgbm',
                             'boosting_type': 'gbdt',
                             'categorical_feature': '',
                             'class_weight': None,
                             'colsample_bytree': 0.8000000000000002,
                             'deterministic': True,
                             'device_type': 'cpu',
                             'disable_gpus': True,
                             'early_stopping_rounds': 1,
                             'early_stopping_threshold': 0.0,
                             'enable_early_stopping_rounds': True,
                             'eval_metric': 'auc',
                             'feature_fraction_seed': 159699530,
                             'gamma': 0.0,
                             'gpu_id': 0,
                             'grow_policy': 'depthwise',
                             'importance_type': 'gain',
                             'label_counts': [18630, 5369],
                             'labels': [0, 1],
                             'learning_rate': 1.0,
                             'max_bin': 249,
                             'max_delta_step': 0.0,
                             'max_depth': 8,
                             'max_leaves': 256,
                             'min_child_samples': 20,
                             'min_child_weight': 0.001,
                             'min_data_in_bin': 1,
                             'min_split_gain': 0.0,
                             'monotonicity_constraints': False,
                             'n_estimators': 3,
                             'n_gpus': 0,
                             'n_jobs': 8,
                             'num_class': 1,
                             'num_classes': 2,
                             'num_leaves': 256,
                             'num_threads': 8,
                             'objective': 'binary',
                             'random_state': 159699529,
                             'reg_alpha': 0.0,
                             'reg_lambda': 0.0,
                             'scale_pos_weight': 1.0,
                             'score_f_name': 'AUC',
                             'seed': 159699529,
                             'silent': True,
                             'subsample': 0.6999999999999998,
                             'subsample_for_bin': 200000,
                             'subsample_freq': 1,
                             'verbose': -1}

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
        self.model_origin = 'FINAL BASE MODEL 0'


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
        self.target_transformer_name = 'None'

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
        self.importances_orig = {'AGE': 0.019255970562231616,
                                 'BILL_AMT1': 0.0443496509921389,
                                 'BILL_AMT2': 0.0,
                                 'BILL_AMT3': 0.011566858223034592,
                                 'BILL_AMT4': 0.016425227663335498,
                                 'BILL_AMT5': 0.0012041895689112083,
                                 'BILL_AMT6': 0.006298225525526133,
                                 'EDUCATION': 0.0008270333447045372,
                                 'LIMIT_BAL': 0.0,
                                 'MARRIAGE': 0.0,
                                 'PAY_0': 0.5618879967374988,
                                 'PAY_2': 0.12337593049225047,
                                 'PAY_3': 0.0,
                                 'PAY_4': 0.0,
                                 'PAY_5': 0.025585774041151442,
                                 'PAY_6': 0.015153216584793714,
                                 'PAY_AMT1': 0.011485086478483272,
                                 'PAY_AMT2': 0.02411861535516454,
                                 'PAY_AMT3': 0.05860905827395108,
                                 'PAY_AMT4': 0.026853821653250766,
                                 'PAY_AMT5': 0.015062602134149883,
                                 'PAY_AMT6': 0.03629424611547213,
                                 'SEX': 0.0016464962539513958}

        ###########################################################################
        #
        # COLUMN TYPES, CAN BE SET
        #
        # By default self._from_exp['columns'] = True and so this is only informative
        # If set self._from_exp['columns'] = False, then the below is used
        # This allows one to control the data types for each column
        self.columns = {'all': ['AGE',
                                'BILL_AMT1',
                                'BILL_AMT2',
                                'BILL_AMT3',
                                'BILL_AMT4',
                                'BILL_AMT5',
                                'BILL_AMT6',
                                'EDUCATION',
                                'LIMIT_BAL',
                                'MARRIAGE',
                                'PAY_0',
                                'PAY_2',
                                'PAY_3',
                                'PAY_4',
                                'PAY_5',
                                'PAY_6',
                                'PAY_AMT1',
                                'PAY_AMT2',
                                'PAY_AMT3',
                                'PAY_AMT4',
                                'PAY_AMT5',
                                'PAY_AMT6',
                                'SEX'],
                        'any': ['AGE',
                                'BILL_AMT1',
                                'BILL_AMT2',
                                'BILL_AMT3',
                                'BILL_AMT4',
                                'BILL_AMT5',
                                'BILL_AMT6',
                                'EDUCATION',
                                'LIMIT_BAL',
                                'MARRIAGE',
                                'PAY_0',
                                'PAY_2',
                                'PAY_3',
                                'PAY_4',
                                'PAY_5',
                                'PAY_6',
                                'PAY_AMT1',
                                'PAY_AMT2',
                                'PAY_AMT3',
                                'PAY_AMT4',
                                'PAY_AMT5',
                                'PAY_AMT6',
                                'SEX'],
                        'categorical': [],
                        'catlabel': [],
                        'date': [],
                        'datetime': [],
                        'id': [],
                        'image': [],
                        'numeric': ['AGE',
                                    'BILL_AMT1',
                                    'BILL_AMT2',
                                    'BILL_AMT3',
                                    'BILL_AMT4',
                                    'BILL_AMT5',
                                    'BILL_AMT6',
                                    'EDUCATION',
                                    'LIMIT_BAL',
                                    'MARRIAGE',
                                    'PAY_0',
                                    'PAY_2',
                                    'PAY_3',
                                    'PAY_4',
                                    'PAY_5',
                                    'PAY_6',
                                    'PAY_AMT1',
                                    'PAY_AMT2',
                                    'PAY_AMT3',
                                    'PAY_AMT4',
                                    'PAY_AMT5',
                                    'PAY_AMT6',
                                    'SEX'],
                        'ohe_categorical': [],
                        'raw': [],
                        'text': [],
                        'time_column': []}

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

        # Gene Normalized Importance: 0.03427
        # Transformed Feature Names and Importances: {'0_AGE': 0.03427012264728546}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['AGE'], 'random_state': 159699529}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=0, **params)

        # Gene Normalized Importance: 0.07893
        # Transformed Feature Names and Importances: {'1_BILL_AMT1': 0.07892969995737076}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT1'], 'random_state': 159699530}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=1, **params)

        # Gene Normalized Importance: 0.020586
        # Transformed Feature Names and Importances: {'3_BILL_AMT3': 0.02058570086956024}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT3'], 'random_state': 159699532}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=3, **params)

        # Gene Normalized Importance: 0.029232
        # Transformed Feature Names and Importances: {'4_BILL_AMT4': 0.02923220954835415}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT4'], 'random_state': 159699533}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=4, **params)

        # Gene Normalized Importance: 0.0021431
        # Transformed Feature Names and Importances: {'5_BILL_AMT5': 0.002143113175407052}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT5'], 'random_state': 159699534}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=5, **params)

        # Gene Normalized Importance: 0.011209
        # Transformed Feature Names and Importances: {'6_BILL_AMT6': 0.011209040880203247}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT6'], 'random_state': 159699535}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=6, **params)

        # Gene Normalized Importance: 0.0014719
        # Transformed Feature Names and Importances: {'7_EDUCATION': 0.0014718829188495874}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['EDUCATION'], 'random_state': 159699536}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=7, **params)

        # Gene Normalized Importance:       1
        # Transformed Feature Names and Importances: {'10_PAY_0': 1.0}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_0'], 'random_state': 159699539}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=10, **params)

        # Gene Normalized Importance: 0.21957
        # Transformed Feature Names and Importances: {'11_PAY_2': 0.21957388520240784}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_2'], 'random_state': 159699540}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=11, **params)

        # Gene Normalized Importance: 0.045535
        # Transformed Feature Names and Importances: {'14_PAY_5': 0.045535363256931305}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_5'], 'random_state': 159699543}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=14, **params)

        # Gene Normalized Importance: 0.026968
        # Transformed Feature Names and Importances: {'15_PAY_6': 0.026968393474817276}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_6'], 'random_state': 159699544}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=15, **params)

        # Gene Normalized Importance: 0.02044
        # Transformed Feature Names and Importances: {'16_PAY_AMT1': 0.020440170541405678}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT1'], 'random_state': 159699545}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=16, **params)

        # Gene Normalized Importance: 0.042924
        # Transformed Feature Names and Importances: {'17_PAY_AMT2': 0.04292424023151398}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT2'], 'random_state': 159699546}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=17, **params)

        # Gene Normalized Importance: 0.10431
        # Transformed Feature Names and Importances: {'18_PAY_AMT3': 0.10430736839771271}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT3'], 'random_state': 159699547}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=18, **params)

        # Gene Normalized Importance: 0.047792
        # Transformed Feature Names and Importances: {'19_PAY_AMT4': 0.04779212549328804}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT4'], 'random_state': 159699548}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=19, **params)

        # Gene Normalized Importance: 0.026807
        # Transformed Feature Names and Importances: {'20_PAY_AMT5': 0.02680712565779686}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT5'], 'random_state': 159699549}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=20, **params)

        # Gene Normalized Importance: 0.064593
        # Transformed Feature Names and Importances: {'21_PAY_AMT6': 0.06459338217973709}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT6'], 'random_state': 159699550}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=21, **params)

        # Gene Normalized Importance: 0.0029303
        # Transformed Feature Names and Importances: {'22_SEX': 0.0029302926268428564}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['SEX'], 'random_state': 159699551}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=22, **params)


        ###########################################################################
        #
        # TIME SERIES GROUP INFO, VARIABLES ARE FOR ACCEPTANCE TESTING ONLY, NO NEED TO SET
        #
        from h2oaicore.timeseries_support import LagTimeSeriesGeneInfo, NonLagTimeSeriesGeneInfo, \
            NonTimeSeriesGeneInfo, EitherTimeSeriesGeneInfoBase
        from h2oaicore.timeseries_support import DateTimeLabelEncoder
        from h2oaicore.timeseries_support import TimeSeriesProperties

        # Note: tsgi will use tsp and encoder, and tsp will use encoder
        self.tsgi_params = {'date_format_strings': {},
                           'encoder': None,
                           'target': None,
                           'tgc': None,
                           'time_column': None,
                           'tsp': None,
                           'ufapt': []}
        self.tsgi = NonTimeSeriesGeneInfo(**self.tsgi_params)

        self.tsp_params = {}
        self.tsp = None

        self.encoder_params = {}
        self.encoder = None

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
