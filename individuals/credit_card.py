"""Custom Final Individual 0 from Experiment test_credit_card_9f03b_f55d_pawisaco """

# EXAMPLE USE CASES THAT REQUIRE MINOR MODIFICATIONS TO RECIPE:
# 1) FROZEN INDIVIDUALS: By default, the custom individual acts like a normal internal DAI individual,
#    which has its features and model hyperparameters mutated.
#    However, mutation of features and model hyperparameters can be avoided, and new features or models can be avoided.
#    This can be achieved by setting self.params values:
#    prob_perturb_xgb = prob_add_genes = prob_prune_genes = prob_prune_by_features = prob_addbest_genes = prob_prune_by_features = 0.0
#    leading to a "frozen" individual that is not mutated.
# 2) ENSEMBLING INDIVIDUALS: If all individuals in an experiment are frozen, then no tuning or evolution is performed.
#    One can set expert toml fixed_ensemble_level to the number of such individuals to include in an ensemble.

from h2oaicore.ga import CustomIndividual


class Indiv_testcreditcard9f03bf55dpawisaco_finalTrue_id0(CustomIndividual):
    """ 
    Custom wrapper class used to construct DAI Individual,
    which contains all information related to model type, model parameters, feature types, and feature parameters.

    _params_valid: dict: items that can be filled for individual-level control of parameters (as opposed to experiment-level)
                         If not set (i.e. not passed in self.params), then new experiment's value is used
                         Many of these parameters match experiment dials or are like expert tomls with a similar name
                         Dict keys are paramters
                         Dict values are the types (or values if list) for each parameter
    _from_exp: dict: parameters that are pulled from experiment-level (if value True)
 """

    """ 
    Individual instances contain structures and methods for:
        1) Feature data science types (col_dict)
        2) Transformers and their parameters (genomes, one for each layer in multi-layer feature pipeline)
        3) Model hyperparameters
        4) Feature importances and description tables
        5) Mutation control parameters
        An individual is the basis of a Population.
        Each individual is a unit, separate from the experiment that created it, that is stored (on disk) as a "brain" artifact,
        which can be re-used by new experiments if certain matching conditions are met.
        A Custom Individual is a class instance used to generate an Individual.
 """

    ###########################################################################
    #
    # Type of Individual and Origin of Recipe
    _regression = False
    _binary = True
    _multiclass = False
    _unsupervised = False
    _description = 'Indiv_testcreditcard9f03bf55dpawisaco_finalTrue_id0'
    _display_name = 'Indiv_testcreditcard9f03bf55dpawisaco_finalTrue_id0'

    # Original Experiment ID
    _experiment_id_origin = 'test_credit_card_9f03b_f55d_pawisaco'
    # Original Experiment Description
    _experiment_description_origin = 'test_credit_card_9f03b_f55d_pawisaco'

    def set_params(self):
        """
        
        Function to set individual-level parameters.
        If don't set any parameters, the new experiment's values are used.
        :return:
        
        """

        ###########################################################################
        #
        # BEGIN: VARIABLES ARE INFORMATIVE, MUST BE KEPT FOR ACCEPTANCE TESTING

        # Was best in population
        self.final_best = True
        # Was final population
        self.final_pop = True
        # Was in final model
        self.is_final = True

        # Which individual by hash
        self.hash = 'e5e57a23-7375-4f05-9951-71c985c2a49a'
        # Which parent individual by hash
        self.parent_hash = None

        # Score function's (hashed) name
        self.score_f_name = 'AUC'
        # Score (if is_final=True, then this is the final base model out-of-fold score)
        self.score = 0.7179035816201912
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 0.0060064196797443145
        # Tournament Score (penalized by features counts or interpretability)
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
        # END: VARIABLES ARE INFORMATIVE, MUST BE KEPT FOR ACCEPTANCE TESTING
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: PARAMETERS SET FOR CUSTOM INDIVIDUAL, self.params MAY BE SET
        #
        # Explanation of entries in self.params
        self._params_doc = {'accuracy': 'accuracy dial',
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
                            'nfeatures_max': 'maximum number of features',
                            'nfeatures_min': 'minimum number of features',
                            'ngenes_max': 'maximum number of genes',
                            'ngenes_min': 'minimum number of genes',
                            'num_as_cat': 'whether to treat numeric as categorical',
                            'output_features_to_drop_more': 'list of features to drop from overall genome '
                                                            'output',
                            'prob_add_genes': 'Unnormalized probability to add genes',
                            'prob_addbest_genes': 'Unnormalized probability to add best genes',
                            'prob_perturb_xgb': 'Unnormalized probability to change model hyperparameters',
                            'prob_prune_by_features': 'Unnormalized probability to prune features',
                            'prob_prune_genes': 'Unnormalized probability to prune genes',
                            'random_state': 'random seed for individual',
                            'time_tolerance': 'time dial'}
        #
        # Valid types for self.params
        self._params_valid = {'accuracy': 'int',
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
                              'nfeatures_max': 'int',
                              'nfeatures_min': 'int',
                              'ngenes_max': 'int',
                              'ngenes_min': 'int',
                              'num_as_cat': 'bool',
                              'output_features_to_drop_more': 'list',
                              'prob_add_genes': 'float',
                              'prob_addbest_genes': 'float',
                              'prob_perturb_xgb': 'float',
                              'prob_prune_by_features': 'float',
                              'prob_prune_genes': 'float',
                              'random_state': 'int',
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
                       'prob_add_genes': 0.5,
                       'prob_addbest_genes': 0.5,
                       'prob_perturb_xgb': 0.25,
                       'prob_prune_by_features': 0.25,
                       'prob_prune_genes': 0.5,
                       'random_state': 159699529,
                       'time_tolerance': 1}
        #
        # END: PARAMETERS SET FOR CUSTOM INDIVIDUAL, MAY BE SET
        #
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: CONTROL IF PARAMETERS COME FROM EXPERIMENT (True) OR CustomIndividual (False), self._from_exp MAY BE SET
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
        #
        # The values of _from_exp are persisted for this individual when doing refit/restart of experiments
        #
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
        # END: CONTROL IF PARAMETERS COME FROM EXPERIMENT (True) OR CustomIndividual (False), self._from_exp MAY BE SET
        #
        ###########################################################################

        ###########################################################################
        #
        # BEGIN: CONTROL SOME VALUES IN THE CONFIG.TOML FILE (AND EXPERT SETTINGS), MAY BE SET
        #
        # config_dicts are python dictionary of config.toml keys and values that should be loadable with toml.loads()
        # Tomls appear in auto-generated code only if different than DAI factory defaults.
        #
        # Any tomls placed into self.config_dict will be enforced for the entire experiment.
        # Some config tomls like time_series_causal_split_recipe must be set for acceptance testing to pass
        # if experiment ran with time_series_causal_split_recipe=false
        # Tomls added to this list in auto-generated code may be required to be set for the individual to function properly,
        # and any experiment-level tomls can be added here to control the experiment independent from
        # config.toml or experiment expert settings.
        #
        self.config_dict = {}
        #
        # self.config_dict_individual contains tomls that may be requried for the individual to behave correctly.
        # self.config_dict are applied at experiment level, while self.config_dict_individual are not.
        # E.g. monotonicity_constraints_dict can be addeed to self.config_dict_individual to only control
        # this individual's transformed features' monotonicity.
        # One can set cols_to_force_in and cols_to_force_in_sanitized to force in a feature at the experiment or individual level,
        # or one can pass force=True to the entire gene in add_transformer() below in set_genes()
        #
        self.config_dict_individual = {'glm_optimal_refit': False,
                                       'included_transformers': ['AutovizRecommendationsTransformer',
                                                                 'BERTTransformer',
                                                                 'CVCatNumEncodeTransformer',
                                                                 'CVTECUMLTransformer',
                                                                 'CVTargetEncodeTransformer',
                                                                 'CatOriginalTransformer',
                                                                 'CatTransformer',
                                                                 'ClusterDistCUMLDaskTransformer',
                                                                 'ClusterDistCUMLTransformer',
                                                                 'ClusterDistTransformer',
                                                                 'ClusterIdAllNumTransformer',
                                                                 'ClusterTETransformer',
                                                                 'DBSCANCUMLDaskTransformer',
                                                                 'DBSCANCUMLTransformer',
                                                                 'DateOriginalTransformer',
                                                                 'DateTimeDiffTransformer',
                                                                 'DateTimeOriginalTransformer',
                                                                 'DatesTransformer',
                                                                 'EwmaLagsTransformer',
                                                                 'FrequentTransformer',
                                                                 'ImageOriginalTransformer',
                                                                 'ImageVectorizerTransformer',
                                                                 'InteractionsTransformer',
                                                                 'IsHolidayTransformer',
                                                                 'IsolationForestAnomalyAllNumericTransformer',
                                                                 'IsolationForestAnomalyNumCatAllColsTransformer',
                                                                 'IsolationForestAnomalyNumCatTransformer',
                                                                 'IsolationForestAnomalyNumericTransformer',
                                                                 'LagsAggregatesTransformer',
                                                                 'LagsInteractionTransformer',
                                                                 'LagsTransformer',
                                                                 'LexiLabelEncoderTransformer',
                                                                 'MeanTargetTransformer',
                                                                 'NumCatTETransformer',
                                                                 'NumToCatTETransformer',
                                                                 'NumToCatWoEMonotonicTransformer',
                                                                 'NumToCatWoETransformer',
                                                                 'OneHotEncodingTransformer',
                                                                 'OriginalTransformer',
                                                                 'RawTransformer',
                                                                 'StandardScalerTransformer',
                                                                 'StringConcatTransformer',
                                                                 'TSNECUMLTransformer',
                                                                 #'TextBiGRUTransformer',
                                                                 #'TextCNNTransformer',
                                                                 #'TextCharCNNTransformer',
                                                                 'TextLinModelTransformer',
                                                                 'TextOriginalTransformer',
                                                                 'TextTransformer',
                                                                 'TimeSeriesTargetEncTransformer',
                                                                 'TruncSVDAllNumTransformer',
                                                                 'TruncSVDCUMLDaskTransformer',
                                                                 'TruncSVDCUMLTransformer',
                                                                 'TruncSVDNumTransformer',
                                                                 'UMAPCUMLDaskTransformer',
                                                                 'UMAPCUMLTransformer',
                                                                 'WeightOfEvidenceTransformer'],
                                       'max_epochs': 1,
                                       'one_hot_encoding_cardinality_threshold': 11,
                                       'prob_default_lags': 0.2,
                                       'prob_lag_non_targets': 0.1,
                                       'prob_lagsaggregates': 0.2,
                                       'prob_lagsinteraction': 0.2}
                                       #'tensorflow_max_epochs_nlp': 1}
        #
        # Some transformers and models may be inconsistent with experiment's config.toml or expert config toml state,
        # such as OHE for LightGBM when config.enable_one_hot_encoding in ['auto', 'off'], yet have no other adverse effect.
        # Leaving the default of self.enforce_experiment_config=False
        # will allow features and models even if they were disabled in experiment settings.
        # This avoid hassle of setting experiment config tomls to enable transformers used by custom individual.
        # Also, if False, then self.config_dict_individual are applied for this custom individual when performing
        # operations on just the individual related to feature types, feature parameters, model types, or model parameters.
        # E.g. if enable_lightgbm_cat_support was True when individual was made, that is set again for the
        # particular custom individual even if experiment using the custom individual did not set it.
        # Setting below to self.enforce_experiment_config=True will enforce consistency checks on custom individual,
        # pruning inconsistent models or transformers according to the experiment's config toml settings.
        # The value of enforce_experiment_config is persisted for this individual,
        # when doing refit/restart of experiments.
        #
        self.enforce_experiment_config = False
        #
        # Optional config toml items that are allowed and maybe useful to control.
        # By default, these are ignored, so that config.toml or experiment expert settings can control them.
        # However, these are tomls one can manually copy over to self.config_dict to enforce certain experiment-level values,
        # regardless of experiment settings.
        # Included lists like included_models and included_scorers will be populated with all allowed values,
        # if no changes to defaults were made.
        #
        self.config_dict_experiment = {'drop_redundant_columns_limit': 0,
                                       'dummy': 1,
                                       'enable_funnel': False,
                                       'feature_brain_level': 0,
                                       'hard_asserts': True,
                                       'included_models': ['Constant',
                                                           'DecisionTree',
                                                           'FTRL',
                                                           'GLM',
                                                           'ImageAuto',
                                                           'ImbalancedLightGBM',
                                                           'ImbalancedXGBoostGBM',
                                                           'LightGBM',
                                                           'LightGBMDask',
                                                           'RFCUML',
                                                           'RFCUMLDask',
                                                           'RuleFit',
                                                           #'TensorFlow',
                                                           'TextALBERT',
                                                           'TextBERT',
                                                           'TextCamemBERT',
                                                           'TextDistilBERT',
                                                           'TextMultilingualBERT',
                                                           'TextRoBERTa',
                                                           'TextXLM',
                                                           'TextXLMRoberta',
                                                           'TextXLNET',
                                                           'TorchGrowNet',
                                                           'XGBoostDart',
                                                           'XGBoostDartDask',
                                                           'XGBoostGBM',
                                                           'XGBoostGBMDask',
                                                           'XGBoostRF',
                                                           'XGBoostRFDask'],
                                       'included_scorers': ['ACCURACY',
                                                            'AUC',
                                                            'AUCPR',
                                                            'F05',
                                                            'F1',
                                                            'F2',
                                                            'FDR',
                                                            'FNR',
                                                            'FOR',
                                                            'FPR',
                                                            'GINI',
                                                            'LOGLOSS',
                                                            'MACROAUC',
                                                            'MACROF1',
                                                            'MACROMCC',
                                                            'MCC',
                                                            'NPV',
                                                            'PRECISION',
                                                            'RECALL',
                                                            'TNR'],
                                       'num_gpus_per_experiment': 0,
                                       'num_gpus_per_model': 0,
                                       'reproducible': True,
                                       'stalled_time_kill_ref': 10000.0,
                                       'stalled_time_min': 10000.0,
                                       'stalled_time_ref': 10000.0,
                                       'threshold_scorer': 'F1'}
        #
        # For new/continued experiments with this custom individual,
        # to avoid any extra auto-generated individuals (that would compete with this custom individual) set
        # enable_genetic_algorithm = 'off' and set
        # fixed_num_individuals equal to the number of indivs to allow in any GA competition.
        # This is done in expert settings or add that to self.config_dict.
        # If all features and model parameters are frozen (i.e. prob_perturb_xgb, etc. are all 0), then:
        # * if 1 individual, the genetic algorithm is changed to Optuna if 'auto', else 'off'.
        # * if >1 individuals, then the genetic algorithm is disabled (set to 'off').
        # For any individuals, for the frozen case, the number of individuals is set to the number of custom individuals.
        # To disable this automatic handling of frozen or 1 custom individual,
        # set toml change_genetic_algorithm_if_one_brain_population to false.

        # For refit/retrained experiments with this custom individual or for final ensembling control,
        # to avoid any extra auto-generated individuals (that would compete with this custom individual) set
        # fixed_ensemble_level equal to the number of custom individuals desired in the final model ensembling.
        # These tomls can be set in expert settings or added to the experiment-level self.config_dict.
        #
        # To ensemble N custom individuals, set config.fixed_ensemble_level = config.fixed_num_individuals = N
        # to avoid auto-generating other competing individuals and refit/retrain a final model
        #
        # END: CONTROL SOME CONFIG TOML VALUES, MAY BE SET
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
        #  e.g. monotonicity_constraints_dict can be used to constrain feature names at experiment-level of individual level.
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
        # A list of model hyperparameters to adjust back to defaults for tuning or final model building
        #  If empty list, then no changes to model parameters will be made unless a tuning stage mutation on model parameters is done
        #  For each item in list, set_default_params() will be used to fill those parameters for GA
        #  If _is_gbm=True for the class, then these parameters also will be changed for the final model based upon DAI dails
        #  _is_gbm = True is set for model_classes based upon LightGBM, XGBoost, CatBoost, etc.
        #   E.g. for _is_gbm=True these will be changed:
        #    * learning_rate
        #    * early_stopping_rounds
        #    * n_estimators (_fit_by_iteration in general if not None, if _fit_by_iteration=True),
        # After experiment is completed, the new individual in any restart/refit will not use this parameter,
        #  so tree parameters will adapt to the dial values as well as being in tuning vs. final model.
        self.adjusted_params = ['learning_rate', 'early_stopping_rounds', 'n_estimators']

        # To prevent mutation of the model hyperparameters (frozen case), in self.params set:
        # prob_perturb_xgb = 0.0

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
        # If set self._from_exp['columns'] = False, then the below col_dict is used
        # This allows one to control the data types for each column in dataset.
        # NOTE: The transformers may only use subset of columns,
        #  in which case "columns" controls any new transformers as well.
        # NOTE: If any genes consume columns that are not in the given column types,
        #  then they will be automatically added.

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
        # NOTE: For custom recipes, experiments use full hashed names for transformers,
        #       which includes the file name,
        #       but if the recipe version is not important or there is only one version,
        #       then just the name of the transformer is sufficient.

        # To prevent mutation of the genome for this individual (frozen case), in self.params set:
        # prob_add_genes = prob_prune_genes = prob_prune_by_features = prob_addbest_genes = prob_prune_by_features = 0.0

        # Doc string for add_transformer():
        """
        
        transformer collector
        :obj: Transformer display name
        :gene_index: int : index to use for gene and transformed feature name
        :layer: Pipeline layer, 0 (normal single layer), 1, ... n - 1 for n layers
        :forced: Whether forcing in gene/transformer instance to avoid pruning at gene level and any derived feature level
        :mono: Whether making feature monotonic.
               False means no constraint
               True means automatic mode done by DAI
               +1, -1, 0 means specific choice
               'experiment' means depend upon only experiment settings
               Only relevant for transformed features that go into the model,
               e.g. for multi-layer case only last layer is relevant.
        :params: parameters for Transformer
        params should have every mutation key filled, else default parameters used for missing ones

        NOTE: column names are sanitized, which means characters like spaces are not allowed or special internal characters are not allowed.
        The function sanitize_string_list(column_names) can be used to convert known column names into sanitized form, however
        if there are multiple features that differ only by a sanitized string, the de-dup process is dependent on the python-sorted order of column names.
        The function sanitize_string_list can be imported as: `from h2oaicore.transformer_utils import sanitize_string_list`.
        In DAI data handling during experiment, the sanitize_string_list is called on all columns, including:
        target, cols_to_drop, weight_column, fold_column, time_groups_columns, and training/validation/test frames.
        :return:
        
        """

        # Gene Normalized Importance: 0.03427
        # Transformed Feature Names and Importances: {'0_AGE': 0.03427012264728546}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['AGE'], 'random_state': 159699529}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=0, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.07893
        # Transformed Feature Names and Importances: {'1_BILL_AMT1': 0.07892969995737076}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT1'], 'random_state': 159699530}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=1, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.020586
        # Transformed Feature Names and Importances: {'3_BILL_AMT3': 0.02058570086956024}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT3'], 'random_state': 159699532}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=3, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.029232
        # Transformed Feature Names and Importances: {'4_BILL_AMT4': 0.02923220954835415}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT4'], 'random_state': 159699533}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=4, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.0021431
        # Transformed Feature Names and Importances: {'5_BILL_AMT5': 0.002143113175407052}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT5'], 'random_state': 159699534}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=5, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.011209
        # Transformed Feature Names and Importances: {'6_BILL_AMT6': 0.011209040880203247}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['BILL_AMT6'], 'random_state': 159699535}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=6, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.0014719
        # Transformed Feature Names and Importances: {'7_EDUCATION': 0.0014718829188495874}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['EDUCATION'], 'random_state': 159699536}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=7, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance:       1
        # Transformed Feature Names and Importances: {'10_PAY_0': 1.0}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_0'], 'random_state': 159699539}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=10, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.21957
        # Transformed Feature Names and Importances: {'11_PAY_2': 0.21957388520240784}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_2'], 'random_state': 159699540}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=11, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.045535
        # Transformed Feature Names and Importances: {'14_PAY_5': 0.045535363256931305}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_5'], 'random_state': 159699543}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=14, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.026968
        # Transformed Feature Names and Importances: {'15_PAY_6': 0.026968393474817276}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_6'], 'random_state': 159699544}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=15, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.02044
        # Transformed Feature Names and Importances: {'16_PAY_AMT1': 0.020440170541405678}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT1'], 'random_state': 159699545}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=16, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.042924
        # Transformed Feature Names and Importances: {'17_PAY_AMT2': 0.04292424023151398}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT2'], 'random_state': 159699546}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=17, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.10431
        # Transformed Feature Names and Importances: {'18_PAY_AMT3': 0.10430736839771271}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT3'], 'random_state': 159699547}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=18, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.047792
        # Transformed Feature Names and Importances: {'19_PAY_AMT4': 0.04779212549328804}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT4'], 'random_state': 159699548}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=19, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.026807
        # Transformed Feature Names and Importances: {'20_PAY_AMT5': 0.02680712565779686}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT5'], 'random_state': 159699549}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=20, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.064593
        # Transformed Feature Names and Importances: {'21_PAY_AMT6': 0.06459338217973709}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['PAY_AMT6'], 'random_state': 159699550}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=21, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.0029303
        # Transformed Feature Names and Importances: {'22_SEX': 0.0029302926268428564}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['SEX'], 'random_state': 159699551}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=22, forced=False, mono=False,
                             **params)

        ###########################################################################
        #
        # TIME SERIES GROUP INFO, VARIABLES ARE FOR ACCEPTANCE TESTING ONLY, NO NEED TO CHANGE
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
        from h2oaicore.systemutils import config
        return config.acceptance_test_timeout
