"""Custom Final Individual 0 from Experiment test_time_series_walmart_nolimits_3f587_2993_miduhoge """

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


class Indiv_testtimeserieswalmartnolimits3f5872993miduhoge_finalTrue_id0(CustomIndividual):
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
    _regression = True
    _binary = False
    _multiclass = False
    _unsupervised = False
    _description = 'Indiv_testtimeserieswalmartnolimits3f5872993miduhoge_finalTrue_id0'
    _display_name = 'Indiv_testtimeserieswalmartnolimits3f5872993miduhoge_finalTrue_id0'

    # Original Experiment ID
    _experiment_id_origin = 'test_time_series_walmart_nolimits_3f587_2993_miduhoge'
    # Original Experiment Description
    _experiment_description_origin = 'test_time_series_walmart_nolimits_3f587_2993_miduhoge'

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
        self.hash = '9b7fd80f-9e56-4d54-8e2c-a6fd7c768324'
        # Which parent individual by hash
        self.parent_hash = None

        # Score function's (hashed) name
        self.score_f_name = 'MAE'
        # Score (if is_final=True, then this is the final base model out-of-fold score)
        self.score = 2339.5791015625
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 17.993017196655273
        # Tournament Score (penalized by features counts or interpretability)
        self.tournament_score = 2339.5791015625
        # Score history during tuning and evolution
        self.score_list = [2454.2722, 2454.272216796875]
        # Score standard deviation history during tuning and evolution
        self.score_sd_list = [545.6918, 545.6917724609375]

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
        self.seed = 37159439
        # factor of extra genes added during activation
        self.default_factor = 1
        # Ensemble level
        self.ensemble_level = 0
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
                       'output_features_to_drop_more': ['1_Date:Date~get_weekday',
                                                        '6_LagsMin:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64',
                                                        '1_Date:Date~get_weekday',
                                                        '6_LagsMin:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64'],
                       'prob_add_genes': 0.5,
                       'prob_addbest_genes': 0.5,
                       'prob_perturb_xgb': 0.25,
                       'prob_prune_by_features': 0.25,
                       'prob_prune_genes': 0.5,
                       'random_state': 37159439,
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
                                                                 'UMAPCUMLTransformer'],
                                       'one_hot_encoding_cardinality_threshold': 45,
                                       'parameter_tuning_num_models': 0,
                                       'prob_default_lags': 0.2,
                                       'prob_lag_non_targets': 0.1,
                                       'prob_lagsaggregates': 0.2,
                                       'prob_lagsinteraction': 0.2,
                                       'stabilize_fs': False,
                                       'stabilize_varimp': False}
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
        self.config_dict_experiment = {'debug_log': True,
                                       'debug_print': True,
                                       'debug_print_server': True,
                                       'feature_brain_level': 0,
                                       'hard_asserts': True,
                                       'included_models': ['Constant',
                                                           'DecisionTree',
                                                           'FTRL',
                                                           'GLM',
                                                           'ImageAuto',
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
                                                           'XGBoostRFDask',
                                                           'ZeroInflatedLightGBM',
                                                           'ZeroInflatedXGBoost'],
                                       'included_scorers': ['GINI',
                                                            'MAE',
                                                            'MAPE',
                                                            'MER',
                                                            'MSE',
                                                            'R2',
                                                            'R2COD',
                                                            'RMSE',
                                                            'RMSLE',
                                                            'RMSPE',
                                                            'SMAPE'],
                                       'make_mojo_scoring_pipeline': 'off',
                                       'make_pipeline_visualization': 'off',
                                       'make_python_scoring_pipeline': 'off',
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
        self.model_params = {'bagging_seed': 37159441,
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
                             'feature_fraction_seed': 37159440,
                             'gamma': 0.0,
                             'gpu_device_id': 0,
                             'gpu_platform_id': 0,
                             'gpu_use_dp': True,
                             'grow_policy': 'depthwise',
                             'importance_type': 'gain',
                             'label_counts': None,
                             'labels': None,
                             'learning_rate': 0.15,
                             'max_bin': 249,
                             'max_delta_step': 0.0,
                             'max_depth': 8,
                             'max_leaves': 256,
                             'min_child_samples': 20,
                             'min_child_weight': 0.001,
                             'min_data_in_bin': 1,
                             'min_split_gain': 0.0,
                             'monotonicity_constraints': False,
                             'n_estimators': 500,
                             'n_gpus': 1,
                             'n_jobs': 8,
                             'num_class': 1,
                             'num_classes': 1,
                             'num_leaves': 256,
                             'num_threads': 8,
                             'objective': 'mse',
                             'random_state': 37159439,
                             'reg_alpha': 0.0,
                             'reg_lambda': 0.0,
                             'scale_pos_weight': 1.0,
                             'score_f_name': 'MAE',
                             'seed': 37159439,
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
        self.target_transformer_name = 'TargetTransformer_center'

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
                                 'Date': 0.3398695780624856,
                                 'Dept': 0.3283386628749174,
                                 'Fuel_Price': 0.0,
                                 'IsHoliday': 0.0034530961876795565,
                                 'MarkDown1': 0.0,
                                 'MarkDown2': 0.0,
                                 'MarkDown3': 0.0,
                                 'MarkDown4': 0.0,
                                 'MarkDown5': 0.0,
                                 'Store': 0.3283386628749174,
                                 'Temperature': 0.0,
                                 'Unemployment': 0.0}

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

        # Gene Normalized Importance: 0.0049974
        # Transformed Feature Names and Importances: {'0_IsHoliday': 0.004997436888515949}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        params = {'num_cols': ['IsHoliday'], 'random_state': 37159439}
        self.add_transformer('OriginalTransformer', col_type='numeric', gene_index=0, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.016474
        # Transformed Feature Names and Importances: {'1_Date:Date~get_dayofyear': 0.008849767036736012, '1_Date:Date~get_day': 0.003183841472491622, '1_Date:Date~get_year': 0.0027236687019467354, '1_Date:Date~get_week': 0.0013966405531391501, '1_Date:Date~get_month': 0.00029818262555636466, '1_Date:Date~get_quarter': 2.153910645574797e-05}
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
                  'output_features_to_drop': ['1_Date:Date~get_weekday'],
                  'random_state': 37159444}
        self.add_transformer('DatesTransformer', col_type='date', gene_index=1, forced=False, mono=False, **params)

        # Gene Normalized Importance:  1.0273
        # Transformed Feature Names and Importances: {'2_EWMA(0.05)(0)TargetLags:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 1.0, '2_EWMA(0.05)(1)TargetLags:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.017974678426980972, '2_EWMA(0.05)(2)TargetLags:Date:Dept:Store.39:40:41:42:43:44:45:46:47:48:49:50:51:52:53:59:60:71': 0.009322620928287506}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'alpha', 'orders', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
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
                  'random_state': 37159447,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('EwmaLagsTransformer', col_type='time_column', gene_index=2, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance:  0.3951
        # Transformed Feature Names and Importances: {'4_TargetLag:Date:Dept:Store.52': 0.1587725579738617, '4_TargetLag:Date:Dept:Store.42': 0.07914897054433823, '4_TargetLag:Date:Dept:Store.40': 0.045720312744379044, '4_TargetLag:Date:Dept:Store.43': 0.014362356625497341, '4_TargetLag:Date:Dept:Store.53': 0.013182159513235092, '4_TargetLag:Date:Dept:Store.48': 0.01315767876803875, '4_TargetLag:Date:Dept:Store.39': 0.012796975672245026, '4_TargetLag:Date:Dept:Store.47': 0.012772217392921448, '4_TargetLag:Date:Dept:Store.41': 0.01051280926913023, '4_TargetLag:Date:Dept:Store.45': 0.008122248575091362, '4_TargetLag:Date:Dept:Store.46': 0.005692858248949051, '4_TargetLag:Date:Dept:Store.51': 0.00555900763720274, '4_TargetLag:Date:Dept:Store.44': 0.005028808489441872, '4_TargetLag:Date:Dept:Store.60': 0.002602748107165098, '4_TargetLag:Date:Dept:Store.71': 0.0022079397458583117, '4_TargetLag:Date:Dept:Store.49': 0.0019649695605039597, '4_TargetLag:Date:Dept:Store.59': 0.001957128755748272, '4_TargetLag:Date:Dept:Store.50': 0.0015369346365332603}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'dropout', 'nan_value', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
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
                  'random_state': 461986708,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsTransformer', col_type='time_column', gene_index=4, forced=False, mono=False,
                             **params)

        # Gene Normalized Importance: 0.0031523
        # Transformed Feature Names and Importances: {'6_LagsMean:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.000979945994913578, '6_LagsMedian:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.0008526783785782754, '6_LagsKurtosis:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.0005758405313827097, '6_LagsSkew:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.00021151971304789186, '6_LagsStd:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.00020116161613259465, '6_LagsMax:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.0001746349735185504, '6_LagsSum:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64': 0.00015649004490114748}
        # Valid parameters: ['time_column', 'encoder', 'tgc', 'pred_gap', 'pred_periods', 'target', 'lag_sizes', 'lag_feature', 'is_ufapt', 'aggregates', 'nan_value', 'dropout', 'mfr', 'n_jobs', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
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
                  'is_ufapt': False,
                  'lag_feature': 'IsHoliday',
                  'lag_sizes': [40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64],
                  'mfr': False,
                  'n_jobs': 1,
                  'nan_value': nan,
                  'output_features_to_drop': [
                      '6_LagsMin:Date:Dept:Store.IsHoliday.40:42:44:46:48:50:52:54:56:58:60:62:64'],
                  'pred_gap': 0,
                  'pred_periods': 39,
                  'random_state': 3905112276,
                  'target': 'Weekly_Sales',
                  'tgc': ['Date', 'Dept', 'Store'],
                  'time_column': ['Date']}
        self.add_transformer('LagsAggregatesTransformer', col_type='time_column', gene_index=6, forced=False,
                             mono=False, **params)

        # Gene Normalized Importance: 0.00021429
        # Transformed Feature Names and Importances: {'9_Date~is_holiday_UnitedKingdom': 0.00021428860782179981}
        # Valid parameters: ['dt_cols', 'country', 'datetime_formats', 'random_state', 'output_features_to_drop', 'labels']
        # Allowed parameters and mutations (first mutation in list is default): {'country': ['UnitedStates', 'UnitedKingdom', 'EuropeanCentralBank', 'Germany', 'Mexico', 'Japan']}
        params = {'country': 'UnitedKingdom',
                  'datetime_formats': {'Date': '%Y-%m-%d'},
                  'dt_cols': ['Date'],
                  'random_state': 635286614}
        self.add_transformer('IsHolidayTransformer', col_type='date', gene_index=9, forced=False, mono=False, **params)

        ###########################################################################
        #
        # TIME SERIES GROUP INFO, VARIABLES ARE FOR ACCEPTANCE TESTING ONLY, NO NEED TO CHANGE
        #
        from h2oaicore.timeseries_support import LagTimeSeriesGeneInfo, NonLagTimeSeriesGeneInfo, \
            NonTimeSeriesGeneInfo, EitherTimeSeriesGeneInfoBase
        from h2oaicore.timeseries_support import DateTimeLabelEncoder
        from h2oaicore.timeseries_support import TimeSeriesProperties

        # Note: tsgi will use tsp and encoder, and tsp will use encoder
        self.tsgi_params = {'_build_info': {'commit': 'b054db7', 'version': '1.10.2'},
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
                               '_build_info': {'commit': 'b054db7', 'version': '1.10.2'},
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
        from h2oaicore.systemutils import config
        return config.acceptance_test_timeout
