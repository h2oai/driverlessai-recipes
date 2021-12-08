"""Custom Individual 0 from Experiment test_credit_card_7b4bc_3e8d_vakifosa """
from h2oaicore.ga import CustomIndividual
class Indivtestcreditcard7b4bc3e8dvakifosa_finalFalse_id0(CustomIndividual):


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
        self.score_f_name = 'AUC'
        # Score
        self.score = 0.7152961659119884
        # Score standard deviation (if folds or repeats or bootstrapping)
        self.score_sd = 0.007006639889333722
        # Tournament Score (penalized by features counts or interpretabilty)
        self.tournament_score = 0.7557431669159572
        # Score history during tuning and evolution
        self.score_list = [0.7557431669159572, 0.7557431669159572]
        # Score standard deviation history during tuning and evolution
        self.score_sd_list = [0.007189177763664762, 0.007189177763664762]

        # Number of classes if supervised
        self.num_classes = 2
        # Labels if classification, None for regression
        self.labels = [0, 1]

        # Shape of training frame (may include target)
        self.train_shape = (23999, 27)
        # Shape of validation frame (may include target)
        self.valid_shape = None
        # Cardinality for each column
        self.cardinality_dict = {'AGE': 55, 'EDUCATION': 7, 'LIMIT_BAL': 79, 'MARRIAGE': 4, 'PAY_0': 11, 'PAY_2': 11, 'PAY_3': 11, 'PAY_4': 11, 'PAY_5': 10, 'PAY_6': 10, 'SEX': 2}

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
                       'output_features_to_drop_more': [],
                       'random_state': 159699529,
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
                             'max_bin': 251,
                             'max_delta_step': 0.0,
                             'max_depth': 8,
                             'max_leaves': 256,
                             'min_child_samples': 20,
                             'min_child_weight': 0.001,
                             'min_data_in_bin': 1,
                             'min_split_gain': 0.0,
                             'model_class_name': 'LightGBMModel',
                             'model_origin': 'FINAL BASE MODEL 0',
                             'model_origin_original': 'SEQUENCE',
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

        # model origin is for informative purposes only
        self.model_origin = 'FINAL BASE MODEL 0'

    def set_target_transformer(self):
        self.target_transformer_name = 'None'

    def set_genes(self):
        import numpy as np
        nan = np.nan
        from collections import OrderedDict, defaultdict

        # Original variable importances are for reference only, not required to be set
        self.importances_orig = {'AGE': 0.01938793434839645,
                                 'BILL_AMT1': 0.04305812667282816,
                                 'BILL_AMT2': 0.0,
                                 'BILL_AMT3': 0.0075596857485257145,
                                 'BILL_AMT4': 0.012594204103632173,
                                 'BILL_AMT5': 0.005706288665502505,
                                 'BILL_AMT6': 0.0064048942443566395,
                                 'EDUCATION': 0.0008290209811933038,
                                 'LIMIT_BAL': 0.0,
                                 'MARRIAGE': 0.0,
                                 'PAY_0': 0.563238400674736,
                                 'PAY_2': 0.12697294590120037,
                                 'PAY_3': 0.0,
                                 'PAY_4': 0.0,
                                 'PAY_5': 0.028528367327390582,
                                 'PAY_6': 0.015189634809523067,
                                 'PAY_AMT1': 0.011297339581552652,
                                 'PAY_AMT2': 0.02263868736675986,
                                 'PAY_AMT3': 0.05672975122572048,
                                 'PAY_AMT4': 0.02567801048861068,
                                 'PAY_AMT5': 0.015098802582184183,
                                 'PAY_AMT6': 0.0374374519452353,
                                 'SEX': 0.0016504533326519412}
        # Column types are for reference only, not required to be set
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

        # All valid parameters for genes should be provided, except:
        # output_features_to_drop need not be passed if empty list
        # Mutations need not be provided if want to use default values
        # Mutations or valid parameters are not shown if none, like for OriginalTransformer

        # Gene Normalized Importance: 0.034422
        # Transformed Feature Names and Importances: {'0_AGE': 0.03442225232720375}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['AGE'], 'random_state': 159699529}
        self.add_transformer('OriginalTransformer', gene_index=0, **params)

        # Gene Normalized Importance: 0.076447
        # Transformed Feature Names and Importances: {'1_BILL_AMT1': 0.07644742727279663}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['BILL_AMT1'], 'random_state': 159699530}
        self.add_transformer('OriginalTransformer', gene_index=1, **params)

        # Gene Normalized Importance: 0.013422
        # Transformed Feature Names and Importances: {'3_BILL_AMT3': 0.013421822339296341}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['BILL_AMT3'], 'random_state': 159699532}
        self.add_transformer('OriginalTransformer', gene_index=3, **params)

        # Gene Normalized Importance: 0.02236
        # Transformed Feature Names and Importances: {'4_BILL_AMT4': 0.022360343486070633}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['BILL_AMT4'], 'random_state': 159699533}
        self.add_transformer('OriginalTransformer', gene_index=4, **params)

        # Gene Normalized Importance: 0.010131
        # Transformed Feature Names and Importances: {'5_BILL_AMT5': 0.010131213814020157}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['BILL_AMT5'], 'random_state': 159699534}
        self.add_transformer('OriginalTransformer', gene_index=5, **params)

        # Gene Normalized Importance: 0.011372
        # Transformed Feature Names and Importances: {'6_BILL_AMT6': 0.0113715510815382}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['BILL_AMT6'], 'random_state': 159699535}
        self.add_transformer('OriginalTransformer', gene_index=6, **params)

        # Gene Normalized Importance: 0.0014719
        # Transformed Feature Names and Importances: {'7_EDUCATION': 0.0014718829188495874}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['EDUCATION'], 'random_state': 159699536}
        self.add_transformer('OriginalTransformer', gene_index=7, **params)

        # Gene Normalized Importance:       1
        # Transformed Feature Names and Importances: {'10_PAY_0': 1.0}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_0'], 'random_state': 159699539}
        self.add_transformer('OriginalTransformer', gene_index=10, **params)

        # Gene Normalized Importance: 0.22543
        # Transformed Feature Names and Importances: {'11_PAY_2': 0.22543375194072723}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_2'], 'random_state': 159699540}
        self.add_transformer('OriginalTransformer', gene_index=11, **params)

        # Gene Normalized Importance: 0.050651
        # Transformed Feature Names and Importances: {'14_PAY_5': 0.05065060779452324}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_5'], 'random_state': 159699543}
        self.add_transformer('OriginalTransformer', gene_index=14, **params)

        # Gene Normalized Importance: 0.026968
        # Transformed Feature Names and Importances: {'15_PAY_6': 0.026968393474817276}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_6'], 'random_state': 159699544}
        self.add_transformer('OriginalTransformer', gene_index=15, **params)

        # Gene Normalized Importance: 0.020058
        # Transformed Feature Names and Importances: {'16_PAY_AMT1': 0.020057829096913338}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_AMT1'], 'random_state': 159699545}
        self.add_transformer('OriginalTransformer', gene_index=16, **params)

        # Gene Normalized Importance: 0.040194
        # Transformed Feature Names and Importances: {'17_PAY_AMT2': 0.040193792432546616}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_AMT2'], 'random_state': 159699546}
        self.add_transformer('OriginalTransformer', gene_index=17, **params)

        # Gene Normalized Importance: 0.10072
        # Transformed Feature Names and Importances: {'18_PAY_AMT3': 0.10072067379951477}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_AMT3'], 'random_state': 159699547}
        self.add_transformer('OriginalTransformer', gene_index=18, **params)

        # Gene Normalized Importance: 0.04559
        # Transformed Feature Names and Importances: {'19_PAY_AMT4': 0.04558994993567467}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_AMT4'], 'random_state': 159699548}
        self.add_transformer('OriginalTransformer', gene_index=19, **params)

        # Gene Normalized Importance: 0.026807
        # Transformed Feature Names and Importances: {'20_PAY_AMT5': 0.02680712565779686}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_AMT5'], 'random_state': 159699549}
        self.add_transformer('OriginalTransformer', gene_index=20, **params)

        # Gene Normalized Importance: 0.066468
        # Transformed Feature Names and Importances: {'21_PAY_AMT6': 0.06646821647882462}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['PAY_AMT6'], 'random_state': 159699550}
        self.add_transformer('OriginalTransformer', gene_index=21, **params)

        # Gene Normalized Importance: 0.0029303
        # Transformed Feature Names and Importances: {'22_SEX': 0.0029302926268428564}
        # Valid parameters: ['num_cols', 'random_state', 'output_features_to_drop', 'labels']
        # 'labels' parameter is handled internally by DAI
        params = {'num_cols': ['SEX'], 'random_state': 159699551}
        self.add_transformer('OriginalTransformer', gene_index=22, **params)

