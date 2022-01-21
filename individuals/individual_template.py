"""Template base class for a custom individual recipe."""

from h2oaicore.ga_custom import BaseIndividual


class CustomIndividual(BaseIndividual):
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
    _params_doc = dict(accuracy="accuracy dial",
                       time_tolerance="time dial",
                       interpretability="interpretability dial",
                       ngenes_min="minimum number of genes",
                       ngenes_max="maximum number of genes",
                       nfeatures_min="minimum number of features",
                       nfeatures_max="maximum number of features",
                       output_features_to_drop_more="list of features to drop from overall genome output",
                       grow_prob="""Probability to grow genome
Fast growth of many genes at once is controlled by chance
grow_prob = max(grow_prob_lowest, grow_prob * grow_anneal_factor)""",
                       grow_anneal_factor="Annealing factor for growth",
                       grow_prob_lowest="Lowest growth probability",
                       explore_prob="""Explore Probability
Exploration vs. Exploitation of Genetic Algorithm feature exploration is controlled via
explore_prob = max(explore_prob_lowest, explore_prob * explore_anneal_factor)""",
                       explore_anneal_factor="Explore anneal factor",
                       explore_prob_lowest="Lowest explore probability",
                       explore_model_prob="""Explore Probability for models
Exploration vs. Exploitation of Genetic Algorithm model hyperparameter is controlled via
explore_model_prob = max(explore_model_prob_lowest, explore_model_prob * explore_model_anneal_factor)""",
                       explore_model_anneal_factor="Explore anneal factor for models",
                       explore_model_prob_lowest="Lowest explore probability for models",

                       prob_perturb_xgb="Unnormalized probability to change model hyperparameters",
                       prob_prune_genes="Unnormalized probability to prune genes",
                       prob_prune_by_features="Unnormalized probability to prune features",
                       prob_add_genes="Unnormalized probability to add genes",
                       prob_addbest_genes="Unnormalized probability to add best genes",

                       random_state="random seed for individual",
                       num_as_cat="whether to treat numeric as categorical",
                       do_te="""Whether to support target encoding (TE) (True, False, 'only', 'catlabel')
True means can do TE, False means cannot do TE, 'only' means only have TE
'catlabel' is special mode for LightGBM categorical handling, to only use that categorical handling""",
                       )

    _params_valid = dict(accuracy=int,
                         time_tolerance=int,
                         interpretability=int,
                         ngenes_min=int,
                         ngenes_max=int,
                         nfeatures_min=int,
                         nfeatures_max=int,
                         output_features_to_drop_more=list,

                         grow_prob=float,
                         grow_anneal_factor=float,
                         grow_prob_lowest=float,
                         explore_prob=float,
                         explore_anneal_factor=float,
                         explore_prob_lowest=float,
                         explore_model_prob=float,
                         explore_model_anneal_factor=float,
                         explore_model_prob_lowest=float,

                         prob_perturb_xgb=float,
                         prob_prune_genes=float,
                         prob_prune_by_features=float,
                         prob_add_genes=float,
                         prob_addbest_genes=float,

                         random_state=int,
                         num_as_cat=bool,
                         do_te=[True, False, 'only', 'catlabel'],
                         )

    _from_exp_doc = """
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

    _from_exp = {  # as in set_genes
        'columns': True,
        # as in set_params
        'num_classes': True,
        'labels': True,
        'default_factor': False,
        'target': True,
        'target_transformer': True,
        'weight_column': True,
        'time_column': True,
        # score_f_name (hashed) is set, but experiment passes in score_f directly
        'score_f': True,
        'num_validation_splits': True,
        'seed': False,
        'ensemble_level': True,
        # time series group information, only passed in from experiment to make new genes
        # e.g. if genetic algorithm uses this individual
        # e.g. new columns are present for which need to make new genes during refit
        'tsgi': True,  # tsgi also contains tsgi.encoder for 'encoder' parameter of genes, pulled from experiment
        'train_shape': True,
        'valid_shape': True,
        'cardinality_dict': True,
        'label_counts': True,
        'imbalance_ratio': True,
    }

    callable_prefix = 'DAI_CALLABLE:'

    def __init__(self):
        self.gene_list = []
        self.params = {}

        # informative for code generation, pulled from experiment level settings
        # related to set_genes
        self.columns = None
        self._col_dict_by_layer = None

        self.experiment_id = "UnsetID"
        self.experiment_description = "Unset Description"

        # related to set_params
        self.final_best = None
        self.final_pop = None
        self.is_final = None
        self.hash = None
        self.parent_hash = None

        self.score = None
        self.score_sd = None
        self.tournament_score = None
        self.score_list = None
        self.score_sd_list = None

        self.num_classes = None
        self.labels = None
        self.train_shape = None
        self.valid_shape = None
        self.score_f_name = None
        self.target = None
        self.num_validation_splits = None
        self.seed = None
        self.default_factor = None
        self.weight_column = None
        self.time_column = None
        self.ensemble_level = None
        self.cardinality_dict = None
        self.label_counts = None
        self.imbalance_ratio = None

        # related to set_model
        self.model_display_name = None
        self.model_params = None
        self.model_origin = "NotSet"
        self.adjusted_params = None

        # related to set_target_transformer
        self.target_transformer_name = None
        self.target_transformer_params = None

        # informative during set_genes, importances of original features
        self.importances_orig = None

        self.tsgi = None
        self.tsp = None
        self.encoder = None

        # config_dict is used for experiment-level behavior,
        # while config_dict_individual adds to individual-level behavior if enforce_experiment_config is False
        # config_dict_experiment are only informative.
        self.config_dict = None
        self.config_dict_individual = None
        self.enforce_experiment_config = None
        self.config_dict_experiment = None

    def add_transformer(self, transformer_name, col_type=None, gene_index=None, layer=0, forced=False, mono=False, **params):
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
        self.gene_list.append(dict(obj=transformer_name, col_type=col_type, gene_index=gene_index, layer=layer, forced=forced, mono=mono, params=params))

    def set_params(self):
        """
        Function to set individual-level parameters.
        If don't set any parameters, the new experiment's values are used.
        :return:
        """
        self.params = {}

    def set_model(self):
        """
        Function to set model and its parameters
        :return:
        """
        self.model_display_name = "NotSet"
        self.model_params = {}
        self.model_origin = "NotSet"
        self.adjusted_params = []
        raise NotImplementedError

    def set_target_transformer(self):
        """
        Function to set target transformer.
        If don't set any target transformer, the new experiment's values are used.  E.g. this is valid for classification.
        self.target_transformer_name = "None" applies to classification
        self.target_transformer_params = {} applies to non-time-series target transformers, only for informative purposes
        :return:
        """
        self.target_transformer_name = "None"
        self.target_transformer_params = {}

    def set_genes(self):
        """
        Function to set genes/transformers
        :return:
        """
        self.importances_orig = None
        params = dict(num_cols=['AGE'])
        self.add_transformer('OriginalTransformer', layer=0, **params)
        raise NotImplementedError

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


