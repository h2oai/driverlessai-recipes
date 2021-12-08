"""Template base class for a custom individual recipe."""

class CustomIndividual:
    """
    Simplified custom wrapper class to construct DAI Individual

    _params_valid: dict: items that can be filled for individual-level control of parameters (as opposed to experiment-level)
    _from_exp: dict: parameters that are pulled from experiment-level (if value True)
    """
    _params_valid = dict(config_dict=None, accuracy=None, time_tolerance=None, interpretability=None,
                         target_transformer=None,
                         ngenes_min=None, ngenes_max=None,
                         nfeatures_min=None, nfeatures_max=None,
                         output_features_to_drop_more=None, grow_prob=None, grow_anneal_factor=None,
                         grow_prob_lowest=None, explore_prob=None, explore_anneal_factor=None,
                         explore_prob_lowest=None, explore_model_prob=None, explore_model_anneal_factor=None,
                         explore_model_prob_lowest=None, random_state=None, model_params=None,
                         num_as_cat=None, do_te=None)

    # "_from_exp" are added from experiment if value True,
    #  overwriting custom individual values assigned to self value of False means use custom individual value.
    # False as an option makes most sense for 'columns', to ensure the exact column types one desires are used
    #  regardless of experiment-level column types.
    # False is default for 'seed' and 'default_factor' to reproduce individual fitting behavior as closely as possible
    #  even if reproducible is not set.
    # False is not currently supported except for 'columns', 'seed', 'default_factor'.
    # One can override the static var value in the constructor or any function call before _from_exp is actually used
    #  when calling make_indiv.
    #  Note that pickling the custom individual object will not preserve such static var changes.

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

    def __init__(self):
        self.gene_list = []
        self.params = {}

        # informative for code generation, pulled from experiment level settings
        # related to set_genes
        self.columns = None
        self._col_dict_by_layer = None

        # related to set_params
        self.final_best = None
        self.final_pop = None
        self.is_final = None

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
        self.model_origin = None

        # related to set_target_transformer
        self.target_transformer_name = None

        # informative during set_genes, importances of original features
        self.importances_orig = None

    def check_params(self):
        """
        check parameters that they are only among the valid parameters
        :return:
        """
        assert isinstance(self.params, dict), "constructor must be dictionary of params"
        for k, v in self.params.items():
            if k not in CustomIndividual._params_valid:
                raise RuntimeError("Invalid param key %s with value %s" % (k, v))

    def add_transformer(self, transformer_name, gene_index=None, layer=0, **params):
        """
        transformer collector
        :obj: Transformer display name
        :gene_index: int : index to use for gene and transformed feature name
        :layer: Pipeline layer, 0 (normal single layer), 1, ... n - 1 for n layers
        :params: parameters for Transformer
        params should have every mutation key filled, else default parameters used for missing ones
        :return:
        """
        self.gene_list.append(dict(obj=transformer_name, gene_index=gene_index, layer=layer, params=params))

    def add_gene(self, gene, gene_index=None, layer=0, **params):
        """
        gene collector
        :obj: Gene, GeneBluePrint, Transformer, or Transformer display name
        :gene_index: int : index to use for gene and transformed feature name
        :layer: Pipeline layer, 0 (normal single layer), 1, ... n - 1 for n layers
        :params: parameters for GeneBluePrint or Transformer
        params should have every blueprint mutations key filled, else default parameters used for missing ones
        :return:
        """
        self.gene_list.append(dict(obj=gene, gene_index=gene_index, layer=layer, params=params))

    def set_params(self):
        """
        Function to set individual-level parameters.
        If don't set any parameters, the new experiment's values are used.
        :return:
        """
        self.params = {}
        self.check_params()

    def set_model(self):
        """
        Function to set model and its parameters
        :return:
        """
        self.model_display_name = "NotSet"
        self.model_params = {}
        self.model_origin = "NotSet"
        raise NotImplementedError

    def set_target_transformer(self):
        """
        Function to set target transformer.
        If don't set any target transformer, the new experiment's values are used.  E.g. this is valid for classification.
        self.target_transformer_name = "None" applies to classification
        :return:
        """
        self.target_transformer_name = "None"

    def set_genes(self):
        """
        Function to set genes/transformers
        :return:
        """
        self.importances_orig = None
        params = dict(num_cols=['AGE'])
        # gene_index is optional, except:
        # a) use transformed feature names in (e.g.) monotonicity_constraints_dict toml
        # b) use multiple layers with specific col_dict per layer for layer > 0
        self.add_transformer('OriginalTransformer', layer=0, **params)
        raise NotImplementedError

