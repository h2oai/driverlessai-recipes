class CustomModel(BaseCustomModel):
    _boosters = ['custom']  # set this to something that is unique for your model
    _included_transformers = None
    _excluded_transformers = None
    _mojo = False
    _parallel_task = True  # assumes will use n_jobs in params_base
    _fixed_threads = False  # whether have set_threads(max_workers) method for how many threads wil really use
    _can_use_gpu = False
    _can_use_multi_gpu = False
    _is_reproducible = True
    _datatable_in_out = True
    _modules_needed_by_name = []
    # _global_modules_needed_by_name = [] # in module global scope
    _display_name = NotImplemented
    _description = NotImplemented
    _regression = False
    _binary = False
    _multiclass = False

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return True

    @property
    def has_pred_contribs(self):
        return False

    @property
    def has_output_margin(self):
        return False

    @staticmethod
    def override_params_for_fs(params, train_shape, accuracy, time_tolerance, interpretability):
        return params

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0):
        return True

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        self.params = {}

    def mutate_params(self,
                      accuracy, time_tolerance, interpretability,
                      **kwargs):
        pass

    def __init__(self, context=None,
                 unfitted_pipeline_path=None,  # pipeline that creates features inside of class instance
                 # the complete set of features supposed to be created by the pipeline, in case it's different due to
                 # folds, data etc. - needed for consistency with expectations for pred_contribs (Shapley) etc.
                 transformed_features=None,
                 original_user_cols=None,
                 date_format_strings=dict(),
                 **kwargs):
        '''

        :param context:
        :param unfitted_pipeline_path:
        :param transformed_features:
        :param original_user_cols:
        :param date_format_strings:
        :param kwargs:

        self is ensured to have:
        self.num_classes: Number of classes
        self.labels: labels for multiclass
        self.params_base: dict of parameters for model
        '''
        kwargs['n_gpus'] = 0  # no GPU support for now
        if context is not None:
            self.tmp_dir = context.working_dir
        kwargs['booster'] = self._boosters[0]
        MainModel.__init__(self, context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                           transformed_features=transformed_features, original_user_cols=original_user_cols,
                           date_format_strings=date_format_strings, **kwargs)
        self.params_base['booster'] = self._boosters[0]

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        raise NotImplemented("No fit for %s" % self.__class__.__name__)

    def set_feature_importances(self, feature_importances):
        df_imp = pd.DataFrame()
        df_imp['fi'] = self.feature_names_fitted
        df_imp['fi_depth'] = 0
        df_imp['gain'] = feature_importances
        df_imp['gain'] /= df_imp['gain'].max()
        self.feature_importances = df_imp

    def predict(self, X, **kwargs):
        raise NotImplemented("No predict for %s" % self.__class__.__name__)

    def to_mojo(self, mojo: MojoWriter, iframe: MojoFrame):  # -> MojoFrame:
        raise CustomMOJONotImplementedError


ts_raw_data_transformers = ['OriginalTransformer', 'CatOriginalTransformer',
                            'DateOriginalTransformer', 'DateTimeOriginalTransformer']


class CustomTimeSeriesModel(CustomModel):
    _included_transformers = ts_raw_data_transformers

    def __init__(self, context=None, unfitted_pipeline_path=None, transformed_features=None,
                 original_user_cols=None, date_format_strings=dict(), **kwargs):
        if self._included_transformers != ts_raw_data_transformers:
            raise ValueError("Must not override _included_transformers for CustomTimeSeriesModel.")
        MainModel.__init__(self, context=context, unfitted_pipeline_path=unfitted_pipeline_path,
                           transformed_features=transformed_features, original_user_cols=original_user_cols,
                           date_format_strings=date_format_strings, **kwargs)
