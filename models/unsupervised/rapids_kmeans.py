"""KMeans clustering using RAPIDS.ai"""
import datatable as dt
import numpy as np
from h2oaicore.systemutils import ngpus_vis, IgnoreEntirelyError, config
from h2oaicore.transformers import CustomTransformer
from h2oaicore.metrics import CustomUnsupervisedScorer
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer
from sklearn.metrics import davies_bouldin_score


class NoMoreThanTenNumericTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        # pick up to 10 numeric columns
        return dict(col_type="numeric", min_cols=1, max_cols=10, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X


class RapidsKMeansClusterLabelTransformer(CustomUnsupervisedTransformer):
    _can_use_gpu = True
    _must_use_gpu = True
    _can_use_multi_gpu = False
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _parallel_task = False
    _testing_can_skip_failure = True  # not stable algo, GPU OOM too often

    @staticmethod
    def get_default_properties():
        if not config.hard_asserts:
            return dict(col_type="numeric", min_cols=1, max_cols="all")
        else:
            # testing mode, to avoid GPU OOM etc.
            return dict(col_type="numeric", min_cols=1, max_cols=3)

    @staticmethod
    def get_parameter_choices():
        return dict(n_clusters=[2, 3, 4, 5, 10, 20],
                    max_iters=[300],  # probably no need to tune
                    tol=[1e-4],  # probably no need to tune
                    )

    def __init__(self, n_clusters, max_iters, tol, **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        if ngpus_vis == 0:
            raise IgnoreEntirelyError("Transformer cannot run without GPUs")

        import cudf
        import cuml
        cuml.common.memory_utils.set_global_output_type('numpy')
        self.n_clusters = min(self.n_clusters, X.nrows)
        self.model = cuml.cluster.KMeans(n_clusters=self.n_clusters,
                                         max_iter=self.max_iters,
                                         tol=self.tol)
        X = X.to_pandas().fillna(0)
        X = cudf.DataFrame(X)
        return self.model.fit_predict(X)

    def transform(self, X: dt.Frame, y: np.array = None):
        if ngpus_vis == 0:
            raise IgnoreEntirelyError("Transformer cannot run without GPUs")

        import cudf
        import cuml
        cuml.common.memory_utils.set_global_output_type('numpy')
        X = X.to_pandas().fillna(0)
        X = cudf.DataFrame(X)
        return self.model.predict(X)


# For illustration, same as DaviesBouldinScorer that is shipped with DAI
class MyDaviesBouldinScorer(CustomUnsupervisedScorer):
    _perfect_score = 0
    _maximize = False

    def score(self, actual, predicted, sample_weight=None, labels=None, X=None, **kwargs):
        if len(predicted.shape) != 1:
            raise RuntimeError("wrong shape of predictions, expected cluster IDs, but got: %s" % str(predicted))
        try:
            return davies_bouldin_score(X.to_numpy(), labels=predicted.astype(int).astype(str))
        except ValueError as e:
            if "Number of labels is" not in str(e):  # happens for Constant Model
                raise
            return 1e10


class RapidsKMeansModel(CustomUnsupervisedModel):
    _included_pretransformers = ['NoMoreThanTenNumericTransformer']  # make your own
    # _included_pretransformers = ['OrigFreqPreTransformer']  # from DAI built-in (frequency-encodes categoricals)
    # _included_pretransformers = ['OrigOHEPreTransformer']  # from DAI built-in (one-hot encodes categoricals)

    _included_transformers = ["RapidsKMeansClusterLabelTransformer"]
    _included_scorers = ['MyDaviesBouldinScorer']  # make your own
    # _included_scorers = ['SilhouetteScorer', 'CalinskiHarabaszScorer', 'DaviesBouldinScorer']  # from DAI built-in
