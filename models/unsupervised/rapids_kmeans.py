"""KMeans clustering using RAPIDS.ai"""
import datatable as dt
import numpy as np
from h2oaicore.transformers import CustomTransformer
from h2oaicore.metrics import CustomUnsupervisedScorer
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer


class RapidsKMeansClusterLabelTransformer(CustomUnsupervisedTransformer):

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols="all")

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        import cudf
        import cuml
        cuml.common.memory_utils.set_global_output_type('numpy')
        self.model = cuml.cluster.KMeans(n_clusters=8, max_iter=300, tol=1e-4)
        X = X.to_pandas().fillna(0)
        X = cudf.DataFrame(X)
        return self.model.fit_predict(X)

    def transform(self, X: dt.Frame, y: np.array = None):
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
    _included_pretransformers = ['StdFreqPreTransformer']
    _included_transformers = ["RapidsKMeansClusterLabelTransformer"]
    _included_scorers = ['MyDaviesBouldinScorer']

