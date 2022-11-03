"""Variety of unsupervised models that mimic internal versions but includes text handling via text embedding using custom transformer"""
import importlib
import sys

from h2oaicore.systemutils import config
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer, Transformer, strip_geneid
from h2oaicore.models import CustomUnsupervisedModel
import datatable as dt
from h2oaicore.models_main import MainModel


class KMeansFreqTextModel(CustomUnsupervisedModel):
    _ngenes_max = 1
    _ngenes_max_by_layer = [1, 1]
    _included_transformers = ["ClusterIdAllNumTransformer"]
    _included_scorers = ['SilhouetteScorer', 'CalinskiHarabaszScorer', 'DaviesBouldinScorer']
    _included_pretransformers = ['OrigFreqTextPreTransformer']
    _description = """Like KMeansFreqModel, but with text support.  This does K-Means clustering on numeric, frequency transformed categorical, and embedded text (integer columns are treated only as numeric)

Clustering algorithms partition observations into clusters. Driverless AI uses [sklearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering algorithm to partition the observations so that they belong to the cluster with the nearest mean (centroid of the cluster).
"""


class IsolationForestAnomalyTextModel(CustomUnsupervisedModel):
    _ngenes_max = 1
    _ngenes_max_by_layer = [1, 1]
    _included_transformers = ["ClusterIdAllNumTransformer"]
    _included_scorers = ['UnsupervisedScorer']
    _included_pretransformers = ['OrigFreqTextPreTransformer']
    _description = """Like IsolationForestAnomalyModel but with text support.  [Isolation forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf) isolates or identifies the anomalous entries by randomly splitting the decision trees. The idea is that an *outlier* will lie farther away from the regular observations in the feature space and hence will require fewer random splits to isolate to the terminal node of a tree. The algorithm assigns an anomaly score to each observation based on its path length (from root node to terminal node) in the forest. The lower the score, the more likely it is that the row is an anomaly.

Internally, Driverless AI runs [sklearn's Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) implementation.

When building a model, the Accuracy and Time knobs of Driverless AI can be toggled to adjust the effort spent on model tuning but presently as there is no scorer being used for isolation forest, when doing genetic algorithm, the model will converge immediately and use one of the models from the tuning phase as the final model. The Interpretability knob is ignored in the default set up. The number of trees or n_estimators for the isolation forest model can be adjusted with the `isolation_forest_nestimators` expert setting parameter.

After building the model, the **scores** can be obtained by predicting on the same dataset. The lower the scores of a row, the more likely it is an outlier or anomaly by the model. The Visualize Scoring Pipeline option summarizes the features used and transformations applied in building the model.

To create **labels** from these scores, quantile value can be used as a threshold. For example, if you know that 5% of the rows are anomalous in your dataset, then this can be used to calculate the 95th quantile of the scores. This quantile can act as a threshold to classify each row as being an anomaly or not.

The Python scoring pipeline can be used to deploy the Isolation Forest model to production (currently no MOJO support).

Use case idea: Given an anomaly detection experiment, you can create predictions on the training dataset, including all original columns, and re-upload into Driverless AI to run a supervised experiment. For a given similar dataset (in production), you now have an unsupervised scorer that tells you the anomaly score for each row, and supervised scorer which makes Shapley per-feature contribution reason codes to explain why each row is an anomaly or not.
"""


class TruncSVDTxtModel(CustomUnsupervisedModel):
    _ngenes_max = 1
    _ngenes_max_by_layer = [1, 1]
    _included_transformers = ["TruncSVDAllNumTransformer"]
    _included_scorers = ['UnsupervisedScorer']
    _included_pretransformers = ['StdFreqTextPreTransformer']
    _description = """[Truncated SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition#Truncated_SVD) is a dimensionality reduction method and can be applied to a dataset to reduce the number of features before running say a supervised algorithm. It factorizes data matrix where the number of columns is equal to the specified  truncation. It is useful in use cases where *sparse* data gets generated like recommender systems or in text processing like tfidf. Internally Driverless AI runs [sklearn Truncated SVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) implementation.

Driverless AI exposes the TRUNCSVD transformer to reduce the number of features. Presently, none of the parameters can be toggled by the user. The n_components created by the TRUNCSVD transformer range between 1 and 5. (Note that these are considered random mutations.) After building the model, Visualizing scoring pipeline can be used to inspect the number of components created. Additionally, the dimensionality reduced dataset can be obtained by predicting on the dataset. Presently as there is no scorer being used for SVD experiment, when doing genetic algorithm, the model will converge immediately and use one of the models from the tuning phase as the final model.

The Dimensionality Reduction model produces MOJOs and Python scoring pipelines to deploy to production.
"""


class KMeansOHETxtModel(KMeansFreqTextModel):
    _included_pretransformers = ['OrigOHETxtPreTransformer']
    _description = """Like KMeansOHEModel, but with text handling.  This does K-Means clustering on numeric and one-hot-encoding transformed categorical columns

Clustering algorithms partition observations into clusters. Driverless AI uses [sklearn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) clustering algorithm to partition the observations so that they belong to the cluster with the nearest mean (centroid of the cluster).
"""


class AggregatorTxtModel(CustomUnsupervisedModel):
    _ngenes_max = 1
    _ngenes_max_by_layer = [1, 1]
    _included_pretransformers = ['StdFreqTextPreTransformer']
    _included_transformers = ['AggregatorTransformer']
    _description = """Like AggregatorModel, but with text handling.  This runs the Aggregator algorithm on numeric and categorical columns

Aggregation of rows into a set of exemplars. Driverless AI uses the [Aggregator algorithm](https://ieeexplore.ieee.org/document/8019881).
"""


class NumCatTxtPreTransformer(CustomUnsupervisedTransformer):
    """
    Based upon NumCatPreTransformer with text handling added using custom recipe
    """
    _ignore_col_names_mojo_test = True  # since can contain Orig Transformer
    _only_as_pretransformer = True
    _num = ("StandardScalerTransformer", "numeric")
    _cat = ("FrequentTransformer", "categorical")
    _txt = ("TextLDATopicTransformer", "text")

    @staticmethod
    def get_default_properties():
        return dict(col_type="all",
                    min_cols='all',
                    max_cols='all',
                    relative_importance=1)

    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None,
                n_gpus=0, num_classes=None, **kwargs):
        return config.recipe == "unsupervised"

    def calc_feature_desc_final(self):
        assert self._output_feature_names

        ncols = len(self.num_cols) if self.num_cols else 0
        ccols = len(self.cat_cols) if self.cat_cols else 0
        tcols = len(self.txt_cols) if self.txt_cols else 0
        num_out = len(self._output_feature_names)
        # make short version, don't want to have long description for every feature if have thousands of features
        self._feature_desc = \
            [
                "Pretransformer for {} numeric column(s) and {} categorical columns(s) and {} text columns(s), component {} out of {}".
                    format(ncols, ccols, tcols, x, num_out) for x in range(num_out)]

    def __init__(self, all_cols, **kwargs):
        Transformer.__init__(self, **kwargs)
        # if have features to drop, remove them,
        # or else mojo will still see even if drop from oframe in write_to_mojo_base() in transformer_utils.py
        # but this only removes this layer, not from iframe
        all_cols = [x for x in all_cols if x not in strip_geneid(self.output_features_to_drop)]

        col_dict = kwargs["col_dict"]  # passed in from make_transformer, not gene
        assert col_dict, "must pass non-empty col_dict"
        assert any([x in col_dict for x in ['numeric', 'categorical', 'ohe_categorical', 'text']]), \
            "col-dict must contain at least one of numeric/categorical/ohe_categorical/text"
        set_all_cols = set(all_cols)
        NumTrans = None
        CatTrans = None
        TxtTrans = None
        self.num_cols = None
        self.cat_cols = None
        self.txt_cols = None

        # fastest way to get the right transformer
        if self._num:
            NumTrans = getattr(sys.modules['h2oaicore.transformers'], self._num[0], False) or \
                       getattr(sys.modules['h2oaicore.transformers_more'], self._num[0], False)
            self.num_cols = col_dict[self._num[1]]
            self.num_cols = sorted(set([x for x in self.num_cols if x in set_all_cols]))
        if self._cat:
            CatTrans = getattr(sys.modules['h2oaicore.transformers'], self._cat[0], False) or \
                       getattr(sys.modules['h2oaicore.transformers_more'], self._cat[0], False)
            self.cat_cols = col_dict[self._cat[1]]
            self.cat_cols = sorted(set([x for x in self.cat_cols if x in set_all_cols]))
        if self._txt:
            module_name = self.get_text_module_name()
            TxtTrans = getattr(importlib.import_module(module_name), self._txt[0])
            self.txt_cols = col_dict[self._txt[1]]
            self.txt_cols = sorted(set([x for x in self.txt_cols if x in set_all_cols]))

        # set up the pipeline, feature by feature
        del kwargs['input_feature_names']
        kwargs_txt = kwargs.copy()
        for k, v in TxtTrans.get_parameter_choices().items():
            kwargs_txt.update({k: v[0]})  # NOTE: default parameter, can choose others
        if self.cat_cols and CatTrans and self.num_cols and NumTrans and self.txt_cols and TxtTrans:
            self.union = [([x], NumTrans(num_cols=[x], input_feature_names=[x], **kwargs)) for x in self.num_cols] + \
                         [([x], CatTrans(cat_cols=[x], input_feature_names=[x], **kwargs)) for x in self.cat_cols] + \
                         [([x], TxtTrans(txt_cols=[x], input_feature_names=[x], **kwargs_txt)) for x in self.txt_cols]
        elif self.cat_cols and CatTrans and self.txt_cols and TxtTrans:
            self.union = [([x], CatTrans(cat_cols=[x], input_feature_names=[x], **kwargs)) for x in self.cat_cols] + \
                         [([x], TxtTrans(txt_cols=[x], input_feature_names=[x], **kwargs_txt)) for x in self.txt_cols]
        elif self.num_cols and NumTrans and self.txt_cols and TxtTrans:
            self.union = [([x], NumTrans(num_cols=[x], input_feature_names=[x], **kwargs)) for x in self.num_cols] + \
                         [([x], TxtTrans(txt_cols=[x], input_feature_names=[x], **kwargs_txt)) for x in self.txt_cols]
        if self.cat_cols and CatTrans and self.num_cols and NumTrans:
            self.union = [([x], NumTrans(num_cols=[x], input_feature_names=[x], **kwargs)) for x in self.num_cols] + \
                         [([x], CatTrans(cat_cols=[x], input_feature_names=[x], **kwargs)) for x in self.cat_cols]
        elif self.txt_cols and TxtTrans:
            self.union = [([x], TxtTrans(txt_cols=[x], input_feature_names=[x], **kwargs_txt)) for x in self.txt_cols]
        elif self.cat_cols and CatTrans:
            self.union = [([x], CatTrans(cat_cols=[x], input_feature_names=[x], **kwargs)) for x in self.cat_cols]
        elif self.num_cols and NumTrans:
            self.union = [([x], NumTrans(num_cols=[x], input_feature_names=[x], **kwargs)) for x in self.num_cols]

    def get_text_module_name(self):
        name = 'transformers'
        from h2oaicore.utils import ContribLoader
        # 'text_preprocessing' is pattern matching name of file for custom recipe we wish to use
        module_names = [k for k, v in sys.modules.items() if
                        '%s%s' % (ContribLoader._contentpattern, name) in k and self._txt[0] in dir(v)]
        if name in config.recipe_activation and self._txt[0] in config.recipe_activation[name]:
            file_name = config.recipe_activation[name][self._txt[0]].split('|')[1]
            module_names = [x for x in module_names if file_name in x]
            assert len(module_names) > 0, "Missing %s recipe, please upload" % self._txt[0]
            assert len(module_names) == 1, "Unexpected multiple matches for %s recipe" % self._txt[0]
            module_name = module_names[0]
        else:
            # for acceptance testing, unsure how else to know which one to use unless specify exact match
            assert len(module_names) > 0, "Missing %s recipe," \
                                          " please upload (config.recipe_activation=%s)" % \
                                          (self._txt[0], config.recipe_activation)
            module_name = module_names[-1]  # use latest one.
        return module_name

    def transform(self, X, y=None, **fit_params):
        frames = []
        for x in self.union:
            kwargs_tr = MainModel.strip_unhandled_args_for_func(fit_params, x[1].transform)
            X1 = dt.Frame(x[1].transform(X[:, x[0]], **kwargs_tr))
            frames.append(X1)
        X = dt.cbind(frames)
        X.replace(None, [0, 0.0])  # replaces missing with 0 in int, with 0.0 in float cols
        return X

    def fit_transform(self, X, y=None, **fit_params):
        frames = []
        for x in self.union:
            kwargs_fit = MainModel.strip_unhandled_args_for_func(fit_params, x[1].fit_transform)
            X1 = dt.Frame(x[1].fit_transform(X[:, x[0]], y, **kwargs_fit))
            frames.append(X1)
        X = dt.cbind(frames)
        X.replace(None, [0, 0.0])  # replaces missing with 0 in int, with 0.0 in float cols
        return X


class OrigFreqTextPreTransformer(NumCatTxtPreTransformer):
    _num = ("OriginalTransformer", "numeric")
    _cat = ("FrequentTransformer", "categorical")


class StdFreqTextPreTransformer(NumCatTxtPreTransformer):
    _num = ("StandardScalerTransformer", "numeric")
    _cat = ("FrequentTransformer", "categorical")


class OrigOHETxtPreTransformer(NumCatTxtPreTransformer):
    _num = ("OriginalTransformer", "numeric")
    _cat = ("OneHotEncodingUnsupervisedTransformer", "ohe_categorical")
