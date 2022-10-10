"""Unsupervised way to extract topic information from one text column"""

import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.separators import orig_feat_prefix, extra_prefix


class TextLDATopicTransformer(CustomUnsupervisedTransformer):
    """Transformer to extract topics from text column using LDA"""
    _is_reproducible = False
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _modules_needed_by_name = ["gensim==3.8.0"]

    def __init__(self, n_topics, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"n_topics": [5]}  # CUSTOMIZE

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        import gensim
        from gensim import corpora
        X = dt.Frame(X)
        new_X = X.to_pandas().astype(str).fillna("NA").iloc[:, 0].values
        new_X = [doc.split() for doc in new_X]
        self.dictionary = corpora.Dictionary(new_X)
        new_X = [self.dictionary.doc2bow(doc) for doc in new_X]
        self.model = gensim.models.ldamodel.LdaModel(new_X,
                                                     num_topics=self.n_topics,
                                                     id2word=self.dictionary,
                                                     passes=10,
                                                     random_state=2019)
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X = dt.Frame(X)
        orig_col_name = X.names[0]
        new_X = X.to_pandas().astype(str).fillna("NA").iloc[:, 0].values
        new_X = [doc.split() for doc in new_X]
        new_X = [self.dictionary.doc2bow(doc) for doc in new_X]
        new_X = self.model.inference(new_X)[0]
        self._output_feature_names = [f'{self.display_name}{orig_feat_prefix}{orig_col_name}{extra_prefix}topic{i}'
                                      for i in range(new_X.shape[1])]
        self._feature_desc = [f'LDA Topic {i} of {self.n_topics} for {orig_col_name} column' for i in
                              range(new_X.shape[1])]
        return new_X


class LDAModel(CustomUnsupervisedModel):
    _included_pretransformers = ['TextOriginalTransformer']
    _included_transformers = ['TextLDATopicTransformer']
    _included_scorers = ['UnsupervisedScorer']
