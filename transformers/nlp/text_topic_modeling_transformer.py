"""Extract topics from text column using LDA"""
import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer


class TextLDATopicTransformer(CustomTransformer):
    """Transformer to extract topics from text column using LDA"""
    _numeric_output = False
    _is_reproducible = False
    _modules_needed_by_name = ["gensim==3.8.0"]

    def __init__(self, n_topics, **kwargs):
        super().__init__(**kwargs)
        self.n_topics = n_topics

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"n_topics" : [3, 5, 10, 50]}

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
        self._output_feature_names = [f'{orig_col_name}_LDATopic{i}' for i in range(new_X.shape[1])]
        self._feature_desc = [f'LDA Topic {i} of {self.n_topics} for {orig_col_name} column' for i in range(new_X.shape[1])]
        return new_X
