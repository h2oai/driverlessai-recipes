"""Row-by-row similarity between two text columns based on pretrained Deep Learning embedding space"""
from h2oaicore.transformer_utils import CustomTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datatable as dt
import numpy as np
import math


class EmbeddingSimilarityTransformer(CustomTransformer):
    _modules_needed_by_name = ['regex==2019.12.17', 'flair==0.4.1', 'segtok==1.5.7']
    _is_reproducible = False
    _can_use_gpu = True
    _repl_val = 0
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    def __init__(self, embedding_name, **kwargs):
        super().__init__(**kwargs)
        self.embedding_name = embedding_name

    @staticmethod
    def is_enabled():
        return False  # sometimes package flair has issues installing

    @staticmethod
    def can_use(accuracy, interpretability, **kwargs):
        """Uses all GPU memory - can lead to OOM failures in combination with other GPU-based transformers"""
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"embedding_name": ["glove", "en", "bert"]}

    @property
    def display_name(self):
        name_map = {"glove": "Glove", "en": "FastText", "bert": "BERT"}
        return "%sEmbedding_CosineSimilarity" % name_map[self.embedding_name]

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        X.replace([None, math.inf, -math.inf], self._repl_val)
        return self.transform(X)

    def transform(self, X: dt.Frame):
        X.replace([None, math.inf, -math.inf], self._repl_val)
        from flair.embeddings import WordEmbeddings, BertEmbeddings, DocumentPoolEmbeddings, Sentence
        if self.embedding_name in ["glove", "en"]:
            self.embedding = WordEmbeddings(self.embedding_name)
        elif self.embedding_name in ["bert"]:
            self.embedding = BertEmbeddings()
        self.doc_embedding = DocumentPoolEmbeddings([self.embedding])
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:, 0].values
        text2_arr = X.iloc[:, 1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = Sentence(str(text1).lower())
                self.doc_embedding.embed(text1)
                text2 = text2_arr[ind]
                text2 = Sentence(str(text2).lower())
                self.doc_embedding.embed(text2)
                score = cosine_similarity(text1.get_embedding().reshape(1, -1),
                                          text2.get_embedding().reshape(1, -1))[0, 0]
                output.append(score)
            except:
                output.append(-99)
        return np.array(output)
