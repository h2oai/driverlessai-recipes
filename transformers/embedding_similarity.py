from h2oaicore.transformer_utils import CustomTransformer
from sklearn.metrics.pairwise import cosine_similarity
import datatable as dt
import numpy as np

class EmbeddingSimilarityTransformer(CustomTransformer):
    _modules_needed_by_name = ['regex', 'flair==0.4.1', 'segtok-1.5.7']

    def __init__(self, embedding_name, **kwargs):
        super().__init__(**kwargs)
        self.embedding_name = embedding_name

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=2, max_cols=2, relative_importance=1)

    @staticmethod
    def get_parameter_choices():
        return {"embedding_name": ["glove", "en"]}

    @property
    def display_name(self):
        name_map = {"glove":"Glove", "en":"FastText"}
        return "%sEmbedding_CosineSimilarity" % self.embedding_name

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, Sentence
        self.embedding = WordEmbeddings(self.embedding_name)
        self.doc_embedding = DocumentPoolEmbeddings([self.embedding])
        output = []
        X = X.to_pandas()
        text1_arr = X.iloc[:,0].values
        text2_arr = X.iloc[:,1].values
        for ind, text1 in enumerate(text1_arr):
            try:
                text1 = Sentence(str(text1).lower())
                document_embeddings.embed(text1)
                text2 = text2_arr[ind]
                text2 = Sentence(str(text2).lower())
                document_embeddings.embed(text2)
                score = cosine_similarity(text1.get_embedding().reshape(1,-1), 
                                          text2.get_embedding().reshape(1,-1))[0,0]
                output.append(score)
            except:
                output.append(-99)
        return np.array(output)
