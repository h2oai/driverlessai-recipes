"""Extract the count of nouns, verbs, adjectives and adverbs in the text"""
import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer


class POSTagTransformer:
    """Transformer to extract the count of POS tags"""
    _method = NotImplemented
    _modules_needed_by_name = ["nltk==3.4"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import nltk
        try:
            self.pos_tagger = nltk.pos_tag
            self.pos_tagger("test")
        except LookupError:
            nltk.download("averaged_perceptron_tagger")
            nltk.download("maxent_treebank_pos_tagger")
            self.pos_tagger = nltk.pos_tag

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def get_pos_count(self, text):
        pos_tag = self.__class__._method
        pos_tagged_text = self.pos_tagger(text.split())
        return len([word for word, pos in pos_tagged_text if pos[0] == pos_tag])

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.get_pos_count(x))


class NounCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of nouns in the text column"""
    _method = "N"


class VerbCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of verbs in the text column"""
    _method = "V"


class AdjectiveCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of adjectives in the text column"""
    _method = "J"


class AdverbCountTransformer(POSTagTransformer, CustomTransformer):
    """Get the count of adverbs in the text column"""
    _method = "R"
