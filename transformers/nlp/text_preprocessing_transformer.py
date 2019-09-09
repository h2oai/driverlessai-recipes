"""Preprocess the text column by stemming, lemmatization and stop word removal"""
import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer


class TextPreprocessingTransformer(CustomTransformer):
    """Transformer to preprocess the text"""
    _numeric_output = False
    _is_reproducible = True
    _modules_needed_by_name = ["nltk==3.4"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.do_stemming = True # turn off as needed
        self.do_lemmatization = True # turn off as needed
        self.remove_stopwords = True # turn off as needed

        import nltk
        if self.do_stemming:
            try:
                self.stemmer = nltk.stem.porter.PorterStemmer()
            except LookupError:
                nltk.download("punkt")
                self.stemmer = nltk.stem.porter.PorterStemmer()
        if self.do_lemmatization:
            try:
                from nltk.corpus import wordnet
                self.lemmatizer = nltk.stem.WordNetLemmatizer()
                self.pos_tagger = nltk.pos_tag
                self.pos_tagger("test")
            except LookupError:
                nltk.download("averaged_perceptron_tagger")
                nltk.download("maxent_treebank_pos_tagger")
                nltk.download("wordnet")
                from nltk.corpus import wordnet
                self.lemmatizer = nltk.stem.WordNetLemmatizer()
                self.pos_tagger = nltk.pos_tag
            self.wordnet_map = {"N":wordnet.NOUN,
                                "V":wordnet.VERB,
                                "J":wordnet.ADJ,
                                "R":wordnet.ADV,
                                "O":wordnet.NOUN}
        if self.remove_stopwords:
            try:
                self.stopwords = set(nltk.corpus.stopwords.words('english'))
            except LookupError:
                nltk.download("stopwords")
                self.stopwords = set(nltk.corpus.stopwords.words('english'))

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @property
    def display_name(self):
        return "PreprocessedText"

    def preprocess(self, text):
        if self.do_stemming:
            text = " ".join([self.stemmer.stem(word) for word in text.split()])
        if self.do_lemmatization:
            pos_tagged_text = self.pos_tagger(text.split())
            text = " ".join([self.lemmatizer.lemmatize(word, self.wordnet_map.get(pos[0], self.wordnet_map["O"]))
                             for word, pos in pos_tagged_text])
        if self.remove_stopwords:
            text = " ".join([word for word in str(text).split()
                             if word.lower() not in self.stopwords])
        return text

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.preprocess(x))
