"""Preprocess the text column by replacing named entities with a standard tag
For example: 'Mary lives in London from 2018' -> '[PERSON] lives in [GPE] from [DATE]' """
import datatable as dt
import numpy as np
from h2oaicore.transformer_utils import CustomTransformer


class NamedEntityConverterTransformer(CustomTransformer):
    """Transformer to replace mentions of named entities with standard tags the text"""
    _numeric_output = False
    _modules_needed_by_name = ["spacy==2.1.8"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.replace_person = True  # turn off as needed
        self.replace_location = True  # turn off as needed
        self.replace_date = True  # turn off as needed

        import spacy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except IOError:
            from spacy.cli import download
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    @property
    def display_name(self):
        return "NamedEntityConvertedText"

    def convert_named_entities(self, text, entity_type):
        tokens = self.nlp(text)
        new_text = []
        for token in tokens:
            if token.ent_type_ == entity_type:
                word = "[{0}]".format(entity_type)
            else:
                word = token.text
            new_text.append(word)
        return " ".join(new_text)

    def convert_text(self, text):
        if self.replace_person:
            text = self.convert_named_entities(text, "PERSON")
        if self.replace_date:
            text = self.convert_named_entities(text, "DATE")
        if self.replace_location:
            text = self.convert_named_entities(text, "LOC")
            text = self.convert_named_entities(text, "GPE")

        return text

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.convert_text(x))
