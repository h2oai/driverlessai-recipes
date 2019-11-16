"""Extract the counts of different named entities in the text (e.g. Person, Organization, Location)"""

import datatable as dt
import numpy as np
import os

from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.utils import ContribLoader
from h2oaicore.systemutils import remove
import filelock


class TextNamedEntityTransformer(CustomTransformer):
    """Transformer to extract the count of Named Entities"""
    _modules_needed_by_name = ["spacy==2.1.8"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import spacy
        env_dir = ContribLoader._env_dir.resolve()
        lock_file = os.path.join(env_dir, "spacy.lock")
        try:
            with filelock.FileLock(lock_file):
                from spacy.cli import download
                download('en_core_web_sm', False, "--install-option=--prefix=%s" % ContribLoader._env_dir.resolve())
                self.nlp = spacy.load('en_core_web_sm')
        finally:
            remove(lock_file)
        self.ne_types = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE"}

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def get_ne_count(self, text):
        entities = self.nlp(text).ents
        if entities:
            return [len([entity for entity in entities if entity.label_ == ne_type]) for ne_type in self.ne_types]
        else:
            return [0]*len(self.ne_types)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        orig_col_name = X.names[0]
        X = dt.Frame(X).to_pandas().astype(str).fillna("NA")
        new_X = X.apply(lambda x: self.get_ne_count(x[orig_col_name]), axis=1, result_type='expand')
        new_X.columns = [f'{orig_col_name}_Count_{ne_type}' for ne_type in self.ne_types]
        return new_X

