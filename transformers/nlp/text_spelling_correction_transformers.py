"""Correct the spelling of text column"""
import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.transformer_utils import CustomTransformer

class SpellingCorrectionTransformer(CustomTransformer):
    _numeric_output = False
    _modules_needed_by_name = ['pyspellchecker==0.5.0']

    @property
    def display_name(self):
        return "Text"

    @staticmethod
    def is_enabled():
        return False
    
    @staticmethod
    def do_acceptance_test():
        return False
    
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def correction(self, x):
        x = x.lower()
        misspells = self.spell.unknown(x.split())
        corrected = [self.spell.correction(w) if w in misspells else w for w in x.split()]
        corrected = " ".join(corrected)
        return corrected

    def transform(self, X: dt.Frame):
        from spellchecker import SpellChecker
        self.spell = SpellChecker()
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: self.correction(x))
