"""
    Custom Recipe to extract Readability features from the text data
"""

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
import string


class ReadabilityTransformer:
    _modules_needed_by_name = ['textstat==0.6.0']
    _method = NotImplemented
    _parallel_task = False
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail

    @staticmethod
    def do_acceptance_test():
        return True

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        import textstat
        method = getattr(textstat, self.__class__._method)
        return X.to_pandas().astype(str).iloc[:, 0].apply(lambda x: method(x))


class AvgSentenceLengthTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "avg_sentence_length"


class AvgSyllablesPerWordTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "avg_syllables_per_word"


class AvgCharacterPerWordTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "avg_character_per_word"


class SyllableCountTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "syllable_count"


class PolySyllableCountTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "polysyllabcount"


class SmogIndexTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "smog_index"


class GunningFogTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "gunning_fog"


class FleschReadingEaseTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "flesch_reading_ease"


class ColemanLiauIndexTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "coleman_liau_index"


class AutomatedReadabilityIndexTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "automated_readability_index"


class DaleChallReadabilityScoreTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "dale_chall_readability_score"


class LinsearWriteFormulaTransformer(ReadabilityTransformer, CustomTransformer):
    _unsupervised = True

    _method = "linsear_write_formula"
