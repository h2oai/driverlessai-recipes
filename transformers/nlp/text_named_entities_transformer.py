"""Extract the counts of different named entities in the text (e.g. Person, Organization, Location)"""

import datatable as dt
import numpy as np
from h2oaicore.systemutils_more import arch_type
from h2oaicore.transformer_utils import CustomTransformer


class TextNamedEntityTransformer(CustomTransformer):
    _unsupervised = True

    """Transformer to extract the count of Named Entities"""
    _testing_can_skip_failure = False  # ensure tested as if shouldn't fail
    _root_path = "https://s3.amazonaws.com/artifacts.h2o.ai/deps/dai/recipes"
    _suffix = "-cp311-cp311-linux_x86_64.whl"
    froms3 = False
    _is_reproducible = False  # some issue with deepcopy and refit, do not get same result
    if froms3:
        # TODO: upload the wheel files to S3
        _modules_needed_by_name = [
            '%s/blis-0.4.1%s' % (_root_path, _suffix),
            '%s/catalogue-1.0.0%s' % (_root_path, _suffix),
            '%s/cymem-2.0.5%s' % (_root_path, _suffix),
            '%s/en_core_web_sm-2.2.5%s' % (_root_path, _suffix),
            '%s/murmurhash-1.0.5%s' % (_root_path, _suffix),
            '%s/plac-1.1.3%s' % (_root_path, _suffix),
            '%s/preshed-3.0.5%s' % (_root_path, _suffix),
            '%s/spacy-2.2.3%s' % (_root_path, _suffix),
            '%s/srsly-1.0.5%s' % (_root_path, _suffix),
            '%s/thinc-7.3.1%s' % (_root_path, _suffix),
            '%s/wasabi-0.8.2%s' % (_root_path, _suffix),
        ]
    else:
        _modules_needed_by_name = ["spacy==3.7.4",
                                   "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm==2.2.5"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ne_types = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "DATE"}

    @staticmethod
    def is_enabled():
        return arch_type != 'ppc64le'

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def get_ne_count(self, text, nlp):
        entities = nlp(text).ents
        if entities:
            return [len([entity for entity in entities if entity.label_ == ne_type]) for ne_type in self.ne_types]
        else:
            return [0] * len(self.ne_types)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        import spacy
        import en_core_web_sm
        nlp = en_core_web_sm.load()

        orig_col_name = X.names[0]
        X = dt.Frame(X).to_pandas().astype(str).fillna("NA")
        new_X = X.apply(lambda x: self.get_ne_count(x[orig_col_name], nlp), axis=1, result_type='expand')
        new_X.columns = [f'{orig_col_name}_Count_{ne_type}' for ne_type in self.ne_types]
        return new_X
