"""Extract the counts of different named entities in the text (e.g. Person, Organization, Location)"""

import datatable as dt
import numpy as np

from h2oaicore.transformer_utils import CustomTransformer


class NamedEntityTransformer:
    """Transformer to extract the count of Named Entities"""
    _method = NotImplemented
    _modules_needed_by_name = ["spacy==2.1.8"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        import spacy
        self.nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def get_ne_count(self, text):
        ne_type = self.__class__._method
        entities = self.nlp(text).ents
        if entities:
            return len([entity for entity in entities if entity.label_ == ne_type])
        else:
            return 0

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        return X.to_pandas().astype(str).fillna("NA").iloc[:, 0].apply(lambda x: self.get_ne_count(x))


class PersonCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of Persons in the text column"""
    _method = "PERSON"


class OrgCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of organizations in the text column"""
    _method = "ORG"


class GeoCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of countries, cities, states in the text column"""
    _method = "GPE"


class LocCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of non-GPE locations in the text column"""
    _method = "LOC"


class ProductCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of products (objects, vehicles, foods, etc.) in the text column"""
    _method = "PRODUCT"


class EventCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of events (named hurricanes, battles, wars, sports events, etc.) in the text column"""
    _method = "EVENT"


class DateCountTransformer(NamedEntityTransformer, CustomTransformer):
    """Get the count of dates and periods in the text column"""
    _method = "DATE"
