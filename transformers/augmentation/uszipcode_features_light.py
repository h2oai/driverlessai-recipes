"""Lightweight transformer to parse and augment US zipcodes with info from zipcode database."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from abc import ABC, abstractmethod

_global_modules_needed_by_name = ['zipcodes==1.2.0']
import zipcodes


class ZipcodeLightBaseTransformer(ABC):

    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)

    @abstractmethod
    def get_property_name(self):
        raise NotImplementedError

    def get_zipcode_property(self, zipcode_obj):
        if zipcode_obj is None:
            return None
        else:
            return zipcode_obj[self.get_property_name()]

    def parse_zipcode(self, value):
        try:
            result = zipcodes.matching(value)
            if (len(result) > 1):
                return result[0]
            else:
                return None
        except ValueError:
            return None
        except TypeError:
            raise TypeError

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        try:
            X = dt.Frame(X)
            X.names = ['zip_key']
            X = X[:, str('zip_key')]
            zip_list = dt.unique(X[~dt.isna(dt.f.zip_key), 0]).to_list()[0]
            zip_features = [self.get_zipcode_property(self.parse_zipcode(x)) for x in zip_list]
            X_g = dt.Frame({"zip_key": zip_list, self.get_property_name(): zip_features})
            X_g.key = 'zip_key'
            X_result = X[:, :, dt.join(X_g)]
            return X_result[:, 1:]
        except:
            return np.zeros(X.shape[0])


class ZipcodeTypeTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'zip_code_type'


class ZipcodeCityTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'city'


class ZipcodeStateTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'state'


class ZipcodeLatitudeTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'lat'


class ZipcodeLongitudeTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'long'


class ZipcodeIsActiveTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'active'


class Zipcode5Transformer(ZipcodeLightBaseTransformer, CustomTransformer):
    _unsupervised = True

    def get_property_name(self, value):
        return 'zip_code'
