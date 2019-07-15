"""Lightweight transformer to parse and augment US zipcodes with info from zipcode database."""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np
from abc import ABC, abstractmethod

# saved for more elobarate zipcode transformer:
# from uszipcode import SearchEngine
_global_modules_needed_by_name = ['zipcodes==1.0.5']
import zipcodes

class ZipcodeLightBaseTransformer(ABC):
    
    @staticmethod
    def get_default_properties():
        return dict(col_type="categorical", min_cols=1, max_cols=1, relative_importance=1)
    
    @abstractmethod
    def get_zipcode_property(self, value):
        raise NotImplementedError
        
    def parse_zipcode(self, value):
        result = zipcodes.matching(value)
        
        return result[0]

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):

        try:
            X = X.astype('str')
            return X[:, {"x": (dt.isna(dt.f[0])) & None | self.get_zipcode_property(self.parse_zipcode(dt.f[0]))}]
        except:
            return np.zeros(X.shape[0])
        

class ZipcodeTypeTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['zip_code_type']
    
class ZipcodeCityTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['city']
    
class ZipcodeStateTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['state']
    
class ZipcodeLatitudeTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['lat']
    
class ZipcodeLongitudeTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['long']
    
class ZipcodeIsActiveTransformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['active']
    
class Zipcode5Transformer(ZipcodeLightBaseTransformer, CustomTransformer):
    def get_zipcode_property(self, value):
        return value['zip_code']
