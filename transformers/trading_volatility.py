#Created by Github user: TomOtt12345
#Calculates Historical Volatility for numeric features

from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import pandas as pd
import numpy as np

#Best applied on financial time series
#The standard rolling window is 252 trading days per year, this may be change to any value you like

class TradingVolatility(CustomTransformer):

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols=1, relative_importance=1)

    #Train
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        x = X.to_pandas()
        vx = (np.log(x / x.shift(1))).rolling(252).std() 
        return vx


    #Validate
    def transform(self, X: dt.Frame):
        x = X.to_pandas()
        vx = (np.log(x / x.shift(1))).rolling(252).std()
        return vx

