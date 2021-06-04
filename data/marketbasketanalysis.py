"""Market basket analysis"""

"""
Settings for this recipe:

MBA_ORDER_COLUMN: Column name of orders
MBA_PRODUCT_COLUMN: Column name of products
MBA_MIN_SUPPORT: Minimum support level of itemsets
MBA_MAX_LEN: Maximum number of products in itemsets
MBA_METRIC: Metric used for ruleset cutoff
MBA_MIN_THRESHOLD: Threshold value to apply on metric
SEPARATOR: Separator used to distinguish products

More details available here: http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.frequent_patterns
Model used is fpgrowth

Sample dataset to try on: https://www.kaggle.com/c/instacart-market-basket-analysis
"""

import datatable as dt
import numpy as np
import pandas as pd
from h2oaicore.data import CustomData
import typing

# Please edit these before usage
MBA_ORDER_COLUMN = 'order_id'
MBA_PRODUCT_COLUMN = 'product_name'
MBA_MIN_SUPPORT = 0.0005
MBA_MAX_LEN = 4
MBA_METRIC = 'confidence'
MBA_MIN_THRESHOLD = 0.1
SEPARATOR = ' | '


def _clean_frozenset(text: frozenset, separator: str = SEPARATOR) -> str:
        """
        Convert frozenset into a string with separator.
        """

        return separator.join(list(text))


class MarketBasketAnalysis(CustomData):
    _modules_needed_by_name = ["mlxtend"]

    @staticmethod
    def create_data(X: dt.Frame = None) -> pd.DataFrame:
        if X is None:
            return []

        from mlxtend.frequent_patterns import fpgrowth, association_rules

        X = X.to_pandas()[[MBA_ORDER_COLUMN, MBA_PRODUCT_COLUMN]]

        X.loc[:, MBA_ORDER_COLUMN] = X[MBA_ORDER_COLUMN].astype(str)
        X.loc[:, MBA_PRODUCT_COLUMN] = X[MBA_PRODUCT_COLUMN].astype(str)

        X = X.drop_duplicates()
        X['_mba_'] = True

        transactions = pd.pivot_table(
            data=X,
            values='_mba_',
            index=MBA_ORDER_COLUMN,
            columns=MBA_PRODUCT_COLUMN,
            aggfunc=lambda x: x,
            fill_value=False
        )

        df_mba = association_rules(
            df=fpgrowth(
                df=transactions,
                min_support=MBA_MIN_SUPPORT,
                use_colnames=True,
                max_len=MBA_MAX_LEN
            ),
            metric=MBA_METRIC,
            min_threshold=MBA_MIN_THRESHOLD
        )

        df_mba['antecedents'] = df_mba.antecedents.apply(_clean_frozenset)
        df_mba['consequents'] = df_mba.consequents.apply(_clean_frozenset)

        return df_mba
