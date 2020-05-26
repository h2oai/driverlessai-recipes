# Author: Michelle Tanco - michelle.tanco@h2o.ai
# Last Updated: July 12th, 2019
"""Uses domain information about user behavior to calculate the profit or loss of a model."""

import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class cost_binary(CustomScorer):
    _description = "Calculates how much revenue a specific model could make or lose (per row)"
    _binary = True
    _maximize = True
    _perfect_score = 100000000000
    _display_name = "Profit"

    """
    This Scorer is for binary classification models where customer type 0 prefers one experience
        and customer type 1 prefers another experience. Based on how we classify someone, and thus what experience
        we give them, they will be more or less likely to purchase a specific product. 
    
    The following are known purchase-probabilities from the domain experts (for a specific use case)
    """
    _is_one_predict_one = 0.42
    _is_one_predict_zero = 0.30
    _is_zero_predict_one = 0.18
    _is_zero_predict_zero = 0.30

    # The above rates apply to likelihood to buy a specific product with this cost
    _product_cost = 7.99

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        # label actuals as 1 or 0
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)

        # label predictions as 1 or 0 - we will likely want to change this threshold later
        predicted = predicted >= 0.5

        # use sklean to get fp and fn
        cm = confusion_matrix(actual, predicted, sample_weight=sample_weight, labels=labels)
        tn, fp, fn, tp = cm.ravel()

        # purely to reduce line length
        cost = self.__class__._product_cost

        tp_profit = tp * cost * (self.__class__._is_one_predict_one - self.__class__._is_one_predict_zero)
        fn_loss = fn * cost * (self.__class__._is_one_predict_zero - self.__class__._is_one_predict_one)
        fp_loss = fp * cost * (self.__class__._is_zero_predict_one - self.__class__._is_zero_predict_zero)
        tn_profit = tn * cost * (self.__class__._is_zero_predict_zero - self.__class__._is_zero_predict_one)

        return (tp_profit + fn_loss + fp_loss + tn_profit) / (
                    tn + fp + fn + tp)  # divide by total weighted count to make loss invariant of data size
