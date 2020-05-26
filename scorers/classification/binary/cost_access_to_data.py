"""Same as CostBinary, but provides access to full Data"""

import typing
import numpy as np
import datatable as dt
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class CostBinaryWithData(CustomScorer):
    _description = "Calculates cost per row in binary classification: `(fp_cost*FP + fn_cost*FN) / N`"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "CostWithData"
    _threshold = 0.5  # Example only, should be adjusted based on domain knowledge and other experiments

    _needs_X = True  # This assures that X is passed in

    # The cost of false positives and negatives will vary by data set, we use the rules from the below as an example
    # https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set
    _fp_cost = 10
    _fn_cost = 500

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              X: typing.Optional[dt.Frame] = None) -> float:
        # can compute arbitrary cost from all original features
        if X is not None:
            assert X.nrows == len(actual)
            assert X.ncols >= 1
            X_pd = X.to_pandas()

        # label actuals as 1 or 0
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)

        # label predictions as 1 or 0
        predicted = predicted >= self._threshold

        # use sklearn to get fp and fn
        cm = confusion_matrix(actual, predicted, sample_weight=sample_weight, labels=labels)
        tn, fp, fn, tp = cm.ravel()

        # calculate`$1*FP + $2*FN`
        return ((fp * self.__class__._fp_cost) + (fn * self.__class__._fn_cost)) / (
                    tn + fp + fn + tp)  # divide by total weighted count to make loss invariant to data size
