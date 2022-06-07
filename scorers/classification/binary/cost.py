"""Using hard-coded dollar amounts x for false positives and y for false negatives, calculate the cost of a model using: `(x * FP + y * FN) / N`"""

import typing
import numpy as np
from h2oaicore.metrics import CustomScorer, prep_actual_predicted
from sklearn.preprocessing import label_binarize
import h2o4gpu.util.metrics as daicx


class CostBinary(CustomScorer):
    _description = "Calculates cost per row in binary classification: `(fp_cost*FP + fn_cost*FN) / N`"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Cost"
    _threshold_optimizer = "f1"  # used to get the optimal threshold to make labels

    @staticmethod
    def _metric(tp, fp, tn, fn):
        # The cost of false positives and negatives will vary by data set, we use the rules from the below as an example
        # https://www.kaggle.com/uciml/aps-failure-at-scania-trucks-data-set
        _fp_cost = 10
        _fn_cost = 500
        return ((fp * _fp_cost) + (fn * _fn_cost)) / (
                tn + fp + fn + tp)  # divide by total weighted count to make loss invariant to data size

    def protected_metric(self, tp, fp, tn, fn):
        try:
            return self.__class__._metric(tp, fp, tn, fn)
        except ZeroDivisionError:
            return 0 if self.__class__._maximize else 1  # return worst score if ill-defined

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:

        if sample_weight is not None:
            sample_weight = sample_weight.ravel()
        enc_actual, enc_predicted, labels = prep_actual_predicted(actual, predicted, labels)
        cm_weights = sample_weight if sample_weight is not None else None

        # multiclass
        if enc_predicted.shape[1] > 1:
            enc_predicted = enc_predicted.ravel()
            enc_actual = label_binarize(enc_actual, labels).ravel()
            cm_weights = np.repeat(cm_weights, predicted.shape[1]).ravel() if cm_weights is not None else None
            assert enc_predicted.shape == enc_actual.shape
            assert cm_weights is None or enc_predicted.shape == cm_weights.shape

        cms = daicx.confusion_matrices(enc_actual.ravel(), enc_predicted.ravel(), sample_weight=cm_weights)
        cms = cms.loc[
            cms[[self.__class__._threshold_optimizer]].idxmax()]  # get row(s) for optimal metric defined above
        cms['metric'] = cms[['tp', 'fp', 'tn', 'fn']].apply(lambda x: self.protected_metric(*x), axis=1, raw=True)
        return cms['metric'].mean()  # in case of ties
