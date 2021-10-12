"""Profit Scorer for binary classification"""

import typing
import datatable as dt
import numpy as np
from h2oaicore.metrics import CustomScorer, prep_actual_predicted
from sklearn.preprocessing import label_binarize
import h2o4gpu.util.metrics as daicx


class ProfitWithData(CustomScorer):
    _description = "Calculates how much revenue a specific model could make or lose"
    _binary = True
    _maximize = True
    _perfect_score = 10000000000000
    _needs_X = True  # This assures that X is passed in
    _display_name = "Profit"
    _threshold_optimizer = "f1"  # used to get the optimal threshold to make labels

    @staticmethod
    def _metric(tp, fp, tn, fn):
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
        cost = 7.99

        tp_profit = tp * cost * (_is_one_predict_one - _is_one_predict_zero)
        fn_loss = fn * cost * (_is_one_predict_zero - _is_one_predict_one)
        fp_loss = fp * cost * (_is_zero_predict_one - _is_zero_predict_zero)
        tn_profit = tn * cost * (_is_zero_predict_zero - _is_zero_predict_one)

        return (tp_profit + fn_loss + fp_loss + tn_profit) / (tn + fp + fn + tp)

    def protected_metric(self, tp, fp, tn, fn, X_pd):  # X_pd can be used if desired
        try:
            return self.__class__._metric(tp, fp, tn, fn)
        except ZeroDivisionError:
            return 0 if self.__class__._maximize else 1  # return worst score if ill-defined

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              X: typing.Optional[dt.Frame] = None,
              **kwargs) -> float:
        # can compute arbitrary cost from all original features
        if X is not None:
            assert X.nrows == len(actual)
            assert X.ncols >= 1
            X_pd = X.to_pandas()

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
        cms['metric'] = cms[['tp', 'fp', 'tn', 'fn']].apply(lambda x: self.protected_metric(*x, X_pd), axis=1, raw=True)
        return cms['metric'].mean()  # in case of ties
