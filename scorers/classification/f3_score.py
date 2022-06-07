"""F3 Score"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer, prep_actual_predicted
from sklearn.preprocessing import label_binarize
import h2o4gpu.util.metrics as daicx


class F3Scorer(CustomScorer):
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _display_name = "F3 Score"
    _threshold_optimizer = "f2"  # closest match to F3

    @staticmethod
    def _metric(tp, fp, tn, fn):
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        beta2 = 3 ** 2  # F3
        return (1 + beta2) * (precision * recall) / ((beta2 * precision) + recall)

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
