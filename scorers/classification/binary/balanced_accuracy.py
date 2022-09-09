"""balanced_accuracy_score"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer, prep_actual_predicted, threshold_scorer, label_binarize

class balancedaccuracy(CustomScorer):
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 1.0 if _maximize else 0.0
    _worst_score = 0.0 if _maximize else 1.0
    #_is_for_user = False  # don't let the user pick since can be trivially optimized (even when using F1-optimal thresh)
    _desc = "(weighted) Balanced Accuracy is defined as the average of recall obtained on each class."

    @classmethod
    def _threshold_optimizer(cls):
        return threshold_scorer()

    @staticmethod
    def _metric(tp, fp, tn, fn):
        return 0.5*(((tp/(tp+fn))+(tn/(tn+fp))))
    
    def protected_metric(self, tp, fp, tn, fn):
        try:
            ret = self.__class__._metric(tp, fp, tn, fn)
            if np.isnan(ret):
                # case of 0/0 - same behavior
                raise ZeroDivisionError
        except ZeroDivisionError:
            ret = 0 if self.__class__._maximize else 1  # return worst score if ill-defined
        return ret

    def score(self, actual, predicted, sample_weight=None, labels=None, **kwargs):
        if sample_weight is not None:
            sample_weight = sample_weight.ravel()
        if len(actual) == 1:
            return 0 if self.__class__._maximize else 1  # return worst score if have only 1 row - even though might be better
        enc_actual, enc_predicted, labels = prep_actual_predicted(actual, predicted, labels)
        cm_weights = sample_weight if sample_weight is not None else None

        # multiclass
        if enc_predicted.shape[1] > 1:
            enc_predicted = enc_predicted.ravel()
            enc_actual = label_binarize(enc_actual, labels).ravel()
            cm_weights = np.repeat(cm_weights, predicted.shape[1]).ravel() if cm_weights is not None else None
            assert enc_predicted.shape == enc_actual.shape
            assert cm_weights is None or enc_predicted.shape == cm_weights.shape

        import h2o4gpu.util.metrics as daicx
        cms = daicx.confusion_matrices(enc_actual.ravel(), enc_predicted.ravel(), sample_weight=cm_weights)
        cms = cms.loc[cms[[self._threshold_optimizer()]].idxmax()]  # get row(s) for optimal threshold-defining metric
        cms['metric'] = cms[['tp', 'fp', 'tn', 'fn']].apply(lambda x: self.protected_metric(*x), axis=1, raw=True)
        return cms['metric'].mean()  # in case of ties
