"""Weighted Recall: `TP / (TP + FN)` at threshold for optimal F1 Score."""

import typing
import numpy as np
from h2oaicore.metrics import CustomScorer, prep_actual_predicted
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize


class Recall(CustomScorer):
    _description = "(weighted) Recall at threshold for optimal F1 Score"
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 1.0
    _display_name = "Recall"
    # True: one-vs-rest, F1-optimal threshold per class with recall averaged over classes
    # False: micro-average, all classes pooled under a single F1-optimal threshold
    _multiclass_one_vs_rest = False

    @staticmethod
    def _f1_opt_recall(y_true, y_score, weights):
        if len(np.unique(y_true)) < 2:
            return None
        precision, recall, _ = precision_recall_curve(
            y_true, y_score, sample_weight=weights
        )
        denom = precision + recall
        numerator = 2 * precision * recall
        f1 = np.divide(numerator, denom, out=np.zeros_like(denom), where=denom > 0)
        return float(recall[f1 == f1.max()].mean())  # in case of ties

    def score(
        self,
        actual: np.array,
        predicted: np.array,
        sample_weight: typing.Optional[np.array] = None,
        labels: typing.Optional[np.array] = None,
        **kwargs,
    ) -> float:

        if sample_weight is not None:
            sample_weight = sample_weight.ravel()

        enc_actual, enc_predicted, labels = prep_actual_predicted(
            actual, predicted, labels
        )

        if enc_predicted.shape[1] == 1:
            ret = self._f1_opt_recall(
                enc_actual.ravel(), enc_predicted.ravel(), sample_weight
            )
            return 0.0 if ret is None else ret

        enc_actual = label_binarize(enc_actual, classes=labels)

        if self._multiclass_one_vs_rest:
            # classes absent from actual have undefined recall and are skipped
            per_class = [
                self._f1_opt_recall(
                    enc_actual[:, k], enc_predicted[:, k], sample_weight
                )
                for k in range(enc_predicted.shape[1])
            ]
            per_class = [r for r in per_class if r is not None]
            return float(np.mean(per_class)) if per_class else 0.0

        weights = (
            np.repeat(sample_weight, enc_predicted.shape[1])
            if sample_weight is not None
            else None
        )
        ret = self._f1_opt_recall(enc_actual.ravel(), enc_predicted.ravel(), weights)
        return 0.0 if ret is None else ret
