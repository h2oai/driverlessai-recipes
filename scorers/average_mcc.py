"""Averaged Matthews Correlation Coefficient (averaged over several thresholds, for imbalanced problems)"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import matthews_corrcoef
from sklearn.preprocessing import LabelEncoder


class MyAverageMCCScorer(CustomScorer):
    _description = "Average MCC over several thresholds"
    _binary = True
    _maximize = True
    _perfect_score = 0
    _display_name = "AVGMCC"
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        """Reasoning behind using several thresholds
        MCC can vary a lot from one threshold to another
        and especially may give different results on train and test datasets
        Using an average over thresholds close to the prior may lead to a flatter
        response and better generalization.
        """
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)

        # Compute thresholds
        prior = np.mean(actual)
        thresholds = [rate * prior for rate in np.arange(0.8, 1.3, 0.1)]
        if sample_weight is not None:
            raise NotImplementedError("sklearn MCC has buggy implementation of sample_weights")

        # Compute average MCC for the thresholds
        avg_score = 0
        for t in thresholds:
            avg_score += matthews_corrcoef(
                y_true=actual,
                y_pred=(predicted > t).astype(np.uint8),
                sample_weight=sample_weight
            )

        return avg_score / len(thresholds)
