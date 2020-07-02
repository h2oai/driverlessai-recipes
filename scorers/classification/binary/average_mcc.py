"""Averaged Matthews Correlation Coefficient (averaged over several thresholds, for imbalanced problems). Example how to use Driverless AI's internal scorer."""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from h2oaicore.metrics import MccScorer
from sklearn.preprocessing import LabelEncoder


class MyAverageMCCScorer(CustomScorer):
    _description = "Average MCC over several thresholds"
    _binary = True
    _maximize = True
    _perfect_score = 1
    _display_name = "AVGMCC"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:
        """Reasoning behind using several thresholds
        MCC can vary a lot from one threshold to another
        and especially may give different results on train and test datasets
        Using an average over thresholds close to the prior may lead to a flatter
        response and better generalization.
        """
        lb = LabelEncoder()
        labels = list(lb.fit_transform(labels))
        actual = lb.transform(actual)

        # Compute thresholds
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])
        prior = np.sum(actual * sample_weight) / np.sum(sample_weight)
        thresholds = [rate * prior for rate in np.arange(0.8, 1.3, 0.1)]

        # Compute average MCC for the thresholds
        avg_score = 0
        for t in thresholds:
            avg_score += MccScorer().score(
                actual=actual,
                predicted=(predicted > t).astype(np.uint8),
                sample_weight=sample_weight,
                labels=labels
            )

        return avg_score / len(thresholds)
