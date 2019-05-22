import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
import sklearn


class MyAverageMCCScorer(CustomScorer):
    _threshold = 0.1
    _description = "Average MCC over several thresholds"
    _binary = True
    _maximize = True
    _perfect_score = 0
    _display_name = "AVGMCC"

    @staticmethod
    def my_mcc(actual, predicted):
        tp = np.sum((actual == 1) & (predicted == 1))
        tn = np.sum((actual == 0) & (predicted == 0))
        fp = np.sum((actual == 0) & (predicted == 1))
        fn = np.sum((actual == 1) & (predicted == 0))

        numerator = (tp * tn - fp * fn)
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5

        return numerator / (denominator + 1e-15)

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
        # If actual is provided as a class label
        # then use Label Encoding first on ground truth
        if labels is not None:
            lb = sklearn.preprocessing.LabelEncoder()
            actual = lb.fit_transform(actual)

        # Compute thresholds
        prior = np.mean(actual)
        thresholds = [rate * prior for rate in np.arange(0.8, 1.3, 0.1)]

        # Compute average MCC for the thresholds
        avg_score = 0
        for t in thresholds:
            avg_score += sklearn.metrics.matthews_corrcoef(
                y_true=actual,
                y_pred=(predicted > t).astype(np.uint8),
                sample_weight=sample_weight
            )

        return avg_score / len(thresholds)
