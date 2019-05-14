import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
import sklearn


class MyFalseDiscoveryRateScorer(CustomScorer):
    _threshold = 0.1
    _description = "My False Discovery Rate for Binary Classification [threshold=%f]." % _threshold
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "FDR"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        predicted = predicted >= self.__class__._threshold  # probability -> label
        cm = sklearn.metrics.confusion_matrix(actual, predicted, sample_weight=sample_weight, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        if (fp + tp) != 0:
            return fp / (fp + tp)
        else:
            return 0
