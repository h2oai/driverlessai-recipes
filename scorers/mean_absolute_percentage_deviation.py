import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import median_absolute_error

class MyMeanAbsolutePercentageDeviation(CustomScorer):
    _description = "Mean absolute percentage deviation - per prediction (not actual)"
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "MAPD"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        return np.mean(abs((actual - predicted)/predicted))