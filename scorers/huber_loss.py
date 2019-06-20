"""Huber Loss for Regression or Binary Classification. Robust loss, combination of quadratic loss and linear loss."""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder


# temp. references:
# Wikipedia: https://en.wikipedia.org/wiki/Huber_loss
# https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
# Tukey loss: https://web.as.uky.edu/statistics/users/pbreheny/764-F11/notes/12-1.pdf

class MyHuberLossScorer(CustomScorer):
    '''
    Huber Loss Scorer is a loss function used in robust regression, that is less 
    sensitive to outliers in data than the squared error loss. This custom
    scorer supports both regression and binary binary classification problems
    using different formulas and different defaults for delta (see below).
    For more details see: https://en.wikipedia.org/wiki/Huber_loss
    
    Parameters
    ----------
    delta : numeric
        Hyperparemeter with defaults =1.345 for regression and =0.1 for binary
        classification
        
    '''
    _delta_regression = 1.345
    _delta_binary = 0.1
    _description = "My Huber Loss for Regression or Binary Classification [delta=%f or %f]." % (
    _delta_regression, _delta_binary)
    _binary = True
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Huber"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:

        isRegression = True if labels is None else False
        delta = MyHuberLossScorer._delta_regression if isRegression else MyHuberLossScorer._delta_binary
        if delta < 0: delta = 0
        if not isRegression:
            lb = LabelEncoder()
            labels = lb.fit_transform(labels)
            actual = lb.transform(actual)
            all0s = np.zeros(actual.shape[0])

        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])

        if isRegression:
            abs_error = np.abs(np.subtract(actual, predicted))
            loss = np.where(abs_error < delta, .5 * (abs_error) ** 2, delta * (abs_error - 0.5 * delta))
        else:
            predicted = np.subtract(np.multiply(predicted, 2), 1)
            actual = np.where(actual == 0, -1, 1)
            actual_mult_predict = np.multiply(actual, predicted)
            loss = np.where(actual_mult_predict >= -1,
                            np.square(np.maximum(all0s, np.subtract(1, actual_mult_predict))),
                            -4 * actual_mult_predict)

        return np.mean(loss) if actual.shape[0] > 0 else 0
