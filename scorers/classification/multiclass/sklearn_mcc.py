"""Template base class for a custom scorer recipe."""

import numpy as np
import datatable as dt
import typing
from sklearn.metrics import matthews_corrcoef
from h2oaicore.metrics import prep_actual_predicted, CustomScorer
from sklearn.preprocessing import label_binarize

_global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file


class SKLearnMCC(CustomScorer):
    _description = "Scikit-learn's MCC"
    _maximize = True  # whether a higher score is better
    _perfect_score = 1.0  # the ideal score, used for early stopping once validation score achieves this value

    _supports_sample_weight = True  # whether the scorer accepts and uses the sample_weight input

    _regression = False
    _binary = False
    _multiclass = True

    def score(
            self,
            actual: np.array,
            predicted: np.array,
            sample_weight: typing.Optional[np.array] = None,
            labels: typing.Optional[np.array] = None,
            X: typing.Optional[dt.Frame] = None,
            **kwargs) -> float:
        """Please implement this function to compute a score from actual and predicted values.

        Args:
            actual (:obj:`np.array`): actual values from target column
                (1 column, 1 numeric or string value per row)
            predicted (:obj:`np.array`): predicted numeric values
                (1 column for regression and binary classification, p columns for p-class problem)
            sample_weight (:obj:`np.array`): Optional, observation weights for each sample
                (1 column, 1 numeric value per row)
            labels (:obj:`List[any]`): Optional, class labels (or `None` for regression)
            X (:obj:`dt.Frame`): Optional, datatable Frame containing dataset

        Returns:
            float: score

        """
        cm_weights = sample_weight
        enc_actual, enc_predicted, labels = prep_actual_predicted(actual, predicted, labels)
        idx = np.argmax(enc_predicted, axis=1)
        enc_predicted = np.zeros_like(enc_predicted)
        enc_predicted[np.arange(len(enc_predicted)), idx] = 1
        enc_actual = label_binarize(enc_actual, labels)

        enc_predicted = enc_predicted.ravel()
        enc_actual = enc_actual.ravel()
        cm_weights = np.repeat(cm_weights, predicted.shape[1]).ravel() if cm_weights is not None else None
        assert enc_predicted.shape == enc_actual.shape
        assert cm_weights is None or enc_predicted.shape == cm_weights.shape

        return matthews_corrcoef(enc_actual, enc_predicted, cm_weights)
