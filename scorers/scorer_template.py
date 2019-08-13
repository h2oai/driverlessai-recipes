"""Template base class for a custom scorer recipe."""

import numpy as np
import typing

_global_modules_needed_by_name = []  # Optional global package requirements, for multiple custom recipes in a file


class CustomScorer(BaseScorer):
    _description = NotImplemented
    _maximize = True  # whether a higher score is better
    _perfect_score = 1.0  # the ideal score, used for early stopping once validation score achieves this value

    _supports_sample_weight = True  # whether the scorer accepts and uses the sample_weight input

    """Please enable the problem types this scorer applies to"""
    _regression = False
    _binary = False
    _multiclass = False

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    _modules_needed_by_name = []  # List[str]

    @staticmethod
    def is_enabled():
        """Toggle to enable/disable recipe. If disabled, recipe will be completely ignored."""
        return True

    @staticmethod
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return True

    def score(
            self,
            actual: np.array,
            predicted: np.array,
            sample_weight: typing.Optional[np.array] = None,
            labels: typing.Optional[List[any]] = None) -> float:
        """Please implement this function to compute a score from actual and predicted values.

        Args:
            actual (:obj:`np.array`): actual values from target column
                (1 column, 1 numeric or string value per row)
            predicted (:obj:`np.array`): predicted numeric values
                (1 column for regression and binary classification, p columns for p-class problem)
            sample_weight (:obj:`np.array`): Optional, observation weights for each sample
                (1 column, 1 numeric value per row)
            labels (:obj:`List[any]`): Optional, class labels (or `None` for regression)

        Returns:
            float: score

        """
        raise NotImplementedError
