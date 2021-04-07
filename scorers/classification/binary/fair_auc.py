"""Custom scorer for detecting and reducing bias in machine learning models."""

"""
Custom scorer for detecting and reducing bias in machine learning models.
Based upon  https://arxiv.org/abs/1903.04561.

The scorer penalizes models/features which favour a particular subgroup (the "privileged group") over another group
(the "unprivileged group").

To use this metric
    - Encode the privileged/ unprivileged groups in a column and set PRIVILEGED_GROUP_NAME to the column name used in
      the DataFrame. The column should be 1 if the sample belongs to a privileged group and 0 otherwise.
      Both train and validation set must contain this column.
    - Upload the custom recipe in the expert settings tab.
    - Set DisparateGroupRemover as a pretransformer in DAI expert settings. It will remove the PRIVILEGED_GROUP_NAME
      column from the dataset, such that won't be used for modelling.
    - Disable DisparateGroupRemover in the transformers tab, if it is enabled.
    - Set FairAUC as the custom Scorer.
"""
import typing

import numpy as np
from datatable import dt
from sklearn.metrics import roc_auc_score

from h2oaicore.metrics import CustomScorer
from h2oaicore.transformer_utils import CustomTransformer

PRIVILEGED_GROUP_NAME = 'male'


class FairAUC(CustomScorer):
    """
    Inspired by https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
    For a given base metric M(y_true, y_pred), FairMetric is the average of the
        - base metric
        - averaged metric over the individual subgroups
        - BPSN metric
        - BNSP metric
    See also https://arxiv.org/abs/1903.04561
    """
    _binary = True
    _regression = False
    _multiclass = False

    _description = "Scorer with subgroup AUC"
    _display_name = "FAIRAUC"

    _supports_sample_weight = False  # whether the scorer accepts and uses the sample_weight input
    _needs_X = True  # This assures that X is passed in

    _maximize = True
    _perfect_score = 1

    @staticmethod
    def do_acceptance_test():
        return False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              X: dt.Frame = None,
              **kwargs) -> float:
        if PRIVILEGED_GROUP_NAME not in X.names:
            # May happen in leakage/shift detection of single features.
            return self._compute_base_metric(actual, predicted)

        mask_privileged_group = (X.to_pandas()[PRIVILEGED_GROUP_NAME].to_numpy().astype(int).flatten() == 1)
        if len(set(mask_privileged_group)) == 1:  # only one group in the data.
            return self._compute_base_metric(actual, predicted)

        scores = [self._compute_base_metric(actual, predicted),
                  self._compute_subgroup_metric(actual, predicted, mask_privileged_group),
                  self._compute_bnsp_metric(actual, predicted, mask_privileged_group),
                  self._compute_bpsn_metric(actual, predicted, mask_privileged_group)]
        return np.average(scores,
                          weights=[0.25, 0.25, 0.25, 0.25])

    def _compute_base_metric(self, y_true, y_pred):
        try:
            return roc_auc_score(np.array(y_true).astype(int), y_pred)
        except ValueError as e:
            return 0.5

    def _compute_subgroup_metric(self, y_true, y_pred, mask_privileged_group):
        return 0.5 * (self._compute_base_metric(y_true[mask_privileged_group], y_pred[mask_privileged_group]) +
                      self._compute_base_metric(y_true[~mask_privileged_group], y_pred[~mask_privileged_group]))

    def _compute_bpsn_metric(self, y_true, y_pred, mask_privileged_group):
        mask_bp = mask_privileged_group & (y_true > 0.5)
        mask_sn = ~mask_privileged_group & (y_true < 0.5)
        mask = mask_bp | mask_sn
        return self._compute_base_metric(y_true[mask], y_pred[mask])

    def _compute_bnsp_metric(self, y_true, y_pred, mask_privileged_group):
        mask_bn = mask_privileged_group & (y_true < 0.5)
        mask_sp = ~mask_privileged_group & (y_true > 0.5)
        mask = mask_bn | mask_sp
        return self._compute_base_metric(y_true[mask], y_pred[mask])


class DisparateGroupRemover(CustomTransformer):
    """
    Use this transformer as a pretransformer to have access to the PRIVILEGED_GROUP_NAME within the scorer,
    but to drop it during the modelling process.
    """

    @staticmethod
    def do_acceptance_test():
        return False

    def transform(self, X: dt.Frame, y: np.array = None):
        if PRIVILEGED_GROUP_NAME in X.names:
            X = X[:, [name for name in X.names if name != PRIVILEGED_GROUP_NAME]]
        return X

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X, y)
