"""Custom Anomaly Score for Isolation Forest"""
import numpy as np
from h2oaicore.metrics import CustomUnsupervisedScorer
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.models_unsupervised import IsolationForestAnomalyModel


class MyAnomalyScorer(CustomUnsupervisedScorer):
    _perfect_score = 1e30
    _maximize = True

    def score(self, actual, predicted, sample_weight=None, labels=None, X=None, **kwargs):
        # Custom scorer for anomaly scores to allow unsupervised AutoML.

        # Idea: just return the variance of the predicted anomaly scores

        # This blindly assumes that higher variance in anomaly score is better.
        # WARNING: This is a strong assumption, and might not be true or useful.

        # Please customize using your domain knowledge!
        # Try to gain insight into anomaly score distributions first, before making a scorer.
        # X is provided here. E.g., one can do aggregations of anomaly scores across specific cohorts.
        return np.var(predicted)


class MyIsolationForestAnomalyModel(IsolationForestAnomalyModel, CustomUnsupervisedModel):
    _included_scorers = ['MyAnomalyScorer']

    # Feature Engineering is controlled by the pretransformers (one-time, upfront).
    # Can use any custom transformer that takes raw input and returns numeric data, anything goes.
    # By default, categorical features are converted to numerics using frequency encoding.
    # By default, dates/text/images are dropped. But this can be changed arbitrarily.

    # _included_pretransformers = ['OrigFreqPreTransformer']  # default - pass-through numerics, frequency-encode categoricals, drop rest
    # _included_pretransformers = ['OrigOHEPreTransformer']   # option  - pass-through numerics, one-hot-encode categoricals, drop rest
