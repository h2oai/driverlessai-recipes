"""Kolmogorov-Smirnov scorer recipe.

If you need to print debug messages into DAI log, uncomment lines with logger and loggerinfo.

Starting 1.10.2 - DAI handles exceptions raised by custom scorers.
Default DAI behavior is to continue experiment in case of Scorer failure.
To enable forcing experiment to fail, in case of scorer error, set following parameters in DAI:
  - skip_scorer_failures=false (Disabled)
  - skip_model_failures=false (Disabled)
"""

import numpy as np
from scipy.stats import ks_2samp
import typing
from h2oaicore.metrics import CustomScorer
from h2oaicore.systemutils import print_debug
import logging
from h2oaicore.systemutils import config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


class KolmogorovSmirnov(CustomScorer):
    _description = NotImplemented
    _maximize = True  # whether a higher score is better
    _perfect_score = 1.0  # the ideal score, used for early stopping once validation score achieves this value
    _supports_sample_weight = False  # whether the scorer accepts and uses the sample_weight input
    _binary = True
    _regression = False
    _multiclass = False
    _display_name = "Kolmogorov_Smirnov"

    def __init__(self):
        CustomScorer.__init__(self)

    @staticmethod
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.
        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return False

    @property
    def logger(self):
        from h2oaicore import application_context
        from h2oaicore.systemutils import exp_dir
        # Don't assign to self, not picklable
        return make_experiment_logger(experiment_id=application_context.context.experiment_id, tmp_dir=None,
                                      experiment_tmp_dir=exp_dir())


    def score(self, actual: np.array, predicted: np.array, sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        """

        :param actual:          Ground truth (correct) target values. Requires actual > 0.
        :param predicted:       Predicted probabilities
        :param sample_weight:   weights
        :param labels:          Dataset labels


        :return: score
        """

        try:
            """Initialize logger to print additional info in case of invalid inputs(exception is raised) and to enable debug prints"""
            logger = self.logger
            from h2oaicore.systemutils import loggerinfo
            # loggerinfo(logger, "Start Gamma Deviance Scorer.......")
            # loggerinfo(logger, 'Actual:%s' % str(actual))
            # loggerinfo(logger, 'Predicted:%s' % str(predicted))
            # loggerinfo(logger, 'Sample W:%s' % str(sample_weight))

            from scipy.stats import ks_2samp

            # loggerinfo(logger, 'Actual:%s' % str(actual))
            # loggerinfo(logger, 'Predicted:%s' % str(predicted))
            # loggerinfo(logger, 'Labels:%s' % str(labels))

            predicted = predicted.astype('float64')

            '''Check if any element of the arrays is nan'''
            if np.isnan(np.sum(predicted)):
                loggerinfo(logger, 'Predicted:%s' % str(predicted))
                loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(predicted))))
                raise RuntimeError(
                    'Error during KS score calculation. Invalid predicted values. Expecting only non-nan values')

            class_0 = predicted[actual == labels[0]]
            class_1 = predicted[actual == labels[1]]

            score = ks_2samp(class_0, class_1)[0]

            '''Validate that score is between 0 and 1'''
            if 0 <= score <= 1:
                pass
            else:
                loggerinfo(logger, 'Invalid calculated score:%s' % str(score))
                raise RuntimeError(
                    'Error during KS score calculation. Invalid calculated score:%s. \
                     Score should be between 0 and 1 . Nan is not valid' % str(score))
        except Exception as e:
            '''Print error message into DAI log file'''
            loggerinfo(logger, 'Error during KS score calculation. Exception raised: %s' % str(e))
            raise
        return score


