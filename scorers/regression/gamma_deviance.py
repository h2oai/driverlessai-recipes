"""Gamma Deviance scorer recipe.
This is same as Tweedie Deviance scorer with power=2

If you need to print debug messages into DAI log, uncomment lines with logger and loggerinfo.

Starting 1.10.2 - DAI handles exceptions raised by custom scorers.
Default DAI behavior is to continue experiment in case of Scorer failure.
To enable forcing experiment to fail, in case of scorer error, set following parameters in DAI:
  - skip_scorer_failures=false (Disabled)
  - skip_model_failures=false (Disabled)
"""

import numpy as np
import typing
from h2oaicore.metrics import CustomScorer
from h2oaicore.systemutils import print_debug
import logging
from h2oaicore.systemutils import config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


class GammaDeviance(CustomScorer):
    _description = NotImplemented
    _maximize = False  # whether a higher score is better
    _perfect_score = 0.0  # the ideal score, used for early stopping once validation score achieves this value
    _supports_sample_weight = True  # whether the scorer accepts and uses the sample_weight input
    _regression = True
    _display_name = "Gamma_Deviance"

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
        :param predicted:       Estimated target values. Requires predicted > 0.
        :param sample_weight:   weights
        :param labels:          not used


        :return: score
        """

        try:
            """Initialize logger to print additional info in case of invalid inputs(exception is raised) and to enable debug prints"""
            logger = self.logger
            from h2oaicore.systemutils import loggerinfo
            #loggerinfo(logger, "Start Gamma Deviance Scorer.......")
            #loggerinfo(logger, 'Actual:%s' % str(actual))
            #loggerinfo(logger, 'Predicted:%s' % str(predicted))
            #loggerinfo(logger, 'Sample W:%s' % str(sample_weight))

            from sklearn.metrics import mean_gamma_deviance

            actual = actual.astype('float64')
            predicted = predicted.astype('float64')
            '''Safety mechanizm in case predictions or actuals are zero'''
            epsilon = 1E-8
            actual += epsilon
            predicted += epsilon
            if (actual <= 0).any():
                loggerinfo(logger, 'Actual:%s' % str(actual))
                loggerinfo(logger, 'Non-positive Actuals:%s' % str(actual[actual <= 0]))
                raise RuntimeError(
                    'Error during Gamma Deviance score calculation. Invalid actuals values. Expecting positive values')
            if (predicted <= 0).any() or np.isnan(np.sum(predicted)):
                loggerinfo(logger, 'Predicted:%s' % str(predicted))
                loggerinfo(logger, 'Invalid Predicted:%s' % str(predicted[predicted <= 0]))
                raise RuntimeError(
                    'Error during Gamma Deviance score calculation. Invalid predicted values. Expecting only positive values')

            '''Check if any element of the arrays is nan'''
            if np.isnan(np.sum(actual)):
                loggerinfo(logger, 'Actual:%s' % str(actual))
                loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(actual))))
                raise RuntimeError(
                    'Error during Gamma Deviance score calculation. Invalid actuals values. Expecting only non-nan values')
            if np.isnan(np.sum(predicted)):
                loggerinfo(logger, 'Predicted:%s' % str(predicted))
                loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(predicted))))
                raise RuntimeError(
                    'Error during Gamma Deviance score calculation. Invalid predicted values. Expecting only non-nan values')

            score = mean_gamma_deviance(actual, predicted, sample_weight=sample_weight)
        except Exception as e:
            raise
        return score


