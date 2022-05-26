"""Tweedie Deviance scorer recipe.
User inputs can be provided through recipe_dict in config.
To pass power parameter
recipe_dict = "{'power':2.0}"
The default value is 1.5

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


class TweedieDeviance(CustomScorer):
    _description = NotImplemented
    _maximize = False  # whether a higher score is better
    _perfect_score = 0.0  # the ideal score, used for early stopping once validation score achieves this value
    _supports_sample_weight = True  # whether the scorer accepts and uses the sample_weight input
    _regression = True
    _display_name = "Tweedie_Deviance"

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

        :param actual:          Ground truth (correct) target values.
        :param predicted:       Estimated target values
        :param sample_weight:   weights
        :param labels:          not used

        power:                  default=1.5 sent to function via toml dictionary
        Tweedie power parameter. Either power <= 0 or power >= 1.
        To non-default power parameter use recipe_dict add via toml config DAI option. Example:
        recipe_dict = "{'power':2.0}"
        Multiple parameters example (first param is for demo only):
        validate_meta_learner=false\nrecipe_dict = "{'power':2.0}"

        The higher p the less weight is given to extreme deviations between true and predicted targets.
        power < 0: Extreme stable distribution. Requires: y_pred > 0.
        power = 0 : Normal distribution, output corresponds to mean_squared_error. y_true and y_pred can be any real numbers.
        power = 1 : Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
        1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
        power = 2 : Gamma distribution. Requires: y_true > 0 and y_pred > 0.
        power = 3 : Inverse Gaussian distribution. Requires: y_true > 0 and y_pred > 0.
        otherwise : Positive stable distribution. Requires: y_true > 0 and y_pred > 0.

        :return: score
        """

        try:
            """Initialize logger to print additional info in case of invalid inputs(exception is raised) and to enable debug prints"""
            logger = self.logger
            from h2oaicore.systemutils import loggerinfo
            #loggerinfo(logger, "Start TW Deviance Scorer.......")
            #loggerinfo(logger, 'Actual:%s' % str(actual))
            #loggerinfo(logger, 'Predicted:%s' % str(predicted))
            #loggerinfo(logger, 'Sample W:%s' % str(sample_weight))

            from sklearn.metrics import mean_tweedie_deviance
            if config.recipe_dict is not None:
                power = config.recipe_dict.get('power', 1.5)
            else:
                power = 1.5

            #loggerinfo(logger, 'Power:%s' % str(power))
            if 0 < power < 1:
                loggerinfo(logger, 'Power:%s' % str(power))
                loggerinfo(logger,
                           """Invalid power value. Power should be one of the following: \n
                            power < 0: Extreme stable distribution. Requires: y_pred > 0.
                            power = 0 : Normal distribution, output corresponds to mean_squared_error. y_true and y_pred can be any real numbers.
                            power = 1 : Poisson distribution.          Requires: y_true >= 0 and y_pred > 0.
                            1 < p < 2 : Compound Poisson distribution. Requires: y_true >= 0 and y_pred > 0.
                            power = 2 : Gamma distribution.            Requires: y_true > 0 and y_pred > 0.
                            power = 3 : Inverse Gaussian distribution. Requires: y_true > 0 and y_pred > 0.
                            otherwise : Positive stable distribution.  Requires: y_true > 0 and y_pred > 0.""")
                raise RuntimeError(
                    'Error during Tweedie Deviance score calculation. Invalid power value.')

            actual = actual.astype('float64')
            predicted = predicted.astype('float64')
            '''Safety mechanizm in case predictions or actuals are zero'''
            epsilon = 1E-8
            actual += epsilon
            predicted += epsilon

            if power == 0:
                '''No need to validate sign of actual or predicted'''
                pass
            elif power < 0:
                if (predicted <= 0).any():
                    loggerinfo(logger, 'Predicted:%s' % str(predicted))
                    loggerinfo(logger, 'Invalid Predicted:%s' % str(predicted[predicted <= 0]))
                    raise RuntimeError(
                        'Power <0. Error during Tweedie Deviance score calculation. Invalid predicted values. Expecting only positive values')
            elif 1 <= power < 2:
                if (actual < 0).any():
                    loggerinfo(logger, 'Actual:%s' % str(actual))
                    loggerinfo(logger, 'Non-positive Actuals:%s' % str(actual[actual < 0]))
                    raise RuntimeError(
                        '1 <= power < 2. Error during Tweedie Deviance score calculation. Invalid actuals values. Expecting zero or positive values')
                if (predicted <= 0).any() or np.isnan(np.sum(predicted)):
                    loggerinfo(logger, 'Predicted:%s' % str(predicted))
                    loggerinfo(logger, 'Invalid Predicted:%s' % str(predicted[predicted <= 0]))
                    raise RuntimeError(
                        '1 <= power < 2. Error during Tweedie Deviance score calculation. Invalid predicted values. Expecting only positive values')
            elif power >= 2:
                if (actual <= 0).any():
                    loggerinfo(logger, 'Actual:%s' % str(actual))
                    loggerinfo(logger, 'Non-positive Actuals:%s' % str(actual[actual <= 0]))
                    raise RuntimeError(
                        'power >= 2. Error during Tweedie Deviance score calculation. Invalid actuals values. Expecting zero or positive values')
                if (predicted <= 0).any() or np.isnan(np.sum(predicted)):
                    loggerinfo(logger, 'Predicted:%s' % str(predicted))
                    loggerinfo(logger, 'Invalid Predicted:%s' % str(predicted[predicted <= 0]))
                    raise RuntimeError(
                        'power >= 2. Error during Tweedie Deviance score calculation. Invalid predicted values. Expecting only positive values')

            '''Check if any element of the arrays is nan'''
            if np.isnan(np.sum(actual)):
                loggerinfo(logger, 'Actual:%s' % str(actual))
                loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(actual))))
                raise RuntimeError(
                    'Error during Tweedie Deviance score calculation. Invalid actuals values. Expecting only non-nan values')
            if np.isnan(np.sum(predicted)):
                loggerinfo(logger, 'Predicted:%s' % str(predicted))
                loggerinfo(logger, 'Nan values index:%s' % str(np.argwhere(np.isnan(predicted))))
                raise RuntimeError(
                    'Error during Tweedie Deviance score calculation. Invalid predicted values. Expecting only non-nan values')

            score = mean_tweedie_deviance(actual, predicted, sample_weight=sample_weight, power=power)
        except Exception as e:
            raise
        return score


