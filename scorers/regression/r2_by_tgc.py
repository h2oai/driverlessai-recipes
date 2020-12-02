"""Custom R2 scorer computes R2 on each time series, then averages them out for the final score."""

import numpy as np
import datatable as dt
from datatable import Frame, f, mean, by, count, join, cbind, isna
import typing
from typing import List
from h2oaicore.metrics import CustomScorer, R2Scorer
from h2oaicore.systemutils import config
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


class R2byTimeSeries(CustomScorer):
    _description = "R2 on each time series, then averaged"
    _display_name = "Mean R2 by TS"
    _maximize = True  # whether a higher score is better
    _perfect_score = 1.0  # the ideal score, used for early stopping once validation score achieves this value

    _supports_sample_weight = True  # whether the scorer accepts and uses the sample_weight input

    """Please enable the problem types this scorer applies to"""
    _regression = True
    _binary = False
    _multiclass = False

    """
    Whether the dataset itself is required to score (in addition to actual and predicted columns).
    If set to True, X will be passed as a datatable Frame, and can be converted to pandas via X.to_pandas() if needed.
    """
    _needs_X = True

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

    @staticmethod
    def acceptance_test_timeout():
        """
        Timeout in minutes for each test of a custom recipe.
        """
        return config.acceptance_test_timeout

    def get_experiment_logger(self):
        logger = None
        if self.context and self.context.experiment_id:
            logger = make_experiment_logger(
                experiment_id=self.context.experiment_id,
                tmp_dir=self.context.tmp_dir,
                experiment_tmp_dir=self.context.experiment_tmp_dir
            )
        return logger

    def score(
            self,
            actual: np.array,
            predicted: np.array,
            sample_weight: typing.Optional[np.array] = None,
            labels: typing.Optional[List[any]] = None,
            X: typing.Optional[dt.Frame] = None,
            **kwargs) -> float:

        # Get the logger if it exists
        logger = self.get_experiment_logger()

        # hard-coded as access to experiment parameters (such as self.tgc) not yet available
        tgc = ["Store", "Dept"]
        # tgc = ["state"]
        # tgc = None

        # enable weighted average over TS R2 scores: weighted based on TS share of rows
        isR2AverageWeighted = False

        if tgc is None or not all(col in X.names for col in tgc):
            loggerinfo(logger, f"TS R2 computes single R2 on {X.nrows} rows as either tgc {tgc} is not defined or incorrect.")
            return R2Scorer().score(actual, predicted, sample_weight, labels, **kwargs)
        else:
            tgc_values = X[:, {"weight": count()/X.nrows, "r2": 0.0}, by(tgc)]
            loggerinfo(logger, f"TS R2 computes multiple R2 on {X.nrows} rows, tgc {tgc} with weighting is {isR2AverageWeighted}.")
            none_values = [None] * X.nrows
            X = cbind(X[:, tgc], Frame(actual = actual, predicted = predicted,
                                       sample_weight = sample_weight if sample_weight is not None else none_values))

            for i in range(0, tgc_values.nrows):
                current_tgc = tgc_values[i, :]
                current_tgc.key = tgc
                ts_frame = X[:, :, join(current_tgc)][~isna(f.r2), :]
                r2_score = R2Scorer().score(ts_frame['actual'].to_numpy(), ts_frame['predicted'].to_numpy(),
                                             ts_frame['sample_weight'].to_numpy() if sample_weight is not None else None,
                                             labels, **kwargs)
                tgc_values[i, f.r2] = r2_score

                loggerinfo(logger, f"TS R2 = {r2_score} on {ts_frame.nrows} rows, tgc = {current_tgc[0, tgc].to_tuples()}")

            if isR2AverageWeighted:
                # return np.average(tgc_values["r2"].to_numpy(), weights=tgc_values["weight"].to_numpy())
                return tgc_values[:, mean(f.r2 * f.weight)][0, 0]
            else:
                return tgc_values[:, mean(f.r2)][0, 0]
