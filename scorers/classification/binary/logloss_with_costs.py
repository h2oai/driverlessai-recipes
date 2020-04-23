"""Logloss with costs associated with each type of 4 outcomes - typically applicable to fraud use case"""

import typing
import numpy as np
import datatable as dt
from datatable import f
from h2oaicore.metrics import CustomScorer
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning, loggerdata
from sklearn.preprocessing import LabelEncoder

class LoglossWithCostsBinary(CustomScorer):
    _description = "Embed costs in logloss for binary classification: `(fp_cost*FP + tp_cost*TP + fn_cost*FN + tn_cost*TN) / N`"
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "LoglossCosts"
    _needs_X = True
    _epsilon = 1e-15
    _make_logger = True  # set to True to make logger

    """
    References:
    https://towardsdatascience.com/fraud-detection-with-cost-sensitive-machine-learning-24b8760d35d9
    https://medium.com/datadriveninvestor/rethinking-the-right-metrics-for-fraud-detection-4edfb629c423
    
    Why regular binary logloss don’t work well for fraud detection. Because they consider the costs for every 
    mistake the same. In the real world, it’s totally not true. For example, customer A makes a transaction and 
    the system predicts that it’s a fraud order, then cancel the order. The cost for the company is the customer 
    service fee when customer A contacts the company about the transaction. Customer B makes a fraud transaction 
    the system approves it. Then the cost is chargeback for the bank and the order value.

    The logloss function punishes false negatives and false positives equally. The logloss with costs is a
    cost-sensitive loss function. Here, all four possible outcomes (False Positives, False Negatives, True 
    Positives and True Negatives) are being considered and each of the outcomes carries a specified cost. 
    """

    # column names for cost values or use numeric value when constant
    # costs of false negatives and false positives
    _fn_cost = 'age'
    _fp_cost = 1.0
    # costs of true negatives and true positives
    _tn_cost = 0.0
    _tp_cost = 0.0

    def make_cost_values(self, cost_value, X, shape, default_value):
        """
        Takes cost_value and checks for its type: if string then use as column name to extract cost from X,
        if numeric then use its actual value for cost. Falls back to default value if column name not found.
        :param cost_value: if string then column name in X, if int or float then actual cost
        :param X: dataset
        :param shape: size of cost array to return
        :param default_value: default cost value to fall back to in case column not found in X
        :return: cost array
        """
        if isinstance(cost_value, str):
            if isinstance(X, dt.Frame) and cost_value in X.names:
                cost = X[cost_value]
            else:
#                loggerwarning(logger, "Column " + cost_value + " not found - falling back to default cost " + default_value)
                cost = np.full(shape, default_value)
        elif isinstance(cost_value, float) or isinstance(cost_value, int):
            cost = np.full(shape, cost_value)
        else:
            raise ValueError("Cost must be a string for column name or numeric for default value.")

        return cost

    def score(self,
              X: dt.Frame,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:

        logger = None
#        if self._make_logger:
            # Example use of logger, with required import of:
            #  from h2oaicore.systemutils import make_experiment_logger, loggerinfo
            # Can use loggerwarning, loggererror, etc. for different levels
#            if self.context and self.context.experiment_id:
#                logger = make_experiment_logger(experiment_id=self.context.experiment_id, tmp_dir=self.context.tmp_dir,
#                                                experiment_tmp_dir=self.context.experiment_tmp_dir)

        N = actual.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(N)

        # label actual values as 1 or 0
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)

        # create datatable with all data
        DT = dt.Frame(actual = lb.transform(actual),
                      predicted = np.minimum(1 - self.__class__._epsilon, np.maximum(self.__class__._epsilon, predicted)),
                      cost_fn = self.make_cost_values(self.__class__._fn_cost, X, N, 1.),
                      cost_tp = self.make_cost_values(self.__class__._tp_cost, X, N, 0.),
                      cost_tn = self.make_cost_values(self.__class__._tn_cost, X, N, 0.),
                      cost_fp = self.make_cost_values(self.__class__._fp_cost, X, N, 1.),
                      sample_weight = sample_weight)
        lloss = DT[:, f.sample_weight * (f.actual * (f.cost_fn * dt.log(f.predicted) +
                                                     f.cost_tp * dt.log(1 - f.predicted)) +
                                         (1 - f.actual) * (f.cost_fp * dt.log(1 - f.predicted) +
                                                           f.cost_tn * dt.log(f.predicted)))]
        lloss = lloss.sum()[0,0] * -1.0 / np.sum(sample_weight)
        return lloss
