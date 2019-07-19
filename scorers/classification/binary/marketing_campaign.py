"""Computes the mean profit per outbound marketing letter, given a fraction of the population addressed, and fixed cost and reward"""

import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


class MarketingCampaign(CustomScorer):
    _description = "Calculates mean profit per letter sent for a marketing campaign"
    _binary = True
    _maximize = True
    _perfect_score = 1e20
    _display_name = "Campaign"
    _supports_sample_weight = False

    # Configure these for your problem
    _cost = 1.0      # cost to send letter
    _reward = 100.0  # reward if get response back
    _quantile = 0.9  # fraction of population to discard (here, only 10% of population gets a letter)

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:

        # label actuals as 1 or 0
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)

        # probability of predicted response likelihood above which we'll send a letter
        cutoff = np.quantile(predicted, self._quantile)
        #print("cutoff: %f" % cutoff)

        # whom we'll send letter to
        selected = (predicted >= cutoff).ravel()
        num_letters = len(np.where(selected))
        #print("number of letters: %d" % num_letters)

        # compute cost and reward
        cost = num_letters * self._cost   # each letter costs _cost
        reward = len(np.where(actual[selected] == 1)) * self._reward   # each true positive leads to _reward
        #print("cost: %f" % cost)
        #print("reward: %f" % reward)

        # compute total net income
        net_income = reward - cost
        #print("net_income: %f" % net_income)

        # return mean profit per letter sent
        return net_income / num_letters
