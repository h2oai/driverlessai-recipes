"""Qudratic Weighted Kappa"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder


class QuadraticWeightedKappaScorer(CustomScorer):
    _description = "Quadratic Weighted Kappa - A measure of inter-rater agreement between two raters that provide discrete numeric ratings. Potential values range from -1 (representing complete disagreement) to 1 (representing complete agreement). A kappa value of 0 is expected if all agreement is due tochance."
    _multiclass = True
    _maximize = True
    _perfect_score = 0
    _display_name = "QWK"

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              **kwargs) -> float:

        # special parameters of QWK
        """
        The ratings should be integers, and it is assumed that they contain
        the complete range of possible ratings.
        quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
        is the minimum possible rating, and max_rating is the maximum possible
        rating
        """

        _min_rating = None
        _max_rating = None

        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
        predicted = np.argmax(predicted, axis=1)
        if _min_rating is None:
            _min_rating = int(np.min(labels))
        if _max_rating is None:
            _max_rating = int(np.max(labels))

        return qwk(actual, predicted, min_rating=_min_rating, max_rating=_max_rating, sample_weight=sample_weight)


def histogram(ratings, sample_weight, min_rating=None, max_rating=None):
    """
    Returns the (weighted) counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r, w in zip(ratings, sample_weight):
        hist_ratings[r - min_rating] += w
    return hist_ratings


def my_confusion_matrix(rater_a, rater_b, sample_weight, min_rating=None, max_rating=None):
    """
    Returns the (weighted) confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b, w in zip(rater_a, rater_b, sample_weight):
        conf_mat[a - min_rating][b - min_rating] += w
    return conf_mat


def qwk(actual, predicted, min_rating=0, max_rating=20, sample_weight=None):
    if sample_weight is None:
        sample_weight = np.ones(actual.shape[0])
    rater_a = np.array(actual, dtype=int)
    rater_b = np.array(predicted, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = my_confusion_matrix(rater_a, rater_b, sample_weight,
                                   min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(np.sum(sample_weight))

    hist_rater_a = histogram(rater_a, sample_weight, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, sample_weight, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator
