"""Probabilistic F Score with optimized threshold"""

import typing
import numpy as np
import pandas as pd
import datatable as dt
from h2oaicore.metrics import CustomScorer, prep_actual_predicted
from sklearn.preprocessing import label_binarize
from h2oaicore.systemutils import make_experiment_logger, loggerinfo, loggerwarning


class ProbF1Opt2(CustomScorer):
    _description = "Probabilistic F Score with optimized threshold"
    _binary = True
    _multiclass = True
    _maximize = True
    _perfect_score = 1.0
    _display_name = "ProbF1Opt2"
    _needs_X = True  # not required, but in order to group by some ID
    _group_id = 'patient_id'
    _group_also_laterality = True
    _opt_threshold = True

    def pfbeta(self, labels, predictions, beta=1.0, sample_weight=None, threshold=None, X=None):
        """
            This is implements the probablistic F score described here:

            https://aclanthology.org/2020.eval4nlp-1.9.pdf

            https://www.kaggle.com/code/sohier/probabilistic-f-score/comments

            Should disable bootstrapping via TOML enable_bootstrap=false

        :param labels:
        :param predictions:
        :param beta:
        :return:
        """

        if threshold is not None:
            predictions[predictions < threshold] = 0

        predictions[predictions < 0.0] = 0.0
        predictions[predictions > 1.0] = 1.0

        if self._group_id is not None and self._group_id in X.names:
            lat = 'laterality'
            if not self._group_also_laterality:
                # ensure some ID has same (and good) probs since
                # e.g. is some patient ID and same patient can't have different outcome for cancer
                df = pd.DataFrame(predictions, columns=['predictions'])
                df[self._group_id] = X[:, self._group_id].to_pandas()
                max_predictions = df.groupby(self._group_id).transform('max')['predictions']  #(lambda x: x.max())['predictions']
                mean_predictions = df.groupby(self._group_id).transform('mean')['predictions']  #(lambda x: x.mean())['predictions']
                predictions = 0.5 * (max_predictions + mean_predictions)
            elif lat in X.names:
                # group by prediction_id, the actual target ID
                df = pd.DataFrame(predictions, columns=['predictions'])
                df[self._group_id] = X[:, self._group_id].to_pandas()
                df[lat] = X[:, lat].to_pandas()

                prediction_ids = df['patient_id'].astype(str) + '_' + df['laterality'].astype(str)
                df_preds = pd.DataFrame({"proba": predictions, 'y': labels, 'prediction_id': prediction_ids.ravel()})
                df_preds = df_preds.groupby('prediction_id')[['proba', 'y']].max().reset_index()

                predictions = df_preds['proba'].values
                labels = df_preds['y'].values
                prediction_ids = df_preds.index.values  # unused
            else:
                pass

        sw = sample_weight if sample_weight is not None else np.ones(predictions.shape)

        y_true_count = np.sum(sw[labels > 0])
        ctp = np.sum(predictions[labels > 0] * sw[labels > 0])
        cfp = np.sum(predictions[labels == 0] * sw[labels == 0])

        beta_squared = beta * beta
        c_precision = ctp / (ctp + cfp)
        c_recall = ctp / y_true_count
        if c_precision > 0 and c_recall > 0:
            probf1_0 = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        else:
            probf1_0 = 0.0

        return probf1_0

    @property
    def logger(self):
        from h2oaicore import application_context
        from h2oaicore.systemutils import exp_dir
        # Don't assign to self, not picklable
        return make_experiment_logger(experiment_id=application_context.context.experiment_id, tmp_dir=None,
                                      experiment_tmp_dir=exp_dir())

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              X: typing.Optional[dt.Frame] = None,
              **kwargs) -> float:

        if sample_weight is not None:
            sample_weight = sample_weight.ravel()
        enc_actual, enc_predicted, labels = prep_actual_predicted(actual, predicted, labels)
        cm_weights = sample_weight if sample_weight is not None else None

        # multiclass
        if enc_predicted.shape[1] > 1:
            enc_predicted = enc_predicted.ravel()
            enc_actual = label_binarize(enc_actual, classes=labels).ravel()
            cm_weights = np.repeat(cm_weights, predicted.shape[1]).ravel() if cm_weights is not None else None
            assert enc_predicted.shape == enc_actual.shape
            assert cm_weights is None or enc_predicted.shape == cm_weights.shape

        act = enc_actual.ravel()
        pred = enc_predicted.ravel()
        sw = cm_weights.ravel() if cm_weights is not None else None
        probf1_0 = self.pfbeta(act, pred, sample_weight=sw, X=X)

        if not self._opt_threshold:
            probf1 = probf1_0
        else:

            max_probf1 = -1
            max_th0 = -1
            # FIXME: Make parallel
            for th0 in np.arange(0.01, 0.2, 0.01):
                probf1 = self.pfbeta(act, pred, sample_weight=sw, threshold=th0, X=X)
                if probf1 > max_probf1:
                    max_probf1 = probf1
                    max_th0 = th0

            probf1 = max_probf1
            loggerinfo(self.logger, "probf1: %s at threshold: %s (default probf1: %s)" % (probf1, max_th0, probf1_0))

        return probf1
