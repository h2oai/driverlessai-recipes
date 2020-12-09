"""Area under uplift curve"""

import typing
import numpy as np
import pandas as pd
import datatable as dt
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from h2oaicore.systemutils import config
from distutils.util import strtobool

class AUUC(CustomScorer):
    _description = "Area under uplift curve"
    _maximize = True  # whether a higher score is better
    _perfect_score = 2.0  # AUUC can be slightly > 1.

    _supports_sample_weight = False  # whether the scorer accepts and uses the sample_weight input

    _regression = True
    _binary = False
    _multiclass = False

    _RANDOM_COL = 'Random'

    #
    # The following functions (except the very scorer()) are copied from https://github.com/uber/causalml/blob/master/causalml/metrics/visualize.py
    #
    def get_cumgain(self, df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                    normalize=False, random_seed=42):
        """Get cumulative gains of model estimates in population.

        If the true treatment effect is provided (e.g. in synthetic data), it's calculated
        as the cumulative gain of the true treatment effect in each population.
        Otherwise, it's calculated as the cumulative difference between the mean outcomes
        of the treatment and control groups in each population.

        For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
        and Uplift Modeling: A review of the literature`.

        For the former, `treatment_effect_col` should be provided. For the latter, both
        `outcome_col` and `treatment_col` should be provided.

        Args:
            df (pandas.DataFrame): a data frame with model estimates and actual data as columns
            outcome_col (str, optional): the column name for the actual outcome
            treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
            treatment_effect_col (str, optional): the column name for the true treatment effect
            normalize (bool, optional): whether to normalize the y-axis to 1 or not
            random_seed (int, optional): random seed for numpy.random.rand()

        Returns:
            (pandas.DataFrame): cumulative gains of model estimates in population
        """

        lift = self.get_cumlift(df, outcome_col, treatment_col, treatment_effect_col, random_seed)

        # cumulative gain = cumulative lift x (# of population)
        gain = lift.mul(lift.index.values, axis=0)

        if normalize:
            gain = gain.div(np.abs(gain.iloc[-1, :]), axis=1)

        return gain

    def get_cumlift(self, df, outcome_col='y', treatment_col='w', treatment_effect_col='tau',
                    random_seed=42):
        """Get average uplifts of model estimates in cumulative population.

        If the true treatment effect is provided (e.g. in synthetic data), it's calculated
        as the mean of the true treatment effect in each of cumulative population.
        Otherwise, it's calculated as the difference between the mean outcomes of the
        treatment and control groups in each of cumulative population.

        For details, see Section 4.1 of Gutierrez and G{\'e}rardy (2016), `Causal Inference
        and Uplift Modeling: A review of the literature`.

        For the former, `treatment_effect_col` should be provided. For the latter, both
        `outcome_col` and `treatment_col` should be provided.

        Args:
            df (pandas.DataFrame): a data frame with model estimates and actual data as columns
            outcome_col (str, optional): the column name for the actual outcome
            treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
            treatment_effect_col (str, optional): the column name for the true treatment effect
            random_seed (int, optional): random seed for numpy.random.rand()

        Returns:
            (pandas.DataFrame): average uplifts of model estimates in cumulative population
        """

        assert ((outcome_col in df.columns) and (treatment_col in df.columns) or
                treatment_effect_col in df.columns)

        df = df.copy()
        np.random.seed(random_seed)
        random_cols = []
        for i in range(10):
            random_col = '__random_{}__'.format(i)
            df[random_col] = np.random.rand(df.shape[0])
            random_cols.append(random_col)

        model_names = [x for x in df.columns if x not in [outcome_col, treatment_col,
                                                          treatment_effect_col]]

        lift = []
        for i, col in enumerate(model_names):
            df_sorted = df.sort_values(col, ascending=False).reset_index(drop=True)
            df_sorted.index = df_sorted.index + 1

            if treatment_effect_col in df_sorted.columns:
                # When treatment_effect_col is given, use it to calculate the average treatment effects
                # of cumulative population.
                lift.append(df_sorted[treatment_effect_col].cumsum() / df_sorted.index)
            else:
                # When treatment_effect_col is not given, use outcome_col and treatment_col
                # to calculate the average treatment_effects of cumulative population.
                df_sorted['cumsum_tr'] = df_sorted[treatment_col].cumsum()
                df_sorted['cumsum_ct'] = df_sorted.index.values - df_sorted['cumsum_tr']
                df_sorted['cumsum_y_tr'] = (df_sorted[outcome_col] * df_sorted[treatment_col]).cumsum()
                df_sorted['cumsum_y_ct'] = (df_sorted[outcome_col] * (1 - df_sorted[treatment_col])).cumsum()

                lift.append(df_sorted['cumsum_y_tr'] / df_sorted['cumsum_tr'] - df_sorted['cumsum_y_ct'] / df_sorted['cumsum_ct'])

        lift = pd.concat(lift, join='inner', axis=1)
        lift.loc[0] = np.zeros((lift.shape[1],))
        lift = lift.sort_index().interpolate()

        lift.columns = model_names
        lift[self._RANDOM_COL] = lift[random_cols].mean(axis=1)
        lift.drop(random_cols, axis=1, inplace=True)

        return lift

    def auuc_score(self, df, outcome_col='y', treatment_col='w', treatment_effect_col='tau', normalize=True, *args, **kwarg):
        """Calculate the AUUC (Area Under the Uplift Curve) score.

         Args:
            df (pandas.DataFrame): a data frame with model estimates and actual data as columns
            outcome_col (str, optional): the column name for the actual outcome
            treatment_col (str, optional): the column name for the treatment indicator (0 or 1)
            treatment_effect_col (str, optional): the column name for the true treatment effect
            normalize (bool, optional): whether to normalize the y-axis to 1 or not

        Returns:
            (float): the AUUC score
        """

        cumgain = self.get_cumgain(df, outcome_col, treatment_col, treatment_effect_col, normalize)
        return cumgain.sum() / cumgain.shape[0]

    @staticmethod
    def do_acceptance_test():
        """
        Whether to enable acceptance tests during upload of recipe and during start of Driverless AI.

        Acceptance tests perform a number of sanity checks on small data, and attempt to provide helpful instructions
        for how to fix any potential issues. Disable if your recipe requires specific data or won't work on random data.
        """
        return False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None,
              X: typing.Optional[dt.Frame] = None,
              **kwargs) -> float:

        df = pd.DataFrame({
            'dai': predicted,
            'outcome': actual,
            'treatment': sample_weight
        })

        return self.auuc_score(df, outcome_col='outcome', treatment_col='treatment', treatment_effect_col=None, normalize=True)['dai']
