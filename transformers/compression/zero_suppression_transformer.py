"""
Performs Zero Suppression

The Zero Suppression algorithm is often used in Particle Physics to compress data produced by Particle Detectors.
The data is usually composed of 2-dimensional slices of the detector. Each cell holds an energy charge that was
measured in that particular part of the detector in a given time window. After a baseline analysis is applied to
eliminate the residual charge accumulated over time due to radiation damage. At this point only cells which were
actually hit by particles store non zero values. Because particle detectors produce immense amounts of data storing
and analyzing it in the raw (NZS-non zero suppressed) format is both inefficient and unnecessary. Only a very small
sample of randomly selected NZS data is kept for diagnostic purposes. The actual data used for further processing
is always stored in a compressed (ZS-zero suppressed) format.

     NZS DATA     ->        ZS DATA
     0   1   2              0   1   2
--  --  --  --          --  --  --  --
 0   0   0   3          0   0   2   3
 1   4   0   6          1   1   0   4
 2   7   0   0          2   1   2   6
                        3   2   0   7

More info can be found here: http://cds.cern.ch/record/689422/files/INT-1996-03.pdf
"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import pandas as pd
import numpy as np


class ZeroSuppressionTransformer(CustomTransformer):
    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols="all", relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):
        zs_indexes = X.to_numpy().nonzero()
        zs_indexes_t = pd.DataFrame(zs_indexes).T
        values = X.to_numpy()[zs_indexes]
        values_t = pd.DataFrame(values).T
        zs_data = pd.concat([zs_indexes_t, values_t], axis=1, ignore_index=True)
        return zs_data
