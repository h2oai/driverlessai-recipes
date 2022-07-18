import copy
from typing import List

from h2oaicore.systemutils import IgnoreEntirelyError, update_precision

"""
Author: KarthikG

Based on
https://github.com/yzhao062/pytod/blob/main/examples/lof_example.py
"""
import datatable as dt
import numpy as np
from h2oaicore.models import CustomUnsupervisedModel
from h2oaicore.transformer_utils import CustomUnsupervisedTransformer


class pyTodLocalOutlierFactorTransformer(CustomUnsupervisedTransformer):
    _can_use_gpu = True
    _must_use_gpu = True
    _can_use_multi_gpu = False
    _get_gpu_lock = True
    _get_gpu_lock_vis = True
    _parallel_task = False
    _testing_can_skip_failure = True  # not stable algo, GPU OOM too often

    _modules_needed_by_name = ['pytod==0.0.3']
    
    def __init__(self,
                 num_cols: List[str] = list(),
                 output_features_to_drop=list(),
                 n_neighbors=20,
                 batch_size=10000,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        init_args_dict = locals().copy()
        self.params = {k: v for k, v in init_args_dict.items() if k in self.get_parameter_choices()}
        self._output_features_to_drop = output_features_to_drop

    @staticmethod
    def get_parameter_choices():
        """
        Possible parameters to use as mutations, where first value is default value
        """

        return dict(
                    n_neighbors=[10,20,5],  # could add to list other values
                    batch_size=[10000,20000,30000],
                    )

    @staticmethod
    def get_default_properties():
        return dict(col_type="numeric", min_cols=1, max_cols="all")

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        import torch
        import pytod
        
        from pytod.models.lof import LOF
        from pytod.utils.utility import validate_device
        
        if X.nrows <= 2:
            raise IgnoreEntirelyError
        params = copy.deepcopy(self.params)
        
        print("pyTodLocalOutlierFactorTransformer params: %s" % params)
        
        device = validate_device(0)
        clf_name = 'lof-PyTOD'
        params.update(dict(device=device))

        self.model = LOF(**params)
        
        # make float, replace of nan/inf won't work on int
        X = update_precision(X, fixup_almost_numeric=False)
        X.replace([None, np.nan, np.inf, -np.inf], 0.0)
        X = X.to_numpy()
        X= torch.from_numpy(X) ## Had to add this as it doesnt work with numpy arrays
        
        return self.model.fit_predict(X) # For labels
    
    def transform(self, X: dt.Frame, y: np.array = None):
        # no state, always finds outliers in any given dataset
        return self.fit_transform(X)


class pyTodLocalOutlierFactorModel(CustomUnsupervisedModel):
    _included_pretransformers = ['OrigFreqPreTransformer']  # frequency-encode categoricals, keep numerics as is
    _included_transformers = ["pyTodLocalOutlierFactorTransformer"]
    _included_scorers = ['UnsupervisedScorer']  # trivial, nothing to score
