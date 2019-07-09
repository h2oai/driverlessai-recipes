"""Decision Forest Model based on Intel DAAL"""

#
# The TAR SH download is recommended for running this recipe, since you can install and run it as your own
# regular userid and easily install the required supplemental DAAL files.
#
# The following three downloads are required to run this recipe.
#
# As of this writing, DAAL is only distributed as conda packages, so we need to download and unpack
# these conda packages and move the important files into the proper place in the DAI python distribution.
# In the future, Intel says they will be able to distribute these packages as regular python wheel files,
# as well, simplifying this process.
#
# The python module needs to be placed in the site-packages directory
# (dai-n.n.n-linux-x86_64/python/lib/python3.6/site-packages/daal4py) and the .so files need to be put
# in LD_LIBRARY_PATH (the dai-n.n.n-linux-x86_64/lib directory works).
#
# You can test if this is working by hand as follows:
# $ ./dai-env.sh python
# >>> import daal4py
#
# In the successful case, the import will succeed with no output.
#
# -----
#
# $ wget https://anaconda.org/intel/daal4py/2019.4/download/linux-64/daal4py-2019.4-py36h7b7c402_6.tar.bz2
# $ wget https://anaconda.org/intel/impi_rt/2019.4/download/linux-64/impi_rt-2019.4-intel_243.tar.bz2
# $ wget https://anaconda.org/intel/daal/2019.4/download/linux-64/daal-2019.4-intel_243.tar.bz2
#
# -rw-rw-r-- 1 tomk tomk  10933960 Mar  6 13:36 daal4py-2019.3-py36h7b7c402_6.tar.bz2
# -rw-rw-r-- 1 tomk tomk  61624320 May 20 11:01 impi_rt-2019.4-intel_243.tar
# -rw-rw-r-- 1 tomk tomk 388853760 May 20 14:46 daal-2019.4-intel_243.tar
#


import datatable as dt
from h2oaicore.models import CustomModel
import daal4py as d4p
import numpy as np
from sklearn.preprocessing import LabelEncoder


class DaalForestModel(CustomModel):
    _regression = False
    _binary = True
    _multiclass = False
    _display_name = "DaalForest"
    _description = "Decision Forest Model based on Intel DAAL (https://intelpython.github.io/daal4py/algorithms.html)"

    def set_default_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params = dict(nClasses=self.num_classes,
                           fptype='float',
                           varImportance='MDI',
                           nTrees=10)

    def mutate_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        pass

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        lb = LabelEncoder()
        lb.fit(self.labels)
        y = lb.transform(y)

        if isinstance(X, dt.Frame):
            X_features = list(X.names)
            X = X.to_numpy()
            dtype = np.float32 if self.params['fptype'] == 'float' else np.float64
            X = np.ascontiguousarray(X, dtype=dtype)
            y = np.ascontiguousarray(y, dtype=dtype).reshape(X.shape[0], 1)
        else:
            raise

        train_algo = d4p.decision_forest_classification_training(**self.params)
        train_result = train_algo.compute(X, y)
        model_tuple = (train_result, self.num_classes, self.params['fptype'])
        importances = train_result.variableImportance.tolist()[0]
        self.set_model_properties(model=model_tuple,
                                  features=X_features,
                                  importances=importances,
                                  iterations=self.params['nTrees'])

    def predict(self, X, **kwargs):
        model_tuple, _, _, _ = self.get_model_properties()
        train_result = model_tuple[0]
        nClasses = model_tuple[1]
        fptype = model_tuple[2]
        predict_algo = d4p.decision_forest_classification_prediction(nClasses=nClasses, fptype=fptype)
        X = X.to_numpy()

        # This is not optimal at the moment because it returns the 0/1 label and not a probability.
        # So the ROC curve in DAI looks very jagged.  A future version of DAAL Decision Forest will
        # support predicting probabilities as well as the label.
        result = predict_algo.compute(X, train_result.model).prediction.ravel()

        return result
