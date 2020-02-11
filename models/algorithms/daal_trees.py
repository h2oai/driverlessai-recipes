"""Binary Classification and Regression for Decision Forest and Gradient Boosting based on Intel DAAL"""

import datatable as dt
from h2oaicore.models import CustomModel
import numpy as np
from sklearn.preprocessing import LabelEncoder
import daal4py as d4p


class DaalBaseModel(object):
    _regression = True
    _binary = True  # FIXME: but returns class, not probabilities
    _multiclass = False  # FIXME: shape issue
    _can_use_gpu = False
    _is_reproducible = False

    @staticmethod
    def is_enabled():
        from h2oaicore.systemutils import arch_type
        return not (arch_type == "ppc64le")

    def dt_to_numpy(self, X, y=None):
        if isinstance(X, dt.Frame):
            X = X.to_numpy()
            dtype = np.float32 if self.params['fptype'] == 'float' else np.float64
            X = np.ascontiguousarray(X, dtype=dtype)
            if y is not None:
                y = np.ascontiguousarray(y, dtype=dtype).reshape(X.shape[0], 1)
        else:
            raise
        return X, y

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        if self.num_classes > 1:
            lb = LabelEncoder()
            lb.fit(self.labels)
            y = lb.transform(y)

        X_features = list(X.names)
        X, y = self.dt_to_numpy(X, y)
        if self.num_classes == 1:
            train_func = self._train_func_regress
        else:
            train_func = self._train_func_class
        train_algo = train_func(**self.params)
        train_result = train_algo.compute(X, y)
        model_tuple = (train_result, self.num_classes, self.params['fptype'])
        if hasattr(train_result, 'variableImportance'):
            importances = train_result.variableImportance.tolist()[0]
        else:
            importances = np.ones(len(X_features))
        self.set_model_properties(model=model_tuple,
                                  features=X_features,
                                  importances=importances,
                                  iterations=self.params.get('nTrees', self.params.get('maxIterations', 100)))

    def predict(self, X, **kwargs):
        model_tuple, _, _, _ = self.get_model_properties()
        train_result = model_tuple[0]
        nClasses = model_tuple[1]
        fptype = model_tuple[2]
        if self.num_classes == 1:
            predict_func = self._predict_func_regress
            other_kwargs = {}
        else:
            predict_func = self._predict_func_class
            other_kwargs = {'nClasses': nClasses}
        predict_algo = predict_func(fptype=fptype, **other_kwargs)
        X, _ = self.dt_to_numpy(X, None)

        # This is not optimal at the moment because it returns the 0/1 label and not a probability.
        # So the ROC curve in DAI looks very jagged.  A future version of DAAL Decision Forest will
        # support predicting probabilities as well as the label.
        if self.num_classes <= 2:
            result = predict_algo.compute(X, train_result.model).prediction.ravel()
        else:
            result = predict_algo.compute(X, train_result.model).prediction

        return result


class DaalTreeModel(DaalBaseModel, CustomModel):
    _display_name = "DaalTree"
    _description = "Decision Tree Model based on Intel DAAL (https://intelpython.github.io/daal4py/algorithms.html)"
    _train_func_class = d4p.gbt_classification_training
    _predict_func_class = d4p.gbt_classification_prediction
    _train_func_regress = d4p.gbt_regression_training
    _predict_func_regress = d4p.gbt_regression_prediction

    def set_default_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params = {
            'nClasses': self.num_classes,
            'fptype': 'float',
            'maxIterations': 200,
            'maxTreeDepth': 6,
            'minSplitLoss': 0.1,
            'shrinkage': 0.3,
            'observationsPerTreeFraction': 1,
            'lambda_': 1,
            'maxBins': 256,
            'featuresPerNode': 0,
            'minBinSize': 5,
            'memorySavingMode': False,
            'minObservationsInLeafNode': 1
        }
        if self.num_classes == 1:
            self.params.pop('nClasses', None)
            self.params.pop('nTrees', None)
            self.params.pop('maxIterations', None)


class DaalForestModel(DaalBaseModel, CustomModel):
    _display_name = "DaalForest"
    _description = "Decision Forest Model based on Intel DAAL (https://intelpython.github.io/daal4py/algorithms.html)"
    _train_func_class = d4p.decision_forest_classification_training
    _predict_func_class = d4p.decision_forest_classification_prediction
    _train_func_regress = d4p.decision_forest_regression_training
    _predict_func_regress = d4p.decision_forest_regression_prediction

    def set_default_params(self, accuracy=None, time_tolerance=None, interpretability=None, **kwargs):
        self.params = dict(nClasses=self.num_classes,
                           fptype='float',
                           varImportance='MDI',
                           nTrees=100)
        if self.num_classes == 1:
            self.params.pop('nClasses', None)
            self.params.pop('nTrees', None)
            self.params.pop('maxIterations', None)


def _setup_recipe():
    # for DAI 1.7.0 one is required to run this function manually
    # in DAI >=1.7.1, this function will be run by DAI itself
    import os
    from h2oaicore.systemutils_more import extract, download
    from h2oaicore.systemutils import config
    import shutil

    from h2oaicore.systemutils import arch_type
    if arch_type == "ppc64le":
        raise RuntimeError("Cannot use daal on PPC")

    daal_is_installed_path = os.path.join(config.data_directory, config.contrib_env_relative_directory, "daal")
    daal_is_installed_file = os.path.join(daal_is_installed_path, "daal_is_installed")
    if not os.path.isfile(daal_is_installed_file):
        daal_temp_path = os.path.join(config.data_directory, config.contrib_relative_directory, "daal")
        os.makedirs(daal_temp_path, exist_ok=True)
        prefix = "https://anaconda.org/intel"
        try:
            file1 = download("%s/daal4py/2019.4/download/linux-64/daal4py-2019.4-py36h7b7c402_6.tar.bz2" % prefix,
                             dest_path=daal_temp_path)
            file2 = download("%s/impi_rt/2019.4/download/linux-64/impi_rt-2019.4-intel_243.tar.bz2" % prefix,
                             dest_path=daal_temp_path)
            file3 = download("%s/daal/2019.4/download/linux-64/daal-2019.4-intel_243.tar.bz2" % prefix,
                             dest_path=daal_temp_path)
            file4 = download("https://github.com/intel/daal/releases/download/2019_u4/l_daal_oss_p_2019.4.007.tgz",
                             dest_path=daal_temp_path)
        except:
            file1 = download("https://0xdata-public.s3.amazonaws.com/daal4py-2019.4-py36h7b7c402_6.tar.bz2", dest_path=daal_temp_path)
            file2 = download("https://0xdata-public.s3.amazonaws.com/impi_rt-2019.4-intel_243.tar.bz2", dest_path=daal_temp_path)
            file3 = download("https://0xdata-public.s3.amazonaws.com/daal-2019.4-intel_243.tar.bz2", dest_path=daal_temp_path)
            file4 = download("https://0xdata-public.s3.amazonaws.com/l_daal_oss_p_2019.4.007.tgz", dest_path=daal_temp_path)
        temp_path = os.path.join(config.data_directory, config.contrib_env_relative_directory, "info")
        os.makedirs(temp_path, exist_ok=True)
        python_site_packages_path = os.path.join(config.data_directory, config.contrib_env_relative_directory)
        extract(file1, python_site_packages_path)
        python_site_packages_path2 = os.path.join(config.data_directory, config.contrib_env_relative_directory)
        extract(file2, python_site_packages_path2)
        extract(file3, python_site_packages_path2)
        extract(file4, python_site_packages_path2, "gz")

        other_path = os.path.join(python_site_packages_path2, "lib/libfabric/")
        import glob
        for file in glob.glob(os.path.join(other_path, "*.so*")):
            new_file = os.path.join(python_site_packages_path2, "lib", os.path.basename(file))
            if not os.path.isfile(new_file):
                shutil.copy(file, new_file)

        other_path = os.path.join(python_site_packages_path2,
                                  "l_daal_oss_p_2019.4.007/daal_prebuild/linux/tbb/lib/intel64_lin/gcc4.4/")
        import glob
        for file in glob.glob(os.path.join(other_path, "*.so*")):
            new_file = os.path.join(python_site_packages_path2, "lib", os.path.basename(file))
            if not os.path.isfile(new_file):
                shutil.copy(file, new_file)
        os.makedirs(daal_is_installed_path, exist_ok=True)
        with open(daal_is_installed_file, "wt") as f:
            f.write("DONE")
        return True
