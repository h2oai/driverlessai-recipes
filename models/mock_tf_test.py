"""For GPU usage testing purposes."""
import os

import numpy as np
from h2oaicore.models import CustomModel
from h2oaicore.models_utils import import_tensorflow
from h2oaicore.systemutils import config, ngpus_vis


class CustomTFGPUCheck(CustomModel):
    _regression = True
    _binary = True
    _multiclass = False  # WIP
    _is_reproducible = False

    _can_use_gpu = True  # if enabled, will use special job scheduler for GPUs
    _get_gpu_lock = True  # whether to lock GPUs for this model before fit and predict
    _get_gpu_lock_vis = True  # since always using gpu 0
    _must_use_gpu = True  # this recipe can only be used if have GPUs
    _predict_on_same_gpus_as_fit = True  # force predict to behave like fit, regardless of config.num_gpus_for_prediction

    def set_default_params(self,
                           accuracy=None, time_tolerance=None, interpretability=None,
                           **kwargs):
        self.params = {}

    def mutate_params(self,
                      **kwargs):
        self.params = {}

    @staticmethod
    def is_enabled():
        try:
            return not config.global_disable_tensorflow
        except:
            return True
    
    @staticmethod
    def enabled_setting():
        try:
            return "auto" if config.global_disable_tensorflow else "off"
        except:
            return "on"
    
    @staticmethod
    def can_use(accuracy, interpretability, train_shape=None, test_shape=None, valid_shape=None, n_gpus=0,
                num_classes=None, **kwargs):
        try:
            return not config.global_disable_tensorflow
        except:
            return True

    @staticmethod
    def acceptance_test_coverage_fraction():
        return 0.05

    def fit(self, X, y, sample_weight=None, eval_set=None, sample_weight_eval_set=None, **kwargs):
        '''
        Basic Multi GPU computation example using TensorFlow library.
        Author: Aymeric Damien
        Project: https://github.com/aymericdamien/TensorFlow-Examples/
        '''

        assert ngpus_vis != 0, "Shouldn't be using/testing this recipe without GPUs"
        '''
        This tutorial requires your machine to have 1 GPU
        "/cpu:0": The CPU of your machine.
        "/gpu:0": The first GPU of your machine
        '''
        tf = import_tensorflow()
        import numpy as np
        import datetime

        # Processing Units logs
        log_device_placement = True

        # Num of multiplications to perform
        n = 3

        '''
        Example: compute A^n + B^n on 2 GPUs
        Results on 8 cores with 2 GTX-980:
         * Single GPU computation time: 0:00:11.277449
         * Multi GPU computation time: 0:00:07.131701
        '''
        # Create random large matrix
        A = np.random.rand(10000, 10000).astype('float32')
        B = np.random.rand(10000, 10000).astype('float32')

        # Create a graph to store results
        c1 = []
        c2 = []

        def matpow(M, n):
            if n < 1:  # Abstract cases where n < 1
                return M
            else:
                return tf.matmul(M, matpow(M, n - 1))

        '''
        Single GPU computing
        '''
        with tf.device('/gpu:0'):
            a = tf.placeholder(tf.float32, [10000, 10000])
            b = tf.placeholder(tf.float32, [10000, 10000])
            # Compute A^n and B^n and store results in c1
            c1.append(matpow(a, n))
            c1.append(matpow(b, n))

        with tf.device('/gpu:0'):
            sum = tf.add_n(c1)  # Addition of all elements in c1, i.e. A^n + B^n

        t1_1 = datetime.datetime.now()
        with tf.Session(
                config=tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=True)) as sess:
            # Run the op.
            sess.run(sum, {a: A, b: B})
        t2_1 = datetime.datetime.now()

        print("Single GPU computation time: " + str(t2_1 - t1_1))

        self.set_model_properties(model=[1],
                                  features=list(X.names),
                                  importances=([1.0] * len(list(X.names))),
                                  iterations=0)

    def predict(self, X, **kwargs):
        """
        Returns: dt.Frame, np.ndarray or pd.DataFrame, containing predictions (target values or class probabilities)
        Shape: (K, c) where c = 1 for regression or binary classification, and c>=3 for multi-class problems.
        """
        return np.random.randint(0, 2, (X.nrows, 1))
