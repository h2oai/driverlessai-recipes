"""This custom transformer processes signal files to create features used by DriverlessAI to solve a regression problem"""

"""
This recipe has been created in the context of LANL Earthquake Prediction challenge on Kaggle
https://www.kaggle.com/c/LANL-Earthquake-Prediction

To use the recipe you have to transform the original data into the following form:
 - Signal data related to one label/target is stored in a separate file
 - The dataset submitted to DAI is of the form : ID, signalFilePath, Target

As an example, a row in the dataset would be : 1, "file_folder/signal_0001.csv", 3.05
which means that row ID 1 has a target value of 3.05
and the related signal can be found in file_folder/signal_0001.csv

The custom transformer uses the following libraries:
  - pywavelets
  - librosa,
  - numba
  - progressbar2

Please make sure to set the file_path feature as a text in DAI
To do so, click on the dataset in the dataset panel and chose DETAILS
Then in the detail panel, hover the file_path feature and choose text as the logical type

You may also want to disable the Text DAI Recipes.

"""
import importlib
from h2oaicore.transformer_utils import CustomTransformer
from h2oaicore.systemutils import small_job_pool, save_obj, load_obj, temporary_files_path, remove
import datatable as dt
import numpy as np
import pandas as pd

from scipy.stats import kurtosis, skew, linregress
from statsmodels.tsa.stattools import acf, adfuller, pacf
import math

# tsfresh python package requires pandas<=0.23.4, which is older than pandas used in DriverlessAI
# To avoid relying on tsfresh package and potential install issues
# It has been decided to just copy tsfresh methods used in this recipe
# Please see https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/

def autocorrelation(x, lag):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    if type(x) is pd.Series:
        x = x.values
    if len(x) < lag:
        return np.nan
        # Slice the relevant subseries based on the lag
    y1 = x[:(len(x) - lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(x)
    if np.isclose(v, 0):
        return np.NaN
    else:
        return sum_product / ((len(x) - lag) * v)


def absolute_sum_of_changes(x):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    return np.sum(np.abs(np.diff(x)))


def abs_energy(x):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


def agg_autocorrelation(x, param):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    # if the time series is longer than the following threshold, we use fft to calculate the acf
    THRESHOLD_TO_USE_FFT = 1250
    var = np.var(x)
    n = len(x)
    max_maxlag = max([config["maxlag"] for config in param])

    if np.abs(var) < 10 ** -10 or n == 1:
        a = [0] * len(x)
    else:
        a = acf(x, unbiased=True, fft=n > THRESHOLD_TO_USE_FFT, nlags=max_maxlag)[1:]
    return [("f_agg_\"{}\"__maxlag_{}".format(config["f_agg"], config["maxlag"]),
             getattr(np, config["f_agg"])(a[:int(config["maxlag"])])) for config in param]


def binned_entropy(x, max_bins):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    hist, bin_edges = np.histogram(x, bins=max_bins)
    probs = hist / x.size
    return - np.sum(p * np.math.log(p) for p in probs if p != 0)


def cid_ce(x, normalize=True):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s != 0:
            x = (x - np.mean(x)) / s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))


def linear_trend(x, param):
    """Credit goes to https://github.com/blue-yonder/tsfresh"""
    linReg = linregress(range(len(x)), x)

    return [("attr_\"{}\"".format(config["attr"]), getattr(linReg, config["attr"]))
            for config in param]


def mad(x, axis=None):
    return np.mean(np.abs(x - np.mean(x, axis)), axis)


def get_features(i_f, sig_file, mfcc_size):

    import librosa

    def get_nb_events_pd(sig, level):
        """Using numba would be faster"""
        a = pd.Series(sig).rolling(window=30).min().dropna().values
        b = pd.Series(sig).rolling(window=30).max().dropna().values
        z = np.log10(b - a + 1e-10)
        return np.sum(z[z>level])

    def wavelet_denoise(x, wavelet='db1', mode='hard'):
        pywt = importlib.import_module('pywt')

        # Extract approximate and detailed coefficients
        c_a, c_d = pywt.dwt(x, wavelet)

        # Determine the threshold
        sigma = 1 / 0.6745 * mad(np.abs(c_d))
        threshold = sigma * math.sqrt(2 * math.log(len(x)))

        # Filter the detail coefficients
        c_d_t = pywt.threshold(c_d, threshold, mode=mode)

        # Reconstruct the signal
        y_d = pywt.idwt(np.zeros_like(c_a), c_d_t, wavelet)

        # Determine the threshold
        sigma_a = 1 / 0.6745 * mad(np.abs(c_a))
        threshold_a = sigma_a * math.sqrt(2 * math.log(len(x)))

        # Filter the detail coefficients
        c_a_t = pywt.threshold(c_a, threshold_a, mode=mode)

        y_a = pywt.idwt(np.zeros_like(c_a), c_a_t, wavelet)

        return y_d, y_a, threshold, threshold_a

    # Read the file
    sig = dt.fread(sig_file).to_numpy()[:, 0]

    # Wavelet info
    denoised_d, denoised_a, threshold_d, threshold_a = wavelet_denoise(sig.astype(np.float64))
    the_mean = np.mean(sig)
    sig = sig - the_mean

    diff = np.diff(sig)
    eps = 1e-10

    sample = {
        # simple stats
        'sig_mean': the_mean,
        'sig_std': sig.std(),
        'sig_kurtosis': kurtosis(sig),
        'sig_skew': skew(sig),
        'sig_amp': np.max(sig) - np.min(sig),
        "sig_med_dist_to_med": np.median(np.abs((sig - np.median(sig)))),
        # Energy features
        'sig_l1_energy': np.abs(sig).mean(),
        'sig_l2_energy': np.abs((sig) ** 2).mean() ** .5,
        "ratio_diff": (diff[diff >= 0].sum() + eps) / (diff[diff < 0].sum() + eps),
        "mean_change": np.mean(diff),

        # Wavelet features
        "denoise_threshold_d": threshold_d,
        "desnoise_abs_sum_d": np.sum(np.abs(denoised_d)),
        "denoise_nb_peaks_d": (denoised_d != 0).astype(int).sum(),
        "denoise_threshold_a": threshold_a,
        "desnoise_abs_sum_a": np.sum(np.abs(denoised_a)),
        "denoise_nb_peaks_a": (denoised_a != 0).astype(int).sum(),
        "amp_max_a": np.max(abs(denoised_a)),
        "amp_max_d": np.max(abs(denoised_d)),

        # TSFresh features
        "trend_stderr": linear_trend(x=sig, param=[{"attr": "stderr"}])[0][1],
        "abs_change": absolute_sum_of_changes(x=sig),
        "abs_energy": abs_energy(x=sig - np.mean(sig)),
        "agg_autocorr_mean": agg_autocorrelation(x=sig, param=[{"f_agg": "mean", "maxlag": 10}])[0][1],
        "agg_autocorr_std": agg_autocorrelation(x=sig, param=[{"f_agg": "std", "maxlag": 10}])[0][1],
        "agg_autocorr_abs_mean": agg_autocorrelation(x=np.abs(sig), param=[{"f_agg": "mean", "maxlag": 10}])[0][1],
        "agg_autocorr_abs_std": agg_autocorrelation(x=np.abs(sig), param=[{"f_agg": "std", "maxlag": 10}])[0][1],
        "binned_entropy": binned_entropy(x=sig, max_bins=250),
        "cid_ce_normed": cid_ce(x=sig, normalize=True),

    }

    # More complex features
    for lag in [1, 2, 3, 5, 10]:
        sample[f"autocorr{lag}"] = autocorrelation(sig, lag)
        sample[f"autocorr_abs_{lag}"] = autocorrelation(np.abs(sig), lag)

    mfcc = librosa.feature.mfcc(sig.astype(np.float64) - the_mean, n_mfcc=mfcc_size).mean(axis=1)
    for i_mf, val in enumerate(mfcc):
        sample['mfcc_%d' % i_mf] = val

    return sample,


class MySignalProcessingTransformer(CustomTransformer):
    """
    SignalProcessing Transformer expects 2 features:
     - The first feature is a file name or path that contains the signal
     - The second feature is the target associated to the signal

    The transformer has no fit method and only transforms the data, at least for now
    """
    _modules_needed_by_name = ["pywavelets", "librosa", "numba", "progressbar2", "tsfresh"]

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols=1, max_cols=1, relative_importance=1)

    @property
    def display_name(self):
        return "SignalProcessingTransformer"

    @staticmethod
    def get_parameter_choices():
        return dict(mfcc_size=[10, 20, 40])

    def __init__(self, mfcc_size=10, **kwargs):
        super().__init__(**kwargs)
        self._mfcc_size = mfcc_size

    def fit(self, X: dt.Frame, y: np.array = None):
        # The transformer does not require to be fitted
        # it only processes signal files
        pass

    def transform(self, X: dt.Frame):
        """
        Transform expects only one column that contains a file name
        :param X: contains file names
        :return: features created on signals contained in the files
        """

        from progressbar import progressbar

        # First we want to make sure that:
        #   - X contains only 1 column
        #   - The column is text
        #   - The column contains file names

        # Make sure we have 1 column
        if X.shape[1] > 1:
            return np.zeros(X.shape[0])

        # Extract file paths
        if isinstance(X, dt.Frame):
            # Datatable can select features directly on type
            if X[:, [str]].shape[1] == 0:
                return np.zeros
            files = X[:, [str]].to_numpy()[:, 0]
        else:
            if X[X.columns[0]].dtype != "object":
                return np.zeros(X.shape[0])
            files = X[X.columns[0]].values[:, 0]

        # Now go through the files and create features
        try:
            def processor(out, res):
                # print(out)
                # print(res, flush=True)
                out.append(res[0])
                # out[res[0]] = res[1]

            num_tasks = X.shape[0]
            pool_to_use = small_job_pool
            pool = pool_to_use(logger=None, processor=processor, num_tasks=num_tasks)
            # Features will be a list of dict
            features = []
            for i_f, file in enumerate(progressbar(files)):
                # Create full path
                full_path = file

                # Send to pool
                args = (i_f, full_path, self._mfcc_size)
                kwargs = {}
                pool.submit_tryget(
                    None, get_features,
                    args=args, kwargs=kwargs,
                    out=features
                )

            pool.finish()

            features_df = pd.DataFrame(features)

            self._output_feature_names = list(features_df.columns)
            self._feature_desc = list(features_df.columns)

            return features_df

        except ValueError as e:
            err_msg = e.args[0]
            if "file" in err_msg.lower() and "does not exist" in err_msg.lower():
                print("Error in {} : {}".format(self.display_name, err_msg))
                return np.zeros(X.shape[0])


    def fit_transform(self, X: dt.Frame, y: np.array = None):
        # no fitting for now
        return self.transform(X)


class MyNumbaSignalProcessingTransformer(CustomTransformer):
    """
    SignalProcessing Transformer expects 2 features:
     - The first feature is a file name or path that contains the signal
     - The second feature is the target associated to the signal

    The transformer has no fit method and only transforms the data, at least for now
    """
    _modules_needed_by_name = ["numba", "progressbar2"]

    @staticmethod
    def is_enabled():
        return True

    @staticmethod
    def do_acceptance_test():
        return False

    @staticmethod
    def get_default_properties():
        return dict(col_type="all", min_cols=1, max_cols=1, relative_importance=1)

    @property
    def display_name(self):
        return "SignalProcNbAmpEvents"

    def __init__(self, mfcc_size=10, **kwargs):
        super().__init__(**kwargs)
        self._mfcc_size = mfcc_size

    def fit(self, X: dt.Frame, y: np.array = None):
        # The transformer does not require to be fitted
        # it only processes signal files
        pass

    def transform(self, X: dt.Frame):
        """
        Transform expects only one column that contains a file name
        :param X: contains file names
        :return: features created on signals contained in the files
        """

        # First we want to make sure that:
        #   - X contains only 1 column
        #   - The column is text
        #   - The column contains file names

        from progressbar import progressbar
        import numba

        @numba.jit(parallel=True, fastmath=True)
        def get_rolling_min(seq):
            window = 30
            l = len(seq) - window
            z = np.empty(l)
            for i in numba.prange(l):
                z[i] = np.min(seq[i:i + window])

            return z

        @numba.jit(parallel=True, fastmath=True)
        def get_rolling_max(seq):
            window = 30
            l = len(seq) - window
            z = np.empty(l)
            for i in numba.prange(l):
                z[i] = np.max(seq[i:i + window])

            return z

        def get_nb_events(file, levels):
            sig = dt.fread(file).to_numpy()[:, 0]
            a = get_rolling_min(sig)
            b = get_rolling_max(sig)
            z = np.log10(b - a + 1e-10)
            return [np.sum(z[z > _level]) for _level in levels]

        if X.shape[1] > 1:
            return np.zeros(X.shape[0])

        if isinstance(X, dt.Frame):
            # Datatable can select features directly on type
            if X[:, [str]].shape[1] == 0:
                return np.zeros(X.shape[0])
            files = X[:, [str]].to_numpy()[:,0]
        else:
            if X[X.columns[0]].dtype != "object":
                return np.zeros(X.shape[0])
            files = X[X.columns[0]].values[:, 0]

        # Now let's go through the files and create features
        try:

            # Here we are supposed to use numba so multi processing is not required
            levels = np.arange(1.0, 1.2, 0.1)
            ret_df = pd.DataFrame(
                [
                    get_nb_events(file, levels)
                    # for file in files
                    for file in progressbar(files)
                ]
            )

        except ValueError as e:
            err_msg = e.args[0]
            if "file" in err_msg.lower() and "does not exist" in err_msg.lower():
                print("Error in {} : {}".format(self.display_name, err_msg))
            return np.zeros(X.shape[0])

        # Use pandas instead of dt.Frame(features)
        # Pending issue #9894
        return ret_df.values

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        # no fitting for now
        return self.transform(X)
