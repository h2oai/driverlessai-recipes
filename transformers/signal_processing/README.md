# Signal Processing

## SignalProcessingTransformer

#### ➡️ Description
This custom transformer processes signal files to create features used by DriverlessAI to solve a regression problem
This recipe has been created in the context of _LANL Earthquake Prediction Challenge_ on Kaggle
https://www.kaggle.com/c/LANL-Earthquake-Prediction

To use the recipe you have to transform the original data into the following form:
 - Signal data related to one label/target is stored in a separate file
 - The dataset submitted to DAI is of the form : ID, signalFilePath, Target
 
Please make sure to set the `file_path` feature as a text in DAI
To do so, click on the dataset in the dataset panel and chose DETAILS
Then in the detail panel, hover the file_path feature and choose text as the logical type
You may also want to disable the Text DAI Recipes.

#### ➡️ Code
- [signal_processing.py](signal_processing.py)

#### ➡️ Inputs
- `signalFilePath`: file location storing signal information

#### ➡️ Outputs
The custom recipe outputs following features:
 - Statistics: mean, median, min/max, standard deviation, skewness, kurtosis
 - Mel Frequency Cepstral Coefficient (MFCC)
 - Autocorrelation at different lags
 - Trend information
 - Number of events having an amplitude greater than a threshold

#### ➡️ Environment expectation
Python 3.6, DAI 1.7.0 and above

#### ➡️ Dependencies
- [librosa](https://librosa.github.io/librosa/) used for music and audio analysis
- [tsfresh](https://tsfresh.readthedocs.io/en/latest/) amazing time series and signal processing package.
- [pywavelets](https://pywavelets.readthedocs.io/en/latest/) for wavelet transforms. 
- [numba](https://numba.pydata.org/) used to accelerate heavy computations
- [progressbar2]() to display progress as signals are processed
