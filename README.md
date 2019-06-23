# Recipes for H2O Driverless AI

* [MODELS](./models)
  * [ALGORITHMS](./models/algorithms)
    * [catboost.py](./models/algorithms/catboost.py) [CatBoost gradient boosting by Yandex. Currently supports regression and binary classification.]
    * [h2o-3-models.py](./models/algorithms/h2o-3-models.py) [H2O-3 Distributed Scalable Machine Learning Models (DL/GLM/GBM/DRF/NB/AutoML)]
    * [knearestneighbour.py](./models/algorithms/knearestneighbour.py) [K-Nearest Neighbor implementation by sklearn. For small data (< 200k rows).]
    * [linear_svm.py](./models/algorithms/linear_svm.py) [Linear Support Vector Machine (SVM) implementation by sklearn. For small data.]
  * [CUSTOM_LOSS](./models/custom_loss)
    * [lightgbm_with_custom_loss.py](./models/custom_loss/lightgbm_with_custom_loss.py) [Modified version of Driverless AI's internal LightGBM implementation with a custom objective function (used for tree split finding).]
    * [xgboost_with_custom_loss.py](./models/custom_loss/xgboost_with_custom_loss.py) [Modified version of Driverless AI's internal XGBoost implementation with a custom objective function (used for tree split finding).]
  * [TIMESERIES](./models/timeseries)
    * [exponential_smoothing.py](./models/timeseries/exponential_smoothing.py) [Linear Model on top of Exponential Weighted Moving Average Lags for Time-Series. Provide appropriate lags and past outcomes during batch scoring for best results.]
    * [historic_mean.py](./models/timeseries/historic_mean.py) [Historic Mean for Time-Series problems. Predicts the mean of the target for each timegroup for regression problems.]
* [RECIPES](./recipes)
  * [amazon.py](./recipes/amazon.py) [Recipe for Kaggle Competition: Amazon.com - Employee Access Challenge]
* [SCORERS](./scorers)
  * [huber_loss.py](./scorers/huber_loss.py) [Huber Loss for Regression or Binary Classification. Robust loss, combination of quadratic loss and linear loss.]
  * [CLASSIFICATION](./scorers/classification)
    * [precision.py](./scorers/classification/precision.py) [Precision: `TP / (TP + FP)`. Binary uses threshold of 0.5, multiclass uses argmax to assign labels.]
    * [recall.py](./scorers/classification/recall.py) [Recall: `TP / (TP + FN)`. Binary uses threshold of 0.5, multiclass uses argmax to assign labels.]
    * [BINARY](./scorers/classification/binary)
      * [average_mcc.py](./scorers/classification/binary/average_mcc.py) [Averaged Matthews Correlation Coefficient (averaged over several thresholds, for imbalanced problems). Example how to use Driverless AI's internal scorer.]
      * [brier_loss.py](./scorers/classification/binary/brier_loss.py) [Brier Loss]
      * [cost.py](./scorers/classification/binary/cost.py) [Using hard-corded dollar amounts x for false positives and y for false negatives, calculate the cost of a model using: `x * FP + y * FN`]
      * [false_discovery_rate.py](./scorers/classification/binary/false_discovery_rate.py) [False Discovery Rate: `FP / (FP + TP) for binary classification. Threshold of 0.1 for assigning labels.`]
    * [MULTICLASS](./scorers/classification/multiclass)
      * [hamming_loss.py](./scorers/classification/multiclass/hamming_loss.py) [Hamming Loss - Misclassification Rate (1 - Accuracy)]
      * [quadratic_weighted_kappa.py](./scorers/classification/multiclass/quadratic_weighted_kappa.py) [Qudratic Weighted Kappa]
  * [REGRESSION](./scorers/regression)
    * [cosh_loss.py](./scorers/regression/cosh_loss.py) [Hyperbolic Cosine Loss]
    * [explained_variance.py](./scorers/regression/explained_variance.py) [Explained Variance. Fraction of variance that is explained by the model.]
    * [largest_error.py](./scorers/regression/largest_error.py) [Largest error for regression problems. Highly sensitive to outliers.]
    * [mean_absolute_scaled_error.py](./scorers/regression/mean_absolute_scaled_error.py) [Mean Absolute Scaled Error for time-series regression]
    * [median_absolute_error.py](./scorers/regression/median_absolute_error.py) [Median Absolute Error for regression]
    * [pearson_correlation.py](./scorers/regression/pearson_correlation.py) [Pearson Correlation Coefficient for regression]
    * [top_decile.py](./scorers/regression/top_decile.py) [Median Absolute Error for predictions in the top decile]
* [TRANSFORMERS](./transformers)
  * [how_to_debug_transformer.py](./transformers/how_to_debug_transformer.py) [Example how to debug a transformer outside of Driverless AI (optional)]
  * [how_to_test_from_py_client.py](./transformers/how_to_test_from_py_client.py) [Functions to ease testing a new custom transformer from the python client]
  * [AUGMENTATION](./transformers/augmentation)
    * [germany_landers_holidays.py](./transformers/augmentation/germany_landers_holidays.py) [Returns a flag for whether a date falls on a holiday for each of Germany's Bundeslaender]
    * [is_ramadan.py](./transformers/augmentation/is_ramadan.py) [Returns a flag for whether a date falls on Ramadan in Saudi Arabia]
    * [singapore_public_holidays.py](./transformers/augmentation/singapore_public_holidays.py) [Flag for whether a date falls on a public holiday in Singapore.]
  * [DATETIME](./transformers/datetime)
    * [datetime_diff_transformer.py](./transformers/datetime/datetime_diff_transformer.py) [Difference in time between two datetime columns]
    * [datetime_encoder_transformer.py](./transformers/datetime/datetime_encoder_transformer.py) [Converts datetime column into an integer (milliseconds since 1970)]
    * [days_until_dec2020.py](./transformers/datetime/days_until_dec2020.py) [Creates new feature for any date columns, by computing the difference in days between the date value and 31st Dec 2020]
  * [GENERIC](./transformers/generic)
    * [count_missing_values_transformer.py](./transformers/generic/count_missing_values_transformer.py) [Count of missing values per row]
    * [missing_flag_transformer.py](./transformers/generic/missing_flag_transformer.py) [Returns 1 if a value is missing, or 0 otherwise]
    * [specific_column_transformer.py](./transformers/generic/specific_column_transformer.py) [Example of a transformer that operates on the entire original frame, and hence on any column(s) desired.]
  * [GEOSPATIAL](./transformers/geospatial)
    * [geodesic.py](./transformers/geospatial/geodesic.py) [Calculates the distance in miles between two latitude/longitude points in space]
    * [myhaversine.py](./transformers/geospatial/myhaversine.py) [Computes miles between first two *_latitude and *_longitude named columns in the data set]
  * [IMAGE](./transformers/image)
    * [image_url_transformer.py](./transformers/image/image_url_transformer.py) [Convert a path to an image (JPG/JPEG/PNG) to a vector of class probabilities created by a pretrained ImageNet deeplearning model (Keras, TensorFlow).]
  * [NLP](./transformers/nlp)
    * [fuzzy_text_similarity_transformers.py](./transformers/nlp/fuzzy_text_similarity_transformers.py) [Row-by-row similarity between two text columns based on FuzzyWuzzy]
    * [text_embedding_similarity_transformers.py](./transformers/nlp/text_embedding_similarity_transformers.py) [Row-by-row similarity between two text columns based on pretrained Deep Learning embedding space]
    * [text_lang_detect_transformer.py](./transformers/nlp/text_lang_detect_transformer.py) [Detect the language for a text value using Google's 'langdetect' package]
    * [text_meta_transformers.py](./transformers/nlp/text_meta_transformers.py) [Extract common meta features from text]
    * [text_sentiment_transformer.py](./transformers/nlp/text_sentiment_transformer.py) [Extract sentiment from text using pretrained models from TextBlob]
    * [text_similarity_transformers.py](./transformers/nlp/text_similarity_transformers.py) [Row-by-row similarity between two text columns based on common N-grams, Jaccard similarity and edit distance.]
  * [NUMERIC](./transformers/numeric)
    * [boxcox_transformer.py](./transformers/numeric/boxcox_transformer.py) [Box-Cox Transform]
    * [exp_diff_transformer.py](./transformers/numeric/exp_diff_transformer.py) [Exponentiated difference of two numbers]
    * [log_transformer.py](./transformers/numeric/log_transformer.py) [Converts numbers to their Logarithm]
    * [random_transformer.py](./transformers/numeric/random_transformer.py) [Creates random numbers]
    * [round_transformer.py](./transformers/numeric/round_transformer.py) [Rounds numbers to 1, 2 or 3 decimals]
  * [OUTLIERS](./transformers/outliers)
    * [h2o3-dl-anomaly.py](./transformers/outliers/h2o3-dl-anomaly.py) [Anomaly score for each row based on reconstruction error of a H2O-3 deep learning autoencoder]
    * [quantile_winsorizer.py](./transformers/outliers/quantile_winsorizer.py) [Winsorizes (truncates) univariate outliers outside of a given quantile threshold]
    * [twosigma_winsorizer.py](./transformers/outliers/twosigma_winsorizer.py) [Winsorizes (truncates) univariate outliers outside of two standard deviations from the mean.]
  * [STRING](./transformers/string)
    * [strlen_transformer.py](./transformers/string/strlen_transformer.py) [Returns the string length of categorical values]
    * [to_string_transformer.py](./transformers/string/to_string_transformer.py) [Converts numbers to strings]
  * [TARGETENCODING](./transformers/targetencoding)
    * [leaky_mean_target_encoder.py](./transformers/targetencoding/leaky_mean_target_encoder.py) [Example implementation of a out-of-fold target encoder (leaky, not recommended)]
  * [TIMESERIES](./transformers/timeseries)
    * [auto_arima_forecast.py](./transformers/timeseries/auto_arima_forecast.py) [Auto ARIMA transformer is a time series transformer that predicts target using ARIMA models]
    * [general_time_series_transformer.py](./transformers/timeseries/general_time_series_transformer.py) [Demonstrates the API for custom time-series transformers.]
    * [normalized_macd.py](./transformers/timeseries/normalized_macd.py) [please add description]
    * [parallel_auto_arima_forecast.py](./transformers/timeseries/parallel_auto_arima_forecast.py) [Parallel Auto ARIMA transformer is a time series transformer that predicts target using ARIMA models.In this implementation, Time Group Models are fitted in parallel]
    * [parallel_prophet_forecast.py](./transformers/timeseries/parallel_prophet_forecast.py) [Parallel FB Prophet transformer is a time series transformer that predicts target using FBProhet models.In this implementation, Time Group Models are fitted in parallel]
    * [serial_prophet_forecast.py](./transformers/timeseries/serial_prophet_forecast.py) [Transformer that uses FB Prophet for time series prediction.Please see the parallel implementation for more information]
    * [time_encoder_transformer.py](./transformers/timeseries/time_encoder_transformer.py) [please add description]
    * [trading_volatility.py](./transformers/timeseries/trading_volatility.py) [Calculates Historical Volatility for numeric features (makes assumptions on the data)]
