# Recipes for H2O Driverless AI

* [TRANSFORMERS](./transformers)
  * [how_to_debug_transformer.py](./transformers/how_to_debug_transformer.py) [please add documentation]
  * [how_to_test_from_py_client.py](./transformers/how_to_test_from_py_client.py) [please add documentation]
  * [TARGETENCODING](./transformers/targetencoding)
    * [leaky_mean_target_encoder.py](./transformers/targetencoding/leaky_mean_target_encoder.py) [please add documentation]
  * [DATETIME](./transformers/datetime)
    * [days_until_dec2020.py](./transformers/datetime/days_until_dec2020.py) [Creates new feature for any date columns, by computing the difference in days between the date value and 31st Dec 2020]
    * [parse_excel_date_transformer.py](./transformers/datetime/parse_excel_date_transformer.py) [please add documentation]
    * [datetime_encoder_transformer.py](./transformers/datetime/datetime_encoder_transformer.py) [please add documentation]
    * [datetime_diff_transformer.py](./transformers/datetime/datetime_diff_transformer.py) [please add documentation]
  * [NUMERIC](./transformers/numeric)
    * [random_transformer.py](./transformers/numeric/random_transformer.py) [please add documentation]
    * [log_transformer.py](./transformers/numeric/log_transformer.py) [please add documentation]
    * [exp_diff_transformer.py](./transformers/numeric/exp_diff_transformer.py) [please add documentation]
    * [round_transformer.py](./transformers/numeric/round_transformer.py) [please add documentation]
  * [OUTLIERS](./transformers/outliers)
    * [quantile_winsorizer.py](./transformers/outliers/quantile_winsorizer.py) [please add documentation]
    * [twosigma_winsorizer.py](./transformers/outliers/twosigma_winsorizer.py) [please add documentation]
    * [h2o3-dl-anomaly.py](./transformers/outliers/h2o3-dl-anomaly.py) [please add documentation]
  * [STRING](./transformers/string)
    * [to_string_transformer.py](./transformers/string/to_string_transformer.py) [please add documentation]
    * [strlen_transformer.py](./transformers/string/strlen_transformer.py) [please add documentation]
  * [GEOSPATIAL](./transformers/geospatial)
    * [geodesic.py](./transformers/geospatial/geodesic.py) [please add documentation]
    * [myhaversine.py](./transformers/geospatial/myhaversine.py) [Computes miles between first two *_latitude and *_longitude named columns in the data set]
  * [TIMESERIES](./transformers/timeseries)
    * [parallel_auto_arima_forecast.py](./transformers/timeseries/parallel_auto_arima_forecast.py) [please add documentation]
    * [auto_arima_forecast.py](./transformers/timeseries/auto_arima_forecast.py) [please add documentation]
    * [time_encoder_transformer.py](./transformers/timeseries/time_encoder_transformer.py) [please add documentation]
    * [serial_prophet_forecast.py](./transformers/timeseries/serial_prophet_forecast.py) [please add documentation]
    * [general_time_series_transformer.py](./transformers/timeseries/general_time_series_transformer.py) [please add documentation]
    * [parallel_prophet_forecast.py](./transformers/timeseries/parallel_prophet_forecast.py) [please add documentation]
    * [normalized_macd.py](./transformers/timeseries/normalized_macd.py) [please add documentation]
    * [trading_volatility.py](./transformers/timeseries/trading_volatility.py) [please add documentation]
  * [AUGMENTATION](./transformers/augmentation)
    * [germany_landers_holidays.py](./transformers/augmentation/germany_landers_holidays.py) [please add documentation]
    * [is_ramadan.py](./transformers/augmentation/is_ramadan.py) [please add documentation]
    * [singapore_public_holidays.py](./transformers/augmentation/singapore_public_holidays.py) [please add documentation]
  * [NLP](./transformers/nlp)
    * [text_embedding_similarity_transformers.py](./transformers/nlp/text_embedding_similarity_transformers.py) [please add documentation]
    * [text_lang_detect_transformer.py](./transformers/nlp/text_lang_detect_transformer.py) [please add documentation]
    * [text_similarity_transformers.py](./transformers/nlp/text_similarity_transformers.py) [please add documentation]
    * [fuzzy_text_similarity_transformers.py](./transformers/nlp/fuzzy_text_similarity_transformers.py) [please add documentation]
    * [text_meta_transformers.py](./transformers/nlp/text_meta_transformers.py) [please add documentation]
    * [text_sentiment_transformer.py](./transformers/nlp/text_sentiment_transformer.py) [please add documentation]
  * [GENERIC](./transformers/generic)
    * [count_missing_values_transformer.py](./transformers/generic/count_missing_values_transformer.py) [please add documentation]
    * [specific_column_transformer.py](./transformers/generic/specific_column_transformer.py) [please add documentation]
  * [IMAGE](./transformers/image)
    * [image_url_transformer.py](./transformers/image/image_url_transformer.py) [please add documentation]
* [RECIPES](./recipes)
  * [amazon.py](./recipes/amazon.py) [Recipe for Kaggle Competition: Amazon.com - Employee Access Challenge]
* [MODELS](./models)
  * [knearestneighbour.py](./models/knearestneighbour.py) [K-Nearest Neighbor implementation by sklearn. For small data (< 200k rows).]
  * [linear_svm.py](./models/linear_svm.py) [Linear Support Vector Machine (SVM) implementation by sklearn. For small data.]
  * [historic_mean.py](./models/historic_mean.py) [Historic Mean for Time-Series problems. Predicts the mean of the target for each timegroup for regression problems.]
  * [catboost.py](./models/catboost.py) [CatBoost gradient boosting by Yandex. Currently supports regression and binary classification.]
  * [h2o-3-models.py](./models/h2o-3-models.py) [H2O-3 Distributed Scalable Machine Learning Models (DL/GLM/GBM/DRF/NB)]
  * [xgboost_with_custom_loss.py](./models/xgboost_with_custom_loss.py) [Modified version of Driverless AI's internal XGBoost implementation with a custom objective function (used for tree split finding).]
  * [lightgbm_with_custom_loss.py](./models/lightgbm_with_custom_loss.py) [Modified version of Driverless AI's internal LightGBM implementation with a custom objective function (used for tree split finding).]
* [SCORERS](./scorers)
  * [mean_absolute_relative_deviation.py](./scorers/mean_absolute_relative_deviation.py) [Mean absolute relative deviation mean(abs(actual-predicted)/predicted)]
  * [precision.py](./scorers/precision.py) [Precision: `TP / (TP + FP)`]
  * [recall.py](./scorers/recall.py) [Recall: `TP / (TP + FN). Binary uses threshold of 0.5, multiclass uses argmax to assign labels.`]
  * [hamming_loss.py](./scorers/hamming_loss.py) [Hamming Loss - Misclassification Rate (1 - Accuracy)]
  * [explained_variance.py](./scorers/explained_variance.py) [Explained Variance. Fraction of variance that is explained by the model.]
  * [huber_loss.py](./scorers/huber_loss.py) [Huber Loss for Regression or Binary Classification. Robust loss, combination of quadratic loss and linear loss.]
  * [average_mcc.py](./scorers/average_mcc.py) [Averaged Matthews Correlation Coefficient (averaged over several thresholds, for imbalanced problems)]
  * [pearson_correlation.py](./scorers/pearson_correlation.py) [Pearson Correlation Coefficient for regression]
  * [median_absolute_error.py](./scorers/median_absolute_error.py) [Median Absolute Error for regression]
  * [mean_absolute_scaled_error.py](./scorers/mean_absolute_scaled_error.py) [Mean Absolute Scaled Error for time-series regression]
  * [cost.py](./scorers/cost.py) [Using hard-corded dollar amounts x for false positives and y for false negatives, calculate the cost of a model using: `x * FP + y * FN`]
  * [largest_error.py](./scorers/largest_error.py) [Largest error for regression problems. Highly sensitive to outliers.]
  * [quadratic_weighted_kappa.py](./scorers/quadratic_weighted_kappa.py) [Qudratic Weighted Kappa]
  * [top_decile.py](./scorers/top_decile.py) [Median Absolute Error for predictions in the top decile]
  * [false_discovery_rate.py](./scorers/false_discovery_rate.py) [False Discovery Rate: `FP / (FP + TP) for binary classification. Threshold of 0.1 for assigning labels.`]
  * [brier_loss.py](./scorers/brier_loss.py) [Brier Loss]
