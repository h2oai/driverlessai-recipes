# Recipes for H2O Driverless AI

* [transformers](./transformers)
  * [how_to_debug_transformer.py](./transformers/how_to_debug_transformer.py)
  * [how_to_test_from_py_client.py](./transformers/how_to_test_from_py_client.py)
  * [targetencoding](./transformers/targetencoding)
    * [leaky_mean_target_encoder.py](./transformers/targetencoding/leaky_mean_target_encoder.py)
  * [datetime](./transformers/datetime)
    * [days_until_dec2020.py](./transformers/datetime/days_until_dec2020.py)
    * [parse_excel_date_transformer.py](./transformers/datetime/parse_excel_date_transformer.py)
    * [datetime_encoder_transformer.py](./transformers/datetime/datetime_encoder_transformer.py)
    * [datetime_diff_transformer.py](./transformers/datetime/datetime_diff_transformer.py)
  * [numeric](./transformers/numeric)
    * [random_transformer.py](./transformers/numeric/random_transformer.py)
    * [log_transformer.py](./transformers/numeric/log_transformer.py)
    * [exp_diff_transformer.py](./transformers/numeric/exp_diff_transformer.py)
    * [round_transformer.py](./transformers/numeric/round_transformer.py)
  * [outliers](./transformers/outliers)
    * [quantile_winsorizer.py](./transformers/outliers/quantile_winsorizer.py)
    * [twosigma_winsorizer.py](./transformers/outliers/twosigma_winsorizer.py)
    * [h2o3-dl-anomaly.py](./transformers/outliers/h2o3-dl-anomaly.py)
  * [string](./transformers/string)
    * [to_string_transformer.py](./transformers/string/to_string_transformer.py)
    * [strlen_transformer.py](./transformers/string/strlen_transformer.py)
  * [geospatial](./transformers/geospatial)
    * [geodesic.py](./transformers/geospatial/geodesic.py)
    * [myhaversine.py](./transformers/geospatial/myhaversine.py)
  * [timeseries](./transformers/timeseries)
    * [parallel_auto_arima_forecast.py](./transformers/timeseries/parallel_auto_arima_forecast.py)
    * [auto_arima_forecast.py](./transformers/timeseries/auto_arima_forecast.py)
    * [time_encoder_transformer.py](./transformers/timeseries/time_encoder_transformer.py)
    * [serial_prophet_forecast.py](./transformers/timeseries/serial_prophet_forecast.py)
    * [general_time_series_transformer.py](./transformers/timeseries/general_time_series_transformer.py)
    * [parallel_prophet_forecast.py](./transformers/timeseries/parallel_prophet_forecast.py)
    * [normalized_macd.py](./transformers/timeseries/normalized_macd.py)
    * [trading_volatility.py](./transformers/timeseries/trading_volatility.py)
  * [augmentation](./transformers/augmentation)
    * [is_ramadan.py](./transformers/augmentation/is_ramadan.py)
    * [singapore_public_holidays.py](./transformers/augmentation/singapore_public_holidays.py)
    * [france_bank_holidays.py](./transformers/augmentation/france_bank_holidays.py)
  * [nlp](./transformers/nlp)
    * [text_embedding_similarity_transformers.py](./transformers/nlp/text_embedding_similarity_transformers.py)
    * [text_lang_detect_transformer.py](./transformers/nlp/text_lang_detect_transformer.py)
    * [text_similarity_transformers.py](./transformers/nlp/text_similarity_transformers.py)
    * [fuzzy_text_similarity_transformers.py](./transformers/nlp/fuzzy_text_similarity_transformers.py)
    * [text_meta_transformers.py](./transformers/nlp/text_meta_transformers.py)
    * [text_sentiment_transformer.py](./transformers/nlp/text_sentiment_transformer.py)
  * [generic](./transformers/generic)
    * [count_missing_values_transformer.py](./transformers/generic/count_missing_values_transformer.py)
    * [specific_column_transformer.py](./transformers/generic/specific_column_transformer.py)
  * [image](./transformers/image)
    * [image_url_transformer.py](./transformers/image/image_url_transformer.py)
* [recipes](./recipes)
  * [amazon.py](./recipes/amazon.py)
* [models](./models)
  * [knearestneighbour.py](./models/knearestneighbour.py)
  * [linear_svm.py](./models/linear_svm.py)
  * [historic_mean.py](./models/historic_mean.py)
  * [catboost.py](./models/catboost.py)
  * [h2o-3-models.py](./models/h2o-3-models.py)
  * [xgboost_with_custom_loss.py](./models/xgboost_with_custom_loss.py)
  * [lightgbm_with_custom_loss.py](./models/lightgbm_with_custom_loss.py)
* [scorers](./scorers)
  * [mean_absolute_percentage_deviation.py](./scorers/mean_absolute_percentage_deviation.py)
  * [precision.py](./scorers/precision.py)
  * [recall.py](./scorers/recall.py)
  * [hamming_loss.py](./scorers/hamming_loss.py)
  * [explained_variance.py](./scorers/explained_variance.py)
  * [huber_loss.py](./scorers/huber_loss.py)
  * [average_mcc.py](./scorers/average_mcc.py)
  * [pearson_correlation.py](./scorers/pearson_correlation.py)
  * [median_absolute_error.py](./scorers/median_absolute_error.py)
  * [mean_absolute_scaled_error.py](./scorers/mean_absolute_scaled_error.py)
  * [cost.py](./scorers/cost.py)
  * [largest_error.py](./scorers/largest_error.py)
  * [quadratic_weighted_kappa.py](./scorers/quadratic_weighted_kappa.py)
  * [top_decile.py](./scorers/top_decile.py)
  * [false_discovery_rate.py](./scorers/false_discovery_rate.py)
  * [brier_loss.py](./scorers/brier_loss.py)
