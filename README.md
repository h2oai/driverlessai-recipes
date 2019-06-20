# Recipes for H2O Driverless AI

* [transformers](./transformers)
  * [how_to_debug_transformer.py](./transformers/how_to_debug_transformer.py) [please add documentation]
  * [how_to_test_from_py_client.py](./transformers/how_to_test_from_py_client.py) [please add documentation]
  * [targetencoding](./transformers/targetencoding)
    * [leaky_mean_target_encoder.py](./transformers/targetencoding/leaky_mean_target_encoder.py) [please add documentation]
  * [datetime](./transformers/datetime)
    * [days_until_dec2020.py](./transformers/datetime/days_until_dec2020.py) [please add documentation]
    * [parse_excel_date_transformer.py](./transformers/datetime/parse_excel_date_transformer.py) [please add documentation]
    * [datetime_encoder_transformer.py](./transformers/datetime/datetime_encoder_transformer.py) [please add documentation]
    * [datetime_diff_transformer.py](./transformers/datetime/datetime_diff_transformer.py) [please add documentation]
  * [numeric](./transformers/numeric)
    * [random_transformer.py](./transformers/numeric/random_transformer.py) [please add documentation]
    * [log_transformer.py](./transformers/numeric/log_transformer.py) [please add documentation]
    * [exp_diff_transformer.py](./transformers/numeric/exp_diff_transformer.py) [please add documentation]
    * [round_transformer.py](./transformers/numeric/round_transformer.py) [please add documentation]
  * [outliers](./transformers/outliers)
    * [quantile_winsorizer.py](./transformers/outliers/quantile_winsorizer.py) [please add documentation]
    * [twosigma_winsorizer.py](./transformers/outliers/twosigma_winsorizer.py) [please add documentation]
    * [h2o3-dl-anomaly.py](./transformers/outliers/h2o3-dl-anomaly.py) [please add documentation]
  * [string](./transformers/string)
    * [to_string_transformer.py](./transformers/string/to_string_transformer.py) [please add documentation]
    * [strlen_transformer.py](./transformers/string/strlen_transformer.py) [please add documentation]
  * [geospatial](./transformers/geospatial)
    * [geodesic.py](./transformers/geospatial/geodesic.py) [please add documentation]
    * [myhaversine.py](./transformers/geospatial/myhaversine.py) [please add documentation]
  * [timeseries](./transformers/timeseries)
    * [parallel_auto_arima_forecast.py](./transformers/timeseries/parallel_auto_arima_forecast.py) [please add documentation]
    * [auto_arima_forecast.py](./transformers/timeseries/auto_arima_forecast.py) [please add documentation]
    * [time_encoder_transformer.py](./transformers/timeseries/time_encoder_transformer.py) [please add documentation]
    * [serial_prophet_forecast.py](./transformers/timeseries/serial_prophet_forecast.py) [please add documentation]
    * [general_time_series_transformer.py](./transformers/timeseries/general_time_series_transformer.py) [please add documentation]
    * [parallel_prophet_forecast.py](./transformers/timeseries/parallel_prophet_forecast.py) [please add documentation]
    * [normalized_macd.py](./transformers/timeseries/normalized_macd.py) [please add documentation]
    * [trading_volatility.py](./transformers/timeseries/trading_volatility.py) [please add documentation]
  * [augmentation](./transformers/augmentation)
    * [is_ramadan.py](./transformers/augmentation/is_ramadan.py) [please add documentation]
    * [singapore_public_holidays.py](./transformers/augmentation/singapore_public_holidays.py) [please add documentation]
    * [france_bank_holidays.py](./transformers/augmentation/france_bank_holidays.py) [please add documentation]
  * [nlp](./transformers/nlp)
    * [text_embedding_similarity_transformers.py](./transformers/nlp/text_embedding_similarity_transformers.py) [please add documentation]
    * [text_lang_detect_transformer.py](./transformers/nlp/text_lang_detect_transformer.py) [please add documentation]
    * [text_similarity_transformers.py](./transformers/nlp/text_similarity_transformers.py) [please add documentation]
    * [fuzzy_text_similarity_transformers.py](./transformers/nlp/fuzzy_text_similarity_transformers.py) [please add documentation]
    * [text_meta_transformers.py](./transformers/nlp/text_meta_transformers.py) [please add documentation]
    * [text_sentiment_transformer.py](./transformers/nlp/text_sentiment_transformer.py) [please add documentation]
  * [generic](./transformers/generic)
    * [count_missing_values_transformer.py](./transformers/generic/count_missing_values_transformer.py) [please add documentation]
    * [specific_column_transformer.py](./transformers/generic/specific_column_transformer.py) [please add documentation]
  * [image](./transformers/image)
    * [image_url_transformer.py](./transformers/image/image_url_transformer.py) [please add documentation]
* [recipes](./recipes)
  * [amazon.py](./recipes/amazon.py) [please add documentation]
* [models](./models)
  * [knearestneighbour.py](./models/knearestneighbour.py) [please add documentation]
  * [linear_svm.py](./models/linear_svm.py) [please add documentation]
  * [historic_mean.py](./models/historic_mean.py) [please add documentation]
  * [catboost.py](./models/catboost.py) [please add documentation]
  * [h2o-3-models.py](./models/h2o-3-models.py) [please add documentation]
  * [xgboost_with_custom_loss.py](./models/xgboost_with_custom_loss.py) [The intention of this custom model is to demonstrate how one can use the framework to use XGBoost with
    a custom objective function, i.e. a loss function that XGBoost would optimize for.
    The code is almost a clone of lightgbm_with_custom_loss.py
    only parent class changes.
    ]
  * [lightgbm_with_custom_loss.py](./models/lightgbm_with_custom_loss.py) [The intention of this custom model is to demonstrate how one can use the framework to use LightGBM with
    a custom objective function, i.e. a loss function that LightGBM would optimize for.
    ]
* [scorers](./scorers)
  * [mean_absolute_percentage_deviation.py](./scorers/mean_absolute_percentage_deviation.py) [please add documentation]
  * [precision.py](./scorers/precision.py) [please add documentation]
  * [recall.py](./scorers/recall.py) [please add documentation]
  * [hamming_loss.py](./scorers/hamming_loss.py) [please add documentation]
  * [explained_variance.py](./scorers/explained_variance.py) [please add documentation]
  * [huber_loss.py](./scorers/huber_loss.py) [please add documentation]
  * [average_mcc.py](./scorers/average_mcc.py) [please add documentation]
  * [pearson_correlation.py](./scorers/pearson_correlation.py) [please add documentation]
  * [median_absolute_error.py](./scorers/median_absolute_error.py) [please add documentation]
  * [mean_absolute_scaled_error.py](./scorers/mean_absolute_scaled_error.py) [please add documentation]
  * [cost.py](./scorers/cost.py) [please add documentation]
  * [largest_error.py](./scorers/largest_error.py) [please add documentation]
  * [quadratic_weighted_kappa.py](./scorers/quadratic_weighted_kappa.py) [please add documentation]
  * [top_decile.py](./scorers/top_decile.py) [please add documentation]
  * [false_discovery_rate.py](./scorers/false_discovery_rate.py) [please add documentation]
  * [brier_loss.py](./scorers/brier_loss.py) [please add documentation]
