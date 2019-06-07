# Recipes for H2O Driverless AI

|-- ./data
|   |-- ./data/taxi_small.csv
|   `-- ./data/weather_missing.csv
|-- ./models
|   |-- ./models/catboost.py
|   |-- ./models/h2o-3-models.py
|   |-- ./models/historic_mean.py
|   |-- ./models/knearestneighbour.py
|   |-- ./models/lightgbm_with_custom_loss.py
|   |-- ./models/linear_svm.py
|   |-- ./models/sklearn_extra_trees.py
|   `-- ./models/xgboost_with_custom_loss.py
|-- ./recipes
|   `-- ./recipes/amazon.py
|-- ./scorers
|   |-- ./scorers/average_mcc.py
|   |-- ./scorers/brier_loss.py
|   |-- ./scorers/cost.py
|   |-- ./scorers/explained_variance.py
|   |-- ./scorers/false_discovery_rate.py
|   |-- ./scorers/hamming_loss.py
|   |-- ./scorers/huber_loss.py
|   |-- ./scorers/largest_error.py
|   |-- ./scorers/mean_absolute_percentage_deviation.py
|   |-- ./scorers/mean_absolute_scaled_error.py
|   |-- ./scorers/median_absolute_error.py
|   |-- ./scorers/pearson_correlation.py
|   |-- ./scorers/precision.py
|   |-- ./scorers/quadratic_weighted_kappa.py
|   |-- ./scorers/recall.py
|   `-- ./scorers/top_decile.py
`-- ./transformers
    |-- ./transformers/augmentation
    |   |-- ./transformers/augmentation/france_bank_holidays.py
    |   `-- ./transformers/augmentation/singapore_public_holidays.py
    |-- ./transformers/datetime
    |   |-- ./transformers/datetime/datetime_diff_transformer.py
    |   |-- ./transformers/datetime/datetime_encoder_transformer.py
    |   `-- ./transformers/datetime/parse_excel_date_transformer.py
    |-- ./transformers/generic
    |   |-- ./transformers/generic/count_missing_values_transformer.py
    |   `-- ./transformers/generic/specific_column_transformer.py
    |-- ./transformers/geospatial
    |   |-- ./transformers/geospatial/geodesic.py
    |   `-- ./transformers/geospatial/myhaversine.py
    |-- ./transformers/how_to_debug_transformer.py
    |-- ./transformers/how_to_test_from_py_client.py
    |-- ./transformers/image
    |   `-- ./transformers/image/image_url_transformer.py
    |-- ./transformers/nlp
    |   |-- ./transformers/nlp/fuzzy_text_similarity_transformers.py
    |   |-- ./transformers/nlp/text_embedding_similarity_transformers.py
    |   |-- ./transformers/nlp/text_lang_detect_transformer.py
    |   |-- ./transformers/nlp/text_meta_transformers.py
    |   |-- ./transformers/nlp/text_sentiment_transformer.py
    |   `-- ./transformers/nlp/text_similarity_transformers.py
    |-- ./transformers/numeric
    |   |-- ./transformers/numeric/exp_diff_transformer.py
    |   |-- ./transformers/numeric/log_transformer.py
    |   |-- ./transformers/numeric/random_transformer.py
    |   `-- ./transformers/numeric/round_transformer.py
    |-- ./transformers/outliers
    |   |-- ./transformers/outliers/h2o3-dl-anomaly.py
    |   |-- ./transformers/outliers/quantile_winsorizer.py
    |   `-- ./transformers/outliers/twosigma_winsorizer.py
    |-- ./transformers/string
    |   |-- ./transformers/string/strlen_transformer.py
    |   `-- ./transformers/string/to_string_transformer.py
    |-- ./transformers/targetencoding
    |   `-- ./transformers/targetencoding/leaky_mean_target_encoder.py
    `-- ./transformers/timeseries
        |-- ./transformers/timeseries/auto_arima_forecast.py
        |-- ./transformers/timeseries/general_time_series_transformer.py
        |-- ./transformers/timeseries/normalized_macd.py
        |-- ./transformers/timeseries/parallel_auto_arima_forecast.py
        |-- ./transformers/timeseries/parallel_prophet_forecast.py
        |-- ./transformers/timeseries/serial_prophet_forecast.py
        |-- ./transformers/timeseries/time_encoder_transformer.py
        `-- ./transformers/timeseries/trading_volatility.py