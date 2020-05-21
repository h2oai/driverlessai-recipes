# Recipes for H2O Driverless AI

## About Driverless AI
H2O Driverless AI is Automatic Machine Learning for the Enterprise. Driverless AI automates feature engineering, model building, visualization and interpretability.
- Learn more about Driverless AI from the [H2O.ai website](https://www.h2o.ai/)
- Take the [test drive](https://www.h2o.ai/try-driverless-ai/)
- Go to the [Driverless AI community Slack channel](https://www.h2o.ai/community/driverless-ai-community/#chat) and ask your BYOR related questions in #general

## About BYOR
**BYOR** stands for **Bring Your Own Recipe** and is a key feature of Driverless AI. It allows domain scientists to solve their problems faster and with more precision.

## What are Custom Recipes?
Custom recipes are Python code snippets that can be uploaded into Driverless AI at runtime, like plugins. No need to restart Driverless AI. Custom recipes can be provided for transformers, models and scorers. During training of a supervised machine learning modeling pipeline (aka experiment), Driverless AI can then use these code snippets as building blocks, in combination with all built-in code pieces (or instead of). By providing your own custom recipes, you can gain control over the optimization choices that Driverless AI makes to best solve your machine learning problems.

## Best Practices for Recipes

### Security
* Recipes are meant to be built by people you trust and each recipe should be code-reviewed before going to production.
* Assume that a user with access to Driverless AI has access to the data inside that instance.
  * Apart from securing access to the instance via private networks, various methods of [authentication](http://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/authentication.html) are possible. Local authentication provides the most control over which users have access to Driverless AI.
  * Unless the `config.toml` setting `enable_dataset_downloading=false` is set, an authenticated user can download all imported datasets as .csv via direct APIs.
* When recipes are enabled (`enable_custom_recipes=true`, the default), be aware that:
  * The code for the recipes runs as the same native Linux user that runs the Driverless AI application.
    * Recipes have explicit access to all data passing through the transformer/model/scorer API
    * Recipes have implicit access to system resources such as disk, memory, CPUs, GPUs, network, etc.
  * A H2O-3 Java process is started in the background, for use by all recipes using H2O-3. Anyone with access to the Driverless AI instance can browse the file system, see models and data through the H2O-3 interface.

* Best ways to control access to Driverless AI and custom recipes:
  * Control access to the Driverless AI instance
  * Use local authentication to specify exactly which users are allowed to access Driverless AI
  * Run Driverless AI in a Docker container, as a certain user, with only certain ports exposed, and only certain mount points mapped
  * To disable all recipes: Set `enable_custom_recipes=false` in the config.toml, or add the environment variable `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES=0` at startup of Driverless AI. This will disable all custom transformers, models and scorers.
  * To disable new recipes: To keep all previously uploaded recipes enabled and disable the upload of any new recipes, set `enable_custom_recipes_upload=false` or `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES_UPLOAD=0` at startup of Driverless AI.

### Safety
* Driverless AI automatically performs basic acceptance tests for all custom recipes unless disabled
* More information in the FAQ

### Performance
* Use fast and efficient data manipulation tools like `data.table`, `sklearn`, `numpy` or `pandas` instead of Python lists, for-loops etc.
* Use disk sparingly, delete temporary files as soon as possible
* Use memory sparingly, delete objects when no longer needed

## Reference Guide
* [FAQ](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#faq)
* [Templates](https://github.com/h2oai/driverlessai-recipes/blob/master/FAQ.md#references)
* [Technical Architecture Diagram](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.png)

## Sample Recipes
[Go to Recipes for Driverless 1.7.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.7.0)
 [1.7.1](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.7.1)
 [1.8.0](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.0)
 [1.8.1](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.1)
 [1.8.2](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.2)
 [1.8.3](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.3)
 [1.8.4](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.4)
 [1.8.5](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.5)
 [1.8.6](https://github.com/h2oai/driverlessai-recipes/tree/rel-1.8.6)
### Count: 164
* [DATA](./data)
  * [GroupAgg.py](./data/GroupAgg.py) [Aggregation features on numeric columns across multiple categorical columns]
  * [KMeansClustering.py](./data/KMeansClustering.py) [Data Recipe to perform KMeans Clustering on a dataset.]
  * [airlines.py](./data/airlines.py) [Create airlines dataset]
  * [airlines_joined_data_flights_in_out.py](./data/airlines_joined_data_flights_in_out.py) [Create augmented airlines datasets]
  * [airlines_joined_data_flights_in_out_regression.py](./data/airlines_joined_data_flights_in_out_regression.py) [Create augmented airlines datasets for regression]
  * [airlines_multiple.py](./data/airlines_multiple.py) [Create airlines dataset]
  * [catchallenge.py](./data/catchallenge.py) [Create cat challenge dataset]
  * [create_transactional_data_or_convert_to_iid.py](./data/create_transactional_data_or_convert_to_iid.py) [Example code to generate and convert transactional data to i.i.d. data.]
  * [creditcard.py](./data/creditcard.py) [Modify credit card dataset]
  * [data_template.py](./data/data_template.py) [Custom data recipe base class]
  * [feature_selection.py](./data/feature_selection.py) [please add description]
  * [ieee_data_puddle.py](./data/ieee_data_puddle.py) [Data recipe to prepare data for Kaggle IEEE-CIS Fraud Detection https://www.kaggle.com/c/ieee-fraud-detection]
  * [kaggle_bosch.py](./data/kaggle_bosch.py) [Create Bosch competition datasets with leak]
  * [kaggle_ieee_fraud.py](./data/kaggle_ieee_fraud.py) [Data recipe to prepare data for Kaggle IEEE-CIS Fraud Detection https://www.kaggle.com/c/ieee-fraud-detection]
  * [keywords_data.py](./data/keywords_data.py) [Check and match a list of words from a specific string column]
  * [load_sas7bdat.py](./data/load_sas7bdat.py) [Data Recipe to load a single sas file__version__ = 0.1authored by @mtanco (Michelle Tanco)Required User Defined Inputs: name of file to load]
  * [mnist.py](./data/mnist.py) [Prep and upload the MNIST datasset]
  * [mozilla_deepspeech_wav2txt.py](./data/mozilla_deepspeech_wav2txt.py) [Speech to text using Mozilla's DeepSpeechSettings for this recipe:Assing MODEL_PATH global variable prior to usageAssign WAV_COLNAME global variable with proper column name from your dataset.This colums should contain absolute paths to .wav file which needs to be converted to text.General requirements to .wav's:1 channel (mono)16 bit16000 frequency]
  * [seattle_rain_modify.py](./data/seattle_rain_modify.py) [Transpose the Monthly Seattle Rain Inches data set for Time Series use cases]
  * [seattle_rain_upload.py](./data/seattle_rain_upload.py) [Upload Monthly Seattle Rain Inches data set from data provided by the City of Seattle]
  * [ts_fill_n_cluster.py](./data/ts_fill_n_cluster.py) [Data Recipe to fill missing values in TS data and then create new data sets from TS Clustering]
  * [wav2txt.py](./data/wav2txt.py) [Speech to text using Azure Cognitive ServicesSettings for this recipe:Assing AZURE_SERVICE_KEY and AZURE_SERVICE_REGION global variable prior to usageAssign WAV_COLNAME global variable with proper column name from your dataset.This colums should contain absolute paths to .wav file which needs to be converted to text.]
  * [DATABASES](./data/databases)
    * [create_dataset_from_mongodb_collection.py](./data/databases/create_dataset_from_mongodb_collection.py) [Create dataset from MonogDB]
  * [NLP](./data/nlp)
    * [sentiment_score.py](./data/nlp/sentiment_score.py) [Data recipe to get sentiment score using textblob]
    * [sentiment_score_vader.py](./data/nlp/sentiment_score_vader.py) [Data recipe to get sentiment score using vader]
    * [text_summarization.py](./data/nlp/text_summarization.py) [Data recipe to get summary of text using gensim]
    * [tokenize_chinese.py](./data/nlp/tokenize_chinese.py) [Chinese text tokenization using jieba package - https://github.com/fxsjy/jieba]
    * [topic_modeling.py](./data/nlp/topic_modeling.py) [Data recipe to perform topic modeling]
* [HOW_TO_WRITE_A_RECIPE](./how_to_write_a_recipe)
  * [ExampleLogTransformer.py](./how_to_write_a_recipe/ExampleLogTransformer.py) [please add description]
* [MODELS](./models)
  * [model_template.py](./models/model_template.py) [Template base class for a custom model recipe.]
  * [ALGORITHMS](./models/algorithms)
    * [calibratedClassifier.py](./models/algorithms/calibratedClassifier.py) [ Calibrated Classifier Model: To calibrate predictions using Platt's scaling or Isotonic regression]
    * [catboost.py](./models/algorithms/catboost.py) [CatBoost gradient boosting by Yandex. Currently supports regression and binary classification.]
    * [daal_trees.py](./models/algorithms/daal_trees.py) [Binary Classification and Regression for Decision Forest and Gradient Boosting based on Intel DAAL]
    * [extra_trees.py](./models/algorithms/extra_trees.py) [Extremely Randomized Trees (ExtraTrees) model from sklearn]
    * [extremeClassifier.py](./models/algorithms/extremeClassifier.py) [ Extreme Classifier Model: To speed up train of multiclass model (100s of classes) for lightGBM.    Caution: can only be used for AUC (or GINI) and accuracy metrics.    Based on: Extreme Classification in Log Memory using Count-Min Sketch: https://arxiv.org/abs/1910.13830]
    * [h2o-3-gbm-poisson.py](./models/algorithms/h2o-3-gbm-poisson.py) [H2O-3 Distributed Scalable Machine Learning Models: Poisson GBM]
    * [h2o-3-models.py](./models/algorithms/h2o-3-models.py) [H2O-3 Distributed Scalable Machine Learning Models (DL/GLM/GBM/DRF/NB/AutoML)]
    * [h2o-glm-poisson.py](./models/algorithms/h2o-glm-poisson.py) [H2O-3 Distributed Scalable Machine Learning Models: Poisson GLM]
    * [knearestneighbour.py](./models/algorithms/knearestneighbour.py) [K-Nearest Neighbor implementation by sklearn. For small data (< 200k rows).]
    * [libfm_fastfm.py](./models/algorithms/libfm_fastfm.py) [LibFM implementation of fastFM ]
    * [linear_svm.py](./models/algorithms/linear_svm.py) [Linear Support Vector Machine (SVM) implementation by sklearn. For small data.]
    * [logistic_regression.py](./models/algorithms/logistic_regression.py) [Logistic Regression based upon sklearn.]
    * [nusvm.py](./models/algorithms/nusvm.py) [Nu-SVM implementation by sklearn. For small data.]
    * [random_forest.py](./models/algorithms/random_forest.py) [Random Forest (RandomForest) model from sklearn]
  * [CUSTOM_LOSS](./models/custom_loss)
    * [lightgbm_quantile_regression.py](./models/custom_loss/lightgbm_quantile_regression.py) [Modified version of Driverless AI's internal LightGBM implementation with for quantile regression]
    * [lightgbm_with_custom_loss.py](./models/custom_loss/lightgbm_with_custom_loss.py) [Modified version of Driverless AI's internal LightGBM implementation with a custom objective function (used for tree split finding).]
    * [xgboost_with_custom_loss.py](./models/custom_loss/xgboost_with_custom_loss.py) [Modified version of Driverless AI's internal XGBoost implementation with a custom objective function (used for tree split finding).]
  * [MLI](./models/mli)
    * [GA2M](./models/mli/ga2m)
      * [model_ga2m.py](./models/mli/ga2m/model_ga2m.py) [Explainable Boosting Machines (EBM), implementation of GA2M]
    * [XNN](./models/mli/xnn)
      * [model_xnn.py](./models/mli/xnn/model_xnn.py) [ Explainable neural net ]
  * [NLP](./models/nlp)
    * [text_tfidf_model.py](./models/nlp/text_tfidf_model.py) [Text classification / regression model using TFIDF]
    * [text_tfidf_model_continuous.py](./models/nlp/text_tfidf_model_continuous.py) [Text classification model using TFIDF]
  * [TIMESERIES](./models/timeseries)
    * [exponential_smoothing.py](./models/timeseries/exponential_smoothing.py) [Linear Model on top of Exponential Weighted Moving Average Lags for Time-Series. Provide appropriate lags and past outcomes during batch scoring for best results.]
    * [fb_prophet.py](./models/timeseries/fb_prophet.py) [Prophet by Facebook for TimeSeries with an example of parameter mutation.]
    * [fb_prophet_parallel.py](./models/timeseries/fb_prophet_parallel.py) [Prophet by Facebook for TimeSeries with an example of parameter mutation.]
    * [historic_mean.py](./models/timeseries/historic_mean.py) [Historic Mean for Time-Series problems. Predicts the mean of the target for each timegroup for regression problems.]
* [RECIPES](./recipes)
  * [amazon.py](./recipes/amazon.py) [Recipe for Kaggle Competition: Amazon.com - Employee Access Challenge]
* [REFERENCE](./reference)
* [SCORERS](./scorers)
  * [huber_loss.py](./scorers/huber_loss.py) [Huber Loss for Regression or Binary Classification. Robust loss, combination of quadratic loss and linear loss.]
  * [scorer_template.py](./scorers/scorer_template.py) [Template base class for a custom scorer recipe.]
  * [CLASSIFICATION](./scorers/classification)
    * [f3_score.py](./scorers/classification/f3_score.py) [F3 Score]
    * [f4_score.py](./scorers/classification/f4_score.py) [F4 Score]
    * [precision.py](./scorers/classification/precision.py) [Weighted Precision: `TP / (TP + FP)` at threshold for optimal F1 Score.]
    * [recall.py](./scorers/classification/recall.py) [Weighted Recall: `TP / (TP + FN)` at threshold for optimal F1 Score.]
    * [BINARY](./scorers/classification/binary)
      * [average_mcc.py](./scorers/classification/binary/average_mcc.py) [Averaged Matthews Correlation Coefficient (averaged over several thresholds, for imbalanced problems). Example how to use Driverless AI's internal scorer.]
      * [brier_loss.py](./scorers/classification/binary/brier_loss.py) [Brier Loss]
      * [cost.py](./scorers/classification/binary/cost.py) [Using hard-coded dollar amounts x for false positives and y for false negatives, calculate the cost of a model using: `(x * FP + y * FN) / N`]
      * [cost_access_to_data.py](./scorers/classification/binary/cost_access_to_data.py) [Same as CostBinary, but provides access to full Data]
      * [cost_smooth.py](./scorers/classification/binary/cost_smooth.py) [Using hard-coded dollar amounts x for false positives and y for false negatives, calculate the cost of a model using: `(1 - y_true) * y_pred * fp_cost + y_true * (1 - y_pred) * fn_cost`]
      * [false_discovery_rate.py](./scorers/classification/binary/false_discovery_rate.py) [Weighted False Discovery Rate: `FP / (FP + TP)` at threshold for optimal F1 Score.]
      * [logloss_with_costs.py](./scorers/classification/binary/logloss_with_costs.py) [Logloss with costs associated with each type of 4 outcomes - typically applicable to fraud use case]
      * [marketing_campaign.py](./scorers/classification/binary/marketing_campaign.py) [Computes the mean profit per outbound marketing letter, given a fraction of the population addressed, and fixed cost and reward]
      * [profit.py](./scorers/classification/binary/profit.py) [Uses domain information about user behavior to calculate the profit or loss of a model.]
    * [MULTICLASS](./scorers/classification/multiclass)
      * [hamming_loss.py](./scorers/classification/multiclass/hamming_loss.py) [Hamming Loss - Misclassification Rate (1 - Accuracy)]
      * [map@k.py](./scorers/classification/multiclass/map@k.py) [Mean Average Precision @ k (MAP@k)]
      * [quadratic_weighted_kappa.py](./scorers/classification/multiclass/quadratic_weighted_kappa.py) [Qudratic Weighted Kappa]
  * [REGRESSION](./scorers/regression)
    * [WAPE_scorer.py](./scorers/regression/WAPE_scorer.py) [Weighted Absoluted Percent Error]
    * [cosh_loss.py](./scorers/regression/cosh_loss.py) [Hyperbolic Cosine Loss]
    * [explained_variance.py](./scorers/regression/explained_variance.py) [Explained Variance. Fraction of variance that is explained by the model.]
    * [largest_error.py](./scorers/regression/largest_error.py) [Largest error for regression problems. Highly sensitive to outliers.]
    * [log_mae.py](./scorers/regression/log_mae.py) [Log Mean Absolute Error for regression]
    * [mean_absolute_scaled_error.py](./scorers/regression/mean_absolute_scaled_error.py) [Mean Absolute Scaled Error for time-series regression]
    * [mean_squared_log_error.py](./scorers/regression/mean_squared_log_error.py) [Mean Squared Log Error for regression]
    * [median_absolute_error.py](./scorers/regression/median_absolute_error.py) [Median Absolute Error for regression]
    * [pearson_correlation.py](./scorers/regression/pearson_correlation.py) [Pearson Correlation Coefficient for regression]
    * [quantile_loss.py](./scorers/regression/quantile_loss.py) [Quantile Loss regression]
    * [top_decile.py](./scorers/regression/top_decile.py) [Median Absolute Error for predictions in the top decile]
* [TRANSFORMERS](./transformers)
  * [how_to_debug_transformer.py](./transformers/how_to_debug_transformer.py) [Example how to debug a transformer outside of Driverless AI (optional)]
  * [how_to_test_from_py_client.py](./transformers/how_to_test_from_py_client.py) [Testing a BYOR Transformer the PyClient - works on 1.7.0 & 1.7.1-17]
  * [transformer_template.py](./transformers/transformer_template.py) [Template base class for a custom transformer recipe.]
  * [AUGMENTATION](./transformers/augmentation)
    * [germany_landers_holidays.py](./transformers/augmentation/germany_landers_holidays.py) [Returns a flag for whether a date falls on a holiday for each of Germany's Bundeslaender. ]
    * [ipaddress_features.py](./transformers/augmentation/ipaddress_features.py) [Parses IP addresses and networks and extracts its properties.]
    * [is_ramadan.py](./transformers/augmentation/is_ramadan.py) [Returns a flag for whether a date falls on Ramadan in Saudi Arabia]
    * [singapore_public_holidays.py](./transformers/augmentation/singapore_public_holidays.py) [Flag for whether a date falls on a public holiday in Singapore.]
    * [usairportcode_origin_dest.py](./transformers/augmentation/usairportcode_origin_dest.py) [Transformer to parse and augment US airport codes with geolocation info.]
    * [usairportcode_origin_dest_geo_features.py](./transformers/augmentation/usairportcode_origin_dest_geo_features.py) [Transformer to augment US airport codes with geolocation info.]
    * [uszipcode_features_database.py](./transformers/augmentation/uszipcode_features_database.py) [Transformer to parse and augment US zipcodes with info from zipcode database.]
    * [uszipcode_features_light.py](./transformers/augmentation/uszipcode_features_light.py) [Lightweight transformer to parse and augment US zipcodes with info from zipcode database.]
  * [DATETIME](./transformers/datetime)
    * [datetime_diff_transformer.py](./transformers/datetime/datetime_diff_transformer.py) [Difference in time between two datetime columns]
    * [datetime_encoder_transformer.py](./transformers/datetime/datetime_encoder_transformer.py) [Converts datetime column into an integer (milliseconds since 1970)]
    * [days_until_dec2020.py](./transformers/datetime/days_until_dec2020.py) [Creates new feature for any date columns, by computing the difference in days between the date value and 31st Dec 2020]
  * [EXECUTABLES](./transformers/executables)
    * [pe_data_directory_features.py](./transformers/executables/pe_data_directory_features.py) [Extract LIEF features from PE files]
    * [pe_exports_features.py](./transformers/executables/pe_exports_features.py) [Extract LIEF features from PE files]
    * [pe_general_features.py](./transformers/executables/pe_general_features.py) [Extract LIEF features from PE files]
    * [pe_header_features.py](./transformers/executables/pe_header_features.py) [Extract LIEF features from PE files]
    * [pe_imports_features.py](./transformers/executables/pe_imports_features.py) [Extract LIEF features from PE files]
    * [pe_normalized_byte_count.py](./transformers/executables/pe_normalized_byte_count.py) [Extract LIEF features from PE files]
    * [pe_section_characteristics.py](./transformers/executables/pe_section_characteristics.py) [Extract LIEF features from PE files]
    * [DATA](./transformers/executables/data)
  * [GENERIC](./transformers/generic)
    * [count_missing_values_transformer.py](./transformers/generic/count_missing_values_transformer.py) [Count of missing values per row]
    * [missing_flag_transformer.py](./transformers/generic/missing_flag_transformer.py) [Returns 1 if a value is missing, or 0 otherwise]
    * [specific_column_transformer.py](./transformers/generic/specific_column_transformer.py) [Example of a transformer that operates on the entire original frame, and hence on any column(s) desired.]
  * [GEOSPATIAL](./transformers/geospatial)
    * [geodesic.py](./transformers/geospatial/geodesic.py) [Calculates the distance in miles between two latitude/longitude points in space]
    * [myhaversine.py](./transformers/geospatial/myhaversine.py) [Computes miles between first two *_latitude and *_longitude named columns in the data set]
  * [HIERARCHICAL](./transformers/hierarchical)
    * [firstNCharCVTE.py](./transformers/hierarchical/firstNCharCVTE.py) [Target-encode high cardinality categorical text by their first few characters in the string ]
    * [log_scale_target_encoding.py](./transformers/hierarchical/log_scale_target_encoding.py) [Target-encode numbers by their logarithm]
  * [IMAGE](./transformers/image)
    * [image_ocr_transformer.py](./transformers/image/image_ocr_transformer.py) [Convert a path to an image to text using OCR based on tesseract]
    * [image_url_transformer.py](./transformers/image/image_url_transformer.py) [Convert a path to an image (JPG/JPEG/PNG) to a vector of class probabilities created by a pretrained ImageNet deeplearning model (Keras, TensorFlow).]
  * [NLP](./transformers/nlp)
    * [continuous_TextTransformer.py](./transformers/nlp/continuous_TextTransformer.py) [please add description]
    * [fuzzy_text_similarity_transformers.py](./transformers/nlp/fuzzy_text_similarity_transformers.py) [Row-by-row similarity between two text columns based on FuzzyWuzzy]
    * [text_char_tfidf_count_transformers.py](./transformers/nlp/text_char_tfidf_count_transformers.py) [Character level TFIDF and Count followed by Truncated SVD on text columns]
    * [text_embedding_similarity_transformers.py](./transformers/nlp/text_embedding_similarity_transformers.py) [Row-by-row similarity between two text columns based on pretrained Deep Learning embedding space]
    * [text_lang_detect_transformer.py](./transformers/nlp/text_lang_detect_transformer.py) [Detect the language for a text value using Google's 'langdetect' package]
    * [text_meta_transformers.py](./transformers/nlp/text_meta_transformers.py) [Extract common meta features from text]
    * [text_named_entities_transformer.py](./transformers/nlp/text_named_entities_transformer.py) [Extract the counts of different named entities in the text (e.g. Person, Organization, Location)]
    * [text_pos_tagging_transformer.py](./transformers/nlp/text_pos_tagging_transformer.py) [Extract the count of nouns, verbs, adjectives and adverbs in the text]
    * [text_preprocessing_transformer.py](./transformers/nlp/text_preprocessing_transformer.py) [Preprocess the text column by stemming, lemmatization and stop word removal]
    * [text_readability_transformers.py](./transformers/nlp/text_readability_transformers.py) [    Custom Recipe to extract Readability features from the text data]
    * [text_sentiment_transformer.py](./transformers/nlp/text_sentiment_transformer.py) [Extract sentiment from text using pretrained models from TextBlob]
    * [text_similarity_transformers.py](./transformers/nlp/text_similarity_transformers.py) [Row-by-row similarity between two text columns based on common N-grams, Jaccard similarity, Dice similarity and edit distance.]
    * [text_spelling_correction_transformers.py](./transformers/nlp/text_spelling_correction_transformers.py) [Correct the spelling of text column]
    * [text_topic_modeling_transformer.py](./transformers/nlp/text_topic_modeling_transformer.py) [Extract topics from text column using LDA]
    * [text_url_summary_transformer.py](./transformers/nlp/text_url_summary_transformer.py) [Extract text from URL and summarizes it]
    * [vader_text_sentiment_transformer.py](./transformers/nlp/vader_text_sentiment_transformer.py) [Extract sentiment from text using lexicon and rule-based sentiment analysis tool called VADER]
  * [NUMERIC](./transformers/numeric)
    * [boxcox_transformer.py](./transformers/numeric/boxcox_transformer.py) [Box-Cox Transform]
    * [count_negative_values_transformer.py](./transformers/numeric/count_negative_values_transformer.py) [Count of negative values per row]
    * [count_positive_values_transformer.py](./transformers/numeric/count_positive_values_transformer.py) [Count of positive values per row]
    * [exp_diff_transformer.py](./transformers/numeric/exp_diff_transformer.py) [Exponentiated difference of two numbers]
    * [log_transformer.py](./transformers/numeric/log_transformer.py) [Converts numbers to their Logarithm]
    * [product.py](./transformers/numeric/product.py) [Products together 3 or more numeric features]
    * [random_transformer.py](./transformers/numeric/random_transformer.py) [Creates random numbers]
    * [round_transformer.py](./transformers/numeric/round_transformer.py) [Rounds numbers to 1, 2 or 3 decimals]
    * [square_root_transformer.py](./transformers/numeric/square_root_transformer.py) [Converts numbers to the square root, preserving the sign of the original numbers]
    * [sum.py](./transformers/numeric/sum.py) [Adds together 3 or more numeric features]
    * [truncated_svd_all.py](./transformers/numeric/truncated_svd_all.py) [Truncated SVD for all columns]
    * [yeojohnson_transformer.py](./transformers/numeric/yeojohnson_transformer.py) [Yeo-Johnson Power Transformer]
  * [OUTLIERS](./transformers/outliers)
    * [h2o3-dl-anomaly.py](./transformers/outliers/h2o3-dl-anomaly.py) [Anomaly score for each row based on reconstruction error of a H2O-3 deep learning autoencoder]
    * [quantile_winsorizer.py](./transformers/outliers/quantile_winsorizer.py) [Winsorizes (truncates) univariate outliers outside of a given quantile threshold]
    * [twosigma_winsorizer.py](./transformers/outliers/twosigma_winsorizer.py) [Winsorizes (truncates) univariate outliers outside of two standard deviations from the mean.]
  * [RECOMMENDATIONS](./transformers/recommendations)
    * [matrixfactorization.py](./transformers/recommendations/matrixfactorization.py) [Collaborative filtering features using various techniques of Matrix Factorization for recommendations.Recommended for large data]
  * [SIGNAL_PROCESSING](./transformers/signal_processing)
    * [signal_processing.py](./transformers/signal_processing/signal_processing.py) [This custom transformer processes signal files to create features used by DriverlessAI to solve a regression problem]
  * [SPEECH](./transformers/speech)
    * [audio_MFCC_transformer.py](./transformers/speech/audio_MFCC_transformer.py) [Extract MFCC and spectrogram features from audio files]
    * [azure_speech_to_text.py](./transformers/speech/azure_speech_to_text.py) [An example of integration with Azure Speech Recognition Service]
  * [STRING](./transformers/string)
    * [simple_grok_parser.py](./transformers/string/simple_grok_parser.py) [Extract column data using grok patterns]
    * [strlen_transformer.py](./transformers/string/strlen_transformer.py) [Returns the string length of categorical values]
    * [to_string_transformer.py](./transformers/string/to_string_transformer.py) [Converts numbers to strings]
    * [user_agent_transformer.py](./transformers/string/user_agent_transformer.py) [A best effort transformer to determine browser device characteristics from a user-agent string]
  * [TARGETENCODING](./transformers/targetencoding)
    * [ExpandingMean.py](./transformers/targetencoding/ExpandingMean.py) [CatBoost-style target encoding. See https://youtu.be/d6UMEmeXB6o?t=818 for short explanation]
    * [leaky_mean_target_encoder.py](./transformers/targetencoding/leaky_mean_target_encoder.py) [Example implementation of a out-of-fold target encoder (leaky, not recommended)]
  * [TIMESERIES](./transformers/timeseries)
    * [auto_arima_forecast.py](./transformers/timeseries/auto_arima_forecast.py) [Auto ARIMA transformer is a time series transformer that predicts target using ARIMA models]
    * [general_time_series_transformer.py](./transformers/timeseries/general_time_series_transformer.py) [Demonstrates the API for custom time-series transformers.]
    * [parallel_auto_arima_forecast.py](./transformers/timeseries/parallel_auto_arima_forecast.py) [Parallel Auto ARIMA transformer is a time series transformer that predicts target using ARIMA models.In this implementation, Time Group Models are fitted in parallel]
    * [parallel_prophet_forecast.py](./transformers/timeseries/parallel_prophet_forecast.py) [Parallel FB Prophet transformer is a time series transformer that predicts target using FBProphet models.]
    * [parallel_prophet_forecast_using_individual_groups.py](./transformers/timeseries/parallel_prophet_forecast_using_individual_groups.py) [Parallel FB Prophet transformer is a time series transformer that predicts target using FBProphet models.This transformer fits one model for each time group column values and is significantly fasterthan the implementation available in parallel_prophet_forecast.py.]
    * [serial_prophet_forecast.py](./transformers/timeseries/serial_prophet_forecast.py) [Transformer that uses FB Prophet for time series prediction.Please see the parallel implementation for more information]
    * [time_encoder_transformer.py](./transformers/timeseries/time_encoder_transformer.py) [converts the Time Column to an ordered integer]
    * [trading_volatility.py](./transformers/timeseries/trading_volatility.py) [Calculates Historical Volatility for numeric features (makes assumptions on the data)]
