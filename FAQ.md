# H2O Driverless AI Bring Your Own Recipes
## FAQ
  #### What are recipes?
  Python code snippets that can be uploaded into Driverless AI at runtime, like plugins. No need to restart Driverless AI.
  #### What exactly are recipes?
  Python code snippets for custom transformers, models and scorers. During training of a supervised machine learning modeling pipeline (we call that experiment), Driverless AI can then use these code snippets as building blocks, in combination with all built-in code pieces (or instead of). By providing your own custom recipes, you can gain control over the optimization choices that Driverless AI makes.
  #### Why do I need to bring my own recipes? Isn't Driverless AI smart enough of the box?
  The only way to find out is to try. Most likely you'll be able to improve performance with custom recipes. Domain knowledge and intuition are essential to getting the best possible performance.
  #### What are some example recipes?
  * Look at the [examples in this repository](https://github.com/h2oai/driverlessai-recipes/blob/master/README.md#sample-recipes). Some illustrative samples:
  * Transformer:
    * Suppose you have a string column that has values like `"A:B:10:5", "A:C:4:10", ...`. It might make sense to split these values by ":" and create four output columns, potentially all numeric, such as `[0,1,10,5], [0,2,4,10], ...` to encode the information more clearly for the algorithm to learn better from.
    * PyTorch deep learning model for [text similarity analysis](https://github.com/h2oai/driverlessai-recipes/blob/master/transformers/nlp/text_embedding_similarity_transformers.py), computes a similary score for any given two text input columns.
    * ARIMA model for [time-series forecasting](https://github.com/h2oai/driverlessai-recipes/blob/master/transformers/timeseries/auto_arima_forecast.py).
    * Data augmentation, such as replacing a zip code with demographic information, or replacing a date column with a [National holiday flag](https://github.com/h2oai/driverlessai-recipes/blob/master/transformers/augmentation/singapore_public_holidays.py).
  * Model:
    * All [H2O-3 Algorithms including H2O AutoML](https://github.com/h2oai/driverlessai-recipes/blob/master/models/algorithms/h2o-3-models.py)
    * Yandex [CatBoost](https://github.com/h2oai/driverlessai-recipes/blob/master/models/algorithms/catboost.py) gradient boosting
    * A custom loss function for [LightGBM](https://github.com/h2oai/driverlessai-recipes/blob/master/models/custom_loss/lightgbm_with_custom_loss.py) or [XGBoost](https://github.com/h2oai/driverlessai-recipes/blob/master/models/custom_loss/xgboost_with_custom_loss.py)
  * Scorer:
    * Maybe you want to optimize your predictions for the [top decile](https://github.com/h2oai/driverlessai-recipes/blob/master/scorers/regression/top_decile.py) for a regression problem..
    * Maybe you care about the [false discovery rate](https://github.com/h2oai/driverlessai-recipes/blob/master/scorers/classification/binary/false_discovery_rate.py) for a binary classification problem.
  #### Driverless is good enough for me, I don't want to do recipes.
  Perfect. Relax and sit back. We'll keep making Driverless AI better and better with every version, so you don't have to.
  Several of the recipes in this repository will likely be included in future releases of Driverless AI out of the box, after more performance improvements and hardening.
  #### What's in it for me if I write a recipe?
  You will get better at doing data science and you will get better results. Writing code is essential to improving your data science skills. Especially when writing data science code. Recipes are perfect for that.
  #### Who can make recipes?
  Anyone who can or wants to. Mostly data scientists or developers. Some of the best recipes are trivial and make a big difference, like custom scorers.
  #### What do I need to make a recipe?
  A text editor. All you need is to create a `.py` text file containing source code.
  #### How do I start?
  * Examine the [references](https://github.com/h2oai/driverlessai-recipes#reference-guide) below for the API specification and architecture diagrams.
  * Look at the [examples in this repository](https://github.com/h2oai/driverlessai-recipes/blob/master/README.md#sample-recipes).
  * Clone this repository and make modifications to existing recipes.
  * Start an experiment and upload the recipe in the expert settings of an experiment.
  #### How do I know whether my recipe works?
  Driverless AI will tell you whether it makes the cut:
  * First, it is subjected to acceptance tests. If it passes, great. If not, Driverless AI provides you with feedback on how to improve it.
  * Then, you can choose to include it in your experiment(s). It will decide which recipes are best suited to solve the problem. At worst, you can cause the experiment to slow down.
  #### How can I debug my recipe?
  * The easiest way (for now) is to keep uploading it to the expert settings in Driverless AI until the recipe is accepted.
  * Another way is to do minimal changes as shown in [this debugging example](./transformers/how_to_debug_transformer.py) and use PyCharm or a similar Python debugger.
  #### What happens if my recipe is rejected during upload?
  * Read the entire error message, it most likely contains the stack trace and helpful information on how to fix the problem.
  * If you can't figure out how to fix the recipe, we suggest you post your questions in the [Driverless AI community Slack channel](https://www.h2o.ai/community/driverless-ai-community/#chat)
  * You can also send us your experiment logs zip file, which will contain the recipe source files.
  #### What happens if my transformer recipe doesn't lead to the highest variable importance for the experiment?
  That's nothing to worry about. It's unlikely that your features have the strongest signal of all features. Even 'magic' Kaggle grandmaster features don't usually make a massive difference, but they still beat most of the competition. 
  #### What happens if my recipe is not used at all by the experiment?
  * Don't give up. You learned something.
  * Check the logs for failures if unsure whether the recipe worked at all or not.
  * Driverless AI will ignore recipe failures unless this robustness feature is specifically disabled. Under Expert Settings, disable `skip_transformer_failures` and `skip_model_failures` if you want to fail the experiment on any unexpected errors due to custom recipes.
  * Inside the experiment logs zip file, there's a folder called `details` and if it contains `.stack` files with stacktraces referring to your custom code, then you know it bombed.
  #### Can I write recipes in Go, C++, Java or R?
  If you can hook it up to Python, then yes. We have many recipes that use Java and C++ backends. Most of Driverless AI uses C++ backends.
  #### Is there a difference between a custom recipe and the recipes shipped with Driverless AI?
  No. Same code base. No performance penalty. No calling overhead. Same inputs and outputs.
  #### Why are some models implemented as transformers?
  Separating of work. With the transformer API, we can replace *only* the particular input column(s) with out-of-fold estimates of the target column. All other columns (features) can be processed by other transformers. The combined union of all features is then passed to the model(s) which can yield higher accuracy than a model that only sees the particular input column(s). For more information about the flow of data, see the technical [references](https://github.com/h2oai/driverlessai-recipes#reference-guide) section.
  #### Isn't it a security risk to run arbitrary Python code? How can I control what recipes are active?
  Recipes are meant to be built by people you trust and each recipe should be code-reviewed before going to production. If you don't want custom code to be executed by Driverless AI, set `enable_custom_recipes=false` in the config.toml, or add the environment variable `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES=0` at startup of Driverless AI. This will disable all custom transformers, models and scorers. If you want to keep all previously uploaded recipes enabled and disable the upload of any new recipes, set `enable_custom_recipes_upload=false` or `DRIVERLESS_AI_ENABLE_CUSTOM_RECIPES_UPLOAD=0` at startup of Driverless AI.
  #### Who can see my recipe?
  Everyone with access to Driverless AI on the particular instance into which the recipe was uploaded. The recipe remains on the instance that runs Driverless AI. Experiment logs may contain relevant information about your recipes, so double-check before you share them.
  #### How do I share my recipe with the world?
  We encourage you to share your recipe in this repository. If your recipe works, please make a pull request and improve the experience for everyone!
    
## References
### Custom Transformers
  * sklearn API
    * Implement `fit_transform(X, y)` for batch transformation of the training data
    * Implement `transform(X)` for row-by-row transformation of validation and test data
  * [Custom Transformer Recipe SDK/API Reference](transformers/transformer_template.py)
  * [Using datatable](https://datatable.readthedocs.io/en/latest/using-datatable.html)
### Custom Models
  * sklearn API
    * Implement `fit(X, y)` for batch training on the training data
    * Implement `predict(X)` for row-by-row predictions on validation and test data
  * [Custom Model Recipe SDK/API Reference](models/model_template.py)
### Custom Scorers
  * sklearn API
    * Implement `score(actual, predicted, sample_weight, labels)` for computing scores and metrics from predictions and actual values
  * [Custom Scorer Recipe SDK/API Reference](scorers/scorer_template.py)
### Technical Architecture Diagram
  * [Direct Link](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.png)
  * [Source File](https://raw.githubusercontent.com/h2oai/driverlessai-recipes/master/reference/DriverlessAI_BYOR.drawio)
  ![BYOR Architecture Diagram](reference/DriverlessAI_BYOR.png)
### Webinar
Webinar: [Extending the H2O Driverless AI Platform with Your Recipes](https://www.brighttalk.com/webcast/16463/360533/extending-the-h2o-driverless-ai-platform-with-your-recipes).

