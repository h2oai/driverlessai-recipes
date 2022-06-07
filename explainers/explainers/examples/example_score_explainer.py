# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExampleScoreExplainer(CustomExplainer, CustomDaiExplainer):
    _display_name = "Example Score Explainer"
    _description = (
        "This is explainer example which demonstrates how to get model predict "
        "method and use it to score dataset."
    )
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

    def setup(self, model, persistence, **e_params):
        CustomExplainer.setup(self, model, persistence, **e_params)
        CustomDaiExplainer.setup(self, **e_params)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        # prepare 1st row of the dataset with features used by the model
        df = X[:1, self.used_features]
        self.logger.info(f"Dataset to score: {df}")

        # model predict method
        prediction = self.model.predict_method(df)
        self.logger.info(f"Prediction     : {prediction}")

        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]
