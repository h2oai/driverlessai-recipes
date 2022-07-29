"""Example of how to define MLI explainer parameters."""
# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
    CustomExplainerParam,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation
from h2oaicore.mli.oss.byor.explainer_utils import CustomExplainerArgs
from h2oaicore.mli.oss.commons import ExplainerParamType


class ExampleParamsExplainer(CustomExplainer, CustomDaiExplainer):
    PARAM_ROWS_TO_SCORE = "rows_to_score"

    _display_name = "Example Params Explainer"
    _description = "This explainer example shows how to define explainer parameters."
    _regression = True
    _parameters = [
        CustomExplainerParam(
            param_name=PARAM_ROWS_TO_SCORE,
            description="The number of dataset rows to be scored by explainer.",
            param_type=ExplainerParamType.int,
            default_value=1,
            src=CustomExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
    ]
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

        self.args = None

    def setup(self, model, persistence, **e_params):
        CustomExplainer.setup(self, model, persistence, **e_params)
        CustomDaiExplainer.setup(self, **e_params)

        # resolve explainer parameters to instance attributes
        self.args = CustomExplainerArgs(ExampleParamsExplainer._parameters)
        self.args.resolve_params(
            explainer_params=CustomExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            )
        )

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        # use parameter
        rows = self.args.get(self.PARAM_ROWS_TO_SCORE)

        df = X[:rows, self.used_features]
        prediction = self.model.predict_method(df)
        self.logger.info(f"Predictions of dataset with shape {df.shape}: {prediction}")
        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]
