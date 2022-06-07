# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import CustomExplanation
from h2oaicore.mli.oss.byor.core.representations import (
    TextCustomExplanationFormat,
)


class MyCustomExplanation(CustomExplanation):
    """Example of a user defined explanation type."""

    _explanation_type = "user-guide-explanation-example"
    _is_global = True

    def __init__(
            self, explainer, display_name: str = None, display_category: str = None
    ) -> None:
        CustomExplanation.__init__(
            self,
            explainer=explainer,
            display_name=display_name,
            display_category=display_category,
        )

    def validate(self) -> bool:
        return self._formats is not None


class ExampleCustomExplanationExplainer(CustomExplainer, CustomDaiExplainer):
    _display_name = "Example Custom Explanation Explainer"
    _description = (
        "Explainer example which shows how to define custom explanation."
    )
    _regression = True
    _explanation_types = [TextCustomExplanationFormat]

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

    def setup(self, model, persistence, **e_params):
        CustomExplainer.setup(self, model, persistence, **e_params)
        CustomDaiExplainer.setup(self, **e_params)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        df = X[:1, self.used_features]
        prediction = self.model.predict_method(df)

        # create CUSTOM explanation
        explanation = MyCustomExplanation(
            explainer=self,
            display_name="Custom Explanation Example",
            display_category="Example",
        )
        # add a text format to CUSTOM explanation
        explanation.add_format(
            TextCustomExplanationFormat(
                explanation=explanation,
                format_data=f"Prediction is: {prediction}",
                format_file=None,
            )
        )

        return [explanation]
