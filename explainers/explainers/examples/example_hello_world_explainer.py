# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExampleHelloWorldExplainer(CustomExplainer):
    _display_name = "Hello, World!"
    _description = "This is 'Hello, World!' explainer example."
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        explanation = self.create_explanation_workdir_archive(
            display_name=self.display_name, display_category="Demo"
        )

        return [explanation]
