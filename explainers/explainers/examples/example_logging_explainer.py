# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExampleLoggingExplainer(CustomExplainer):
    _display_name = "Example Logging Explainer"
    _description = "This is logging explainer example."
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)

    def setup(self, model, persistence, **kwargs):
        CustomExplainer.setup(self, model, persistence, **kwargs)

        self.logger.info(f"{self.display_name} explainer initialized")

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        self.logger.debug(f"explain() method invoked with args: {kwargs}")

        if not explanations_types:
            self.logger.warning(
                f"Explanation types to be returned by {self.display_name} not specified"
            )

        try:
            return [
                self.create_explanation_workdir_archive(
                    display_name=self.display_name, display_category="Demo"
                )
            ]
        except Exception as ex:
            self.logger.error(
                f"Explainer '{ExampleLoggingExplainer.__name__}' failed with: {ex}"
            )
            raise ex
