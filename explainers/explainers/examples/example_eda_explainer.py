# Copyright 2017-2021 H2O.ai, Inc. All rights reserved.
from h2oaicore.mli.oss.byor.core.explainers import (
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import WorkDirArchiveExplanation


class ExampleEdaExplainer(CustomExplainer):

    _display_name = "Example Dataset Explainer"
    _description = "This is Explanatory Data Analysis explainer example."
    _regression = True
    _explanation_types = [WorkDirArchiveExplanation]

    def __init__(self):
        CustomExplainer.__init__(self)

    def setup(self, model, persistence, **kwargs):
        CustomExplainer.setup(self, model, persistence, **kwargs)

    def explain(self, X, y=None, explanations_types=None, **kwargs) -> list:
        self.logger.debug("explain() method invoked with dataset:")
        self.logger.debug(f"  type:    {type(X)}")
        self.logger.debug(f"  shape:   {X.shape}")
        self.logger.debug(f"  columns: {X.names}")
        self.logger.debug(f"  types:   {X.stypes}")
        self.logger.debug(f"  unique:  {X.nunique()}")
        self.logger.debug(f"  max:     {X.max()}")
        self.logger.debug(f"  min:     {X.min()}")

        return [
            self.create_explanation_workdir_archive(
                display_name=self.display_name, display_category="Demo"
            )
        ]
