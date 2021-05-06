"""Accumulated Local Effects (ALE) explainer

Note:
This example repurposes the Partial Dependence format render data. As such, the label
"Average Prediction of {response}" is used for the y-axis instead of 
"ALE of {response}".
"""

import datatable as dt
import json
import pandas as pd

from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
    CustomExplainerParam,
)
from h2oaicore.mli.oss.byor.core.explanations import PartialDependenceExplanation
from h2oaicore.mli.oss.byor.core.representations import PartialDependenceJSonFormat
from h2oaicore.mli.oss.byor.explainer_utils import CustomExplainerArgs
from h2oaicore.mli.oss.commons import ExplainerParamType


class ALEExplainer(CustomExplainer, CustomDaiExplainer):
    """Accumulated Local Effects (ALE) explainer https://github.com/blent-ai/ALEPython)
    """

    _display_name = "Accumulated Local Effects"
    _description = (
        'Accumulated Local Effects (ALE) main effects for quantitative features - '
        'Apley, Daniel W. "Visualizing the effects of predictor variables in black '
        'box supervised learning models."'
    )
    _regression = True
    _binary = True
    _multiclass = False
    _global_explanation = True
    _local_explanation = False
    _explanation_types = [PartialDependenceExplanation]
    _modules_needed_by_name = ["git+https://github.com/blent-ai/ALEPython"]
    _parameters = [
        CustomExplainerParam(
            param_name="bins",
            description=(
                "Maximum number of bins to use if not specified for feature."
            ),
            param_type=ExplainerParamType.int,
            default_value=10,
            value_min=1,
            src=CustomExplainerParam.SRC_EXPLAINER_PARAMS,
        ),
        CustomExplainerParam(
            param_name="feature_bins",
            description=(
                "Mapping of feature name to maximum number of bins."
            ),
            param_type=ExplainerParamType.dict,
            default_value={},
            src=CustomExplainerParam.SRC_EXPLAINER_PARAMS,
        )
    ]
    ALE_CLASS = "None"

    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)

    def setup(self, model, persistence, key = None, params = None, **explainer_params):
        CustomExplainer.setup(self, model, persistence, key, params, **explainer_params)
        CustomDaiExplainer.setup(self, **explainer_params)
        self.args = CustomExplainerArgs(ALEExplainer._parameters)
        self.args.resolve_params(
            explainer_params=CustomExplainerArgs.json_str_to_dict(
                self.explainer_params_as_str
            )
        )
        self.cfg_bins = self.args.get("bins")
        self.cfg_feature_bins = json.loads(str(self.args.get("feature_bins")))

    def fit(self, X: dt.Frame, y: dt.Frame = None, **kwargs):
        # nothing to pre-compute
        return self

    def explain(self, X, y=None, explanations_types: list = None, **kwargs):
        X = X[:, self.used_features] if self.used_features else X

        # CALCULATION
        ale_per_feature = self._do_ale_per_feature(X, features=self.used_features)

        # NORMALIZATION: convert ALE's output to Grammar of MLI JSon format
        explanations = [self._ale_to_gom(ale_per_feature)]

        return explanations

    def _do_ale(self, X, feature, bins):
        """Calculates ALE function of specified features based on training set.

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            Training set on which model was trained.
        feature : str
            Feature for which to plot ALE.
        bins : int
            Number of bins used to split feature's space.
        """
        import alepython.ale as ale_impl
        ale, quantiles = ale_impl._first_order_ale_quant(
            self.model.predict, X, feature, bins
        )
        quantile_centers = ale_impl._get_centres(quantiles)
        if len(quantile_centers) < 2:
            raise RuntimeError("Not enough quantile centers to plot.")
        return pd.DataFrame({"ale": ale, "quantile_center": quantile_centers})

    def _do_ale_per_feature(self, X: dt.Frame, features: list) -> dict:
        ale_per_feature = dict()
        for feature in features:
            try:
                ale_per_feature[feature] = self._do_ale(
                    X=X.to_pandas(),
                    feature=feature,
                    bins=self.cfg_feature_bins.get(feature, self.cfg_bins),
                )
            except Exception as ex:
                self.logger.warning(
                    f"ALE: skipping feature {feature}"
                )
                self.logger.debug(
                    f"ALE: skipping feature {feature} as it failed with: {ex}"
                )
        return ale_per_feature

    def _ale_to_gom(self, ale_per_feature: dict) -> PartialDependenceExplanation:
        """ALE to Grammar of MLI JSon PD representation."""
        ale_explanation = PartialDependenceExplanation(
            explainer=self, 
            display_name=self._display_name, 
            display_category=PartialDependenceExplanation.DISPLAY_CAT_DAI_MODEL
        )
        index_dict, index_str = PartialDependenceJSonFormat.serialize_index_file(
            features=list(ale_per_feature.keys()), 
            classes=[self.ALE_CLASS], 
            doc=self._description
        )
        json_format = PartialDependenceJSonFormat(
            explanation=ale_explanation, json_data=index_str
        )
        # JSON: data files: per-feature/per-class (see PartialDependenceJSonFormat doc)
        for feature in ale_per_feature:
            json_format.add_data(
                format_data=self._ale_to_json(ale_per_feature[feature]),
                file_name=index_dict[
                    PartialDependenceJSonFormat.KEY_FEATURES
                ][
                    feature
                ][
                    PartialDependenceJSonFormat.KEY_FILES
                ][
                    ALEExplainer.ALE_CLASS
                ]
            )
        ale_explanation.add_format(json_format)
        return ale_explanation

    @staticmethod
    def _ale_to_json(frame: pd.DataFrame) -> str:
        """ALE frame to explainer JSON data file - see
        PartialDependenceJSonFormat documentation.
        """
        data = list()
        for i in range(frame.shape[0]):
            data.append(
                {
                    PartialDependenceJSonFormat.KEY_BIN: frame.loc[
                        i, "quantile_center"
                    ],
                    PartialDependenceJSonFormat.KEY_PD: frame.loc[i, "ale"],
                    PartialDependenceJSonFormat.KEY_SD: None,
                    PartialDependenceJSonFormat.KEY_OOR: False
                }
            )
        json_str = json.dumps({PartialDependenceJSonFormat.KEY_DATA: data})
        return json_str
