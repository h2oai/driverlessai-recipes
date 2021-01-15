"""Morris Sensitivity Analysis Explainer"""

from functools import partial

import datatable as dt
import numpy as np
import pandas as pd

from h2oaicore.mli.oss.byor.core.explainers import (
    CustomDaiExplainer,
    CustomExplainer,
)
from h2oaicore.mli.oss.byor.core.explanations import GlobalFeatImpExplanation
from h2oaicore.mli.oss.byor.core.representations import (
    GlobalFeatImpJSonDatatableFormat,
    GlobalFeatImpJSonFormat,
)
from h2oaicore.mli.oss.byor.explainer_utils import clean_dataset


# Explainer MUST extend abstract CustomExplainer class to be discovered and
# deployed. In addition it inherits common metadata and (default) functionality. The
# explainer must implement fit() and explain() methods.
#
# Explainer CAN extend CustomDaiExplainer class if it will run on Driverless AI server
# and use experiments. CustomDaiExplainer class provides easy access/handle to the
# dataset and model (metadata and artifacts), filesystem, ... and common logic.
class MorrisSensitivityLeExplainer(CustomExplainer, CustomDaiExplainer):
    """InterpretML: Morris sensitivity (https://github.com/interpretml/interpret)"""

    # explainer display name (used e.g. in UI explainer listing)
    _display_name = "Morris Sensitivity Analysis"
    # declaration of supported experiments: regression / binary / multiclass
    _regression = True
    _binary = True
    # declaration of provided explanations: global, local or both
    _global_explanation = True
    # declaration of explanation types this explainer creates e.g. feature importance
    _explanation_types = [GlobalFeatImpExplanation]
    # Python package dependencies (can be installed using pip)
    _modules_needed_by_name = ["interpret==0.1.20"]

    # explainer constructor must not have any required parameters
    def __init__(self):
        CustomExplainer.__init__(self)
        CustomDaiExplainer.__init__(self)
        self.cat_variables = None
        self.mcle = None

    # setup() method is used to initialize the explainer based on provided parameters
    # which are passed from client/UI. See parent classes setup() methods docstrings
    # and source to check the list of instance fields which are initialized for the
    # explainer
    def setup(self, model, persistence, key=None, params=None, **e_params):
        CustomExplainer.setup(self, model, persistence, key, params, **e_params)
        CustomDaiExplainer.setup(self, **e_params)

    # abstract fit() method must be implemented - its purpose is to pre-compute
    # any artifacts e.g. surrogate models, to be used by explain() method
    def fit(self, X: dt.Frame, y: dt.Frame = None, **kwargs):
        # nothing to pre-compute
        return self

    # explain() method is responsible for the creation of the explanations
    def explain(
        self, X, y=None, explanations_types: list = None, **kwargs
    ) -> list:
        # 3rd party Morris SA library import
        from interpret.blackbox import MorrisSensitivity

        # DATASET: categorical features encoding (for 3rd party libraries which
        # support numeric features only), rows w/ missing values filtering, ...
        X = X[:, self.used_features] if self.used_features else X
        x, self.cat_variables, self.mcle, _ = clean_dataset(
            frame=X.to_pandas(),
            le_map_file=self.persistence.get_explainer_working_file("mcle"),
            logger=self.logger,
        )

        # PREDICT FUNCTION: Driverless AI scorer -> library compliant predict function
        def predict_function(
            pred_fn, col_names, cat_variables, label_encoder, X
        ):
            X = pd.DataFrame(X.tolist(), columns=col_names)

            # categorical features inverse label encoding used in case of 3rd party
            # libraries which support numeric only
            if label_encoder:
                X[cat_variables] = X[cat_variables].astype(np.int64)
                label_encoder.inverse_transform(X)

            # score
            preds = pred_fn(X)

            # scoring output conversion to the format expected by 3rd party library
            if isinstance(preds, pd.core.frame.DataFrame):
                preds = preds.to_numpy()
            if preds.ndim == 2:
                preds = preds.flatten()
            return preds

        predict_fn = partial(
            predict_function,
            self.model.predict_method,
            self.used_features,
            self.cat_variables,
            self.mcle,
        )

        # CALCULATION of the Morris SA explanation
        sensitivity: MorrisSensitivity = MorrisSensitivity(
            predict_fn=predict_fn, data=x, feature_names=list(x.columns)
        )
        morris_explanation = sensitivity.explain_global(name=self.display_name)

        # NORMALIZATION of proprietary Morris SA library data to explanation w/
        # Grammar of MLI format for the visualization in Driverless AI UI
        explanations = [self._normalize_to_gom(morris_explanation)]

        # explainer MUST return declared explanation(s) (_explanation_types)
        return explanations

    #
    # optional NORMALIZATION to Grammar of MLI
    #
    """
        explainer_morris_sensitivity_explainer_..._MorrisSensitivityExplainer_<UUID>
        ├── global_feature_importance
        │   ├── application_json
        │   │   ├── explanation.json
        │   │   └── feature_importance_class_0.json
        │   └── application_vnd_h2oai_json_datatable_jay
        │       ├── explanation.json
        │       └── feature_importance_class_0.jay
        ├── log
        │   ├── explainer_job.log
        │   └── logger.lock
        └── work
    """

    # Normalization of the data to the Grammar of MLI defined format. Normalized data
    # can be visualized using Grammar of MLI UI components in Driverless AI web UI.
    #
    # This method creates explanation (data) and its representations (JSon, datatable)
    def _normalize_to_gom(self, morris_explanation) -> GlobalFeatImpExplanation:
        # EXPLANATION
        explanation = GlobalFeatImpExplanation(
            explainer=self,
            # display name of explanation's tile in UI
            display_name=self.display_name,
            # tab name where to put explanation's tile in UI
            display_category=GlobalFeatImpExplanation.DISPLAY_CAT_CUSTOM,
        )

        # FORMAT: explanation representation as JSon+datatable (JSon index file which
        # references datatable frame for each class)
        jdf = GlobalFeatImpJSonDatatableFormat
        # data normalization: 3rd party frame to Grammar of MLI defined frame
        # conversion - see GlobalFeatImpJSonDatatableFormat docstring for format
        # documentation and source for helpers to create the representation easily
        explanation_frame = dt.Frame(
            {
                jdf.COL_NAME: morris_explanation.data()["names"],
                jdf.COL_IMPORTANCE: list(morris_explanation.data()["scores"]),
                jdf.COL_GLOBAL_SCOPE: [True]
                * len(morris_explanation.data()["scores"]),
            }
        ).sort(-dt.f[jdf.COL_IMPORTANCE])
        # index file (of per-class data files)
        (
            idx_dict,
            idx_str,
        ) = GlobalFeatImpJSonDatatableFormat.serialize_index_file(["global"])
        json_dt_format = GlobalFeatImpJSonDatatableFormat(explanation, idx_str)
        json_dt_format.update_index_file(
            idx_dict, total_rows=explanation_frame.shape[0]
        )
        # data file
        json_dt_format.add_data_frame(
            format_data=explanation_frame,
            file_name=idx_dict[jdf.KEY_FILES]["global"],
        )
        # JSon+datatable format can be added as explanation's representation
        explanation.add_format(json_dt_format)

        # FORMAT: explanation representation as JSon
        #
        # Having JSon+datatable formats it's easy to get other formats like CSV,
        # datatable, ZIP, ... using helpers - adding JSon representation:
        explanation.add_format(
            explanation_format=GlobalFeatImpJSonFormat.from_json_datatable(
                json_dt_format
            )
        )

        return explanation
