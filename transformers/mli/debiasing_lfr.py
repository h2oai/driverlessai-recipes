from h2oaicore.transformer_utils import CustomTransformer

import datatable as dt
import numpy as np


class LfrDebiasingTransformer(CustomTransformer):
    _regression = False
    _multiclass = False

    _modules_needed_by_name = ['aif360']

    _display_name = "LrfDebiasingTransformer"

    @staticmethod
    def get_default_properties():
        return dict(
            col_type="all",
            min_cols="all",
            max_cols="all",
            relative_importance=1,
        )

    @staticmethod
    def do_acceptance_test():
        return False

    def fit(self, X: dt.Frame, y: np.array = None):
        from h2oaicore.systemutils import config
        from aif360.datasets import BinaryLabelDataset
        from aif360.algorithms.preprocessing.lfr import LFR

        if y is not None:
            if 'recipe_dict' in config:
                config = config['recipe_dict']

            # LFR supports only numerical columns
            # But categoricals which are numeric are ok so setting col_type="all"
            if any(unsupported in str(X.ltypes) for unsupported in ['str', 'obj']):
                return

            X_pd = X.to_pandas()
            X = dt.Frame(X_pd.fillna(X_pd.mean()))

            frame = dt.cbind(X, dt.Frame(y))
            self.label_names = [frame.names[-1]]

            self.privileged_groups = config['privileged_groups']
            self.unprivileged_groups = config['unprivileged_groups']
            self.favorable_label = float(config['favorable_label'])
            self.unfavorable_label = float(config['unfavorable_label'])
            self.protected_attribute_names = config['protected_attribute_names']

            self.lfr = LFR(
                unprivileged_groups=self.unprivileged_groups,
                privileged_groups=self.privileged_groups,
                verbose=0,
            )

            self.lfr.fit(
                BinaryLabelDataset(
                    df=frame.to_pandas(),
                    favorable_label=self.favorable_label,
                    unfavorable_label=self.unfavorable_label,
                    label_names=self.label_names,
                    protected_attribute_names=self.protected_attribute_names,
                )
            )
            self.fitted = True

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X: dt.Frame, y: np.array = None):
        from aif360.datasets import BinaryLabelDataset
        # Transformation should only occur during training when y is present
        if self.fitted and (self.label_names in X.names or y is not None):
            if self.label_names not in X.names:
                X = dt.cbind(X, dt.Frame(y))

            X_pd = X.to_pandas()
            X = dt.Frame(X_pd.fillna(X_pd.mean()))
            transformed_X: BinaryLabelDataset = self.lfr.transform(
                BinaryLabelDataset(
                    df=X.to_pandas(),
                    favorable_label=self.favorable_label,
                    unfavorable_label=self.unfavorable_label,
                    label_names=self.label_names,
                    protected_attribute_names=self.protected_attribute_names,
                )
            )

            return dt.Frame(
                transformed_X.features,
                names=[name+"_lfr" for name in transformed_X.feature_names],
            )
        # For predictions no transformation is required
        else:
            return X
