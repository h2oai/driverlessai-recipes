"""Test individual 1"""
from h2oaicore.ga import CustomIndividual


class Indivtestinternal1(CustomIndividual):

    def set_model(self):
        self.model_display_name = 'XGBoostGBM'

        self.model_params = {'base_score': 0.5,
                             'booster': 'gbtree',
                             'colsample_bytree': 0.8,
                             'early_stopping_rounds': 1,
                             'eval_metric': 'auc',
                             'gamma': 0.0,
                             'grow_policy': 'depthwise',
                             'importance_type': 'gain',
                             'learning_rate': 1.0,
                             'max_bin': 64,
                             'max_delta_step': 0.0,
                             'max_depth': 6,
                             'max_leaves': 64,
                             'min_child_weight': 1,
                             'n_estimators': 3,
                             'num_class': 1,
                             'objective': 'binary:logistic',
                             'reg_alpha': 0.0,
                             'reg_lambda': 0.0,
                             'scale_pos_weight': 1.0,
                             'seed': 159699529,
                             'silent': 1,
                             'subsample': 0.7,
                             'tree_method': 'hist'}

    def set_genes(self):
        params = {'num_cols': ['PAY_0'], 'random_state': 159699540}
        self.add_transformer('OriginalTransformer', **params)
