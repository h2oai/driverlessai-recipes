"""MAE with a penalty that differs for positive and negative errors"""
import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.metrics import mean_absolute_error


class CostMeanAbsoluteError(CustomScorer):
    _description = "MAE function with a penalty that differs for positive and negative errors"
    _regression = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Asymmetric MAE"
    _supports_sample_weight = False

    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        
        
        # Specify the per unit cost of over or underestimating
        per_unit_cost_of_underestimating = 2.0
        per_unit_cost_of_overestimating = 0.5
        
        cost_function = np.abs(actual - predicted)
        
        cost_function[predicted > actual] = cost_function[predicted > actual] * per_unit_cost_of_overestimating 
        cost_function[predicted < actual] = cost_function[predicted < actual] * per_unit_cost_of_underestimating     
        
        mean_cost = sum(cost_function) / len(cost_function)
        
        return mean_cost
