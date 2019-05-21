import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score

class precision(CustomScorer):
  
    
    _description = " Calculates precision: `tp / (tp + fp)`"
    _binary = True    
    _multiclass = True
    _maximize = True
    _perfect_score = 1
    _display_name = "Precision"
    
    
    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
   
        
        if labels is not None:
            actual = LabelEncoder().fit(labels).transform(actual)
        else:
            actual = LabelEncoder().fit_transform(actual)
            
        unique_values=len(np.unique(actual))     
        method="binary"
        if unique_values>2:
           predicted = np.argmax(predicted, axis=1)
           method="micro"
        else :
            predicted=np.array([1 if pr>0.5 else 0 for pr in predicted])


        return  precision_score(actual, predicted, labels=None, average=method, sample_weight=sample_weight)
    


