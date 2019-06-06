import typing
import numpy as np
from h2oaicore.metrics import CustomScorer
from sklearn.preprocessing import LabelEncoder

class MyHuberLossScorer(CustomScorer):
    _delta = 1.
    _description = "My Huber Loss for Binary Classification [delta=%f]." % _delta
    _binary = True
    _maximize = False
    _perfect_score = 0
    _display_name = "Huber"
    
        
    def score(self,
              actual: np.array,
              predicted: np.array,
              sample_weight: typing.Optional[np.array] = None,
              labels: typing.Optional[np.array] = None) -> float:
        
        lb = LabelEncoder()
        labels = lb.fit_transform(labels)
        actual = lb.transform(actual)
         
        if sample_weight is None:
            sample_weight = np.ones(actual.shape[0])
         
        delta = self.__class__._delta
        if delta < 0:
             delta = 0
             
        abs_error = np.abs(np.subtract(actual,predicted))
        loss = np.where(abs_error < delta, .5*(abs_error)**2, delta*(abs_error-0.5*delta))
         
        return np.sum(loss)
         