
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
class AEC(BaseEstimator,ClassifierMixin):
    def __init__(self,AE,X=None,y=None) :
        self.model=AE

    def fit(self, X, y):
        self.model.train(X, y)
        return self.model
    
    def predict(self, X):
        threshold = 0.0049
        predictedRMSE=self.model.execute(X)
        precitedValue = 1 if predictedRMSE > threshold else 0
        return precitedValue
    
        
    def predict_proba(self, X):
        return self.model.execute(X)
        