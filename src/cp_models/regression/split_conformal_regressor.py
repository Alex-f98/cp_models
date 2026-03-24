import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class SplitConformalRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, model, alpha=0.1, score=None):
        self.model = model
        self.alpha = alpha
        self.score = score
        self.q_hat = None

    def fit(self, X_train, y_train):
        """Train base model"""
        self.model.fit(X_train, y_train)
        return self

    def calibrate(self, X_cal, y_cal):
        """Compute conformity scores"""
        y_pred = self.model.predict(X_cal)
        
        if self.score:
            scores = self.score(y_cal, y_pred)
        else:
            scores = np.abs(y_cal - y_pred)

        n = len(scores)
        q_index = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        self.q_hat = np.sort(scores)[q_index]

        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_interval(self, X):
        preds = self.predict(X)
        lower = preds - self.q_hat
        upper = preds + self.q_hat
        return lower, upper
