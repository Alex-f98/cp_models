import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class AbsoluteScore:
    """ |y - y_pred| """

    def __call__(self, y, y_pred):
        return np.abs(y - y_pred)


class SquaredScore:
    """ (y - y_pred)^2 """

    def __call__(self, y, y_pred):
        return (y - y_pred) ** 2


class SplitConformalPredictor(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Unified Conformal Prediction for Classification and Regression
    
    Args:
        model: Base model with fit/predict methods
        task_type: 'clf' for classification, 'reg' for regression
        alpha: Significance level (1 - confidence level)
        score: Score function for regression
    """
    
    def __init__(self, model, task_type='clf', alpha=0.1, score=None):
        self.model = model
        self.task_type = task_type
        self.alpha = alpha
        self.score = score if score else AbsoluteScore()
        self.q_hat = None
        self.classes_ = None
        
    def fit(self, X_train, y_train):
        """Train base model"""
        self.model.fit(X_train, y_train)
        
        # Store classes for classification
        if self.task_type == 'clf':
            self.classes_ = np.unique(y_train)
            
        return self
    
    def calibrate(self, X_cal, y_cal):
        """Compute conformity scores"""
        
        if self.task_type == 'reg':
            # Regression: use absolute or squared error
            y_pred = self.model.predict(X_cal)
            scores = self.score(y_cal, y_pred)
            
            n = len(scores)
            q_index = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
            self.q_hat = np.sort(scores)[q_index]
            
        elif self.task_type == 'clf':
            assert hasattr(self.model, 'predict_proba'), "Model must have predict_proba method for classification"
            # Classification: use conformity scores

            # Use probability-based conformity scores
            y_proba = self.model.predict_proba(X_cal) #[len(y_cal), n_classes]
            # scores = 1 - p(y_true | x)
            scores = np.array(1 - y_proba[np.arange(len(y_cal)), y_cal.numpy()]) #[len(y_cal)]
            
            n = len(scores)
            q_val = np.ceil((1 - self.alpha) * (len(scores) + 1)) / len(scores)
            self.q_hat = np.quantile(scores, q_val, method="higher")
        
        return self
    
    def predict(self, X):
        """Make point predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities (classification only)"""
        if self.task_type != 'clf':
            raise ValueError("predict_proba only available for classification")
            
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise ValueError("Base model does not support predict_proba")
    
    def predict_set(self, X):
        """Predict conformal sets"""
        if self.task_type == 'clf':
            return self._predict_classification_set(X)
        elif self.task_type == 'reg':
            return self._predict_regression_interval(X)
        else:
            raise ValueError("task_type must be 'clf' or 'reg'")
    
    def _predict_classification_set(self, X):
        """Predict conformal classification sets"""
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Classification conformal sets require predict_proba")
            
        y_proba = self.model.predict_proba(X)
        prediction_sets = []
        
        for i in range(len(X)):
            # Include classes where probability >= threshold
            threshold = 1 - self.q_hat
            valid_classes = y_proba[i] >= threshold
            prediction_sets.append(self.classes_[valid_classes])
            
        return prediction_sets
    
    def _predict_regression_interval(self, X):
        """Predict conformal regression intervals"""
        y_pred = self.model.predict(X)
        
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        
        return np.column_stack([lower, upper])
    
    def predict_interval(self, X):
        """Alias for predict_set"""
        return self.predict_set(X)


# Backward compatibility
class SplitConformalClassifier(SplitConformalPredictor):
    """Backward compatibility wrapper"""
    
    def __init__(self, model, alpha=0.1, score=None):
        super().__init__(model, task_type='clf', alpha=alpha, score=score)


class SplitConformalRegressor(SplitConformalPredictor):
    """Backward compatibility wrapper"""
    
    def __init__(self, model, alpha=0.1, score=None):
        super().__init__(model, task_type='reg', alpha=alpha, score=score)