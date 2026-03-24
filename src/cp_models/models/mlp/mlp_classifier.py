import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from ..utils.train import train_model
from ..utils.predict import predict, predict_proba



class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for classification"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=10, dropout=0.2):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)



class GenericMLP(BaseEstimator, ClassifierMixin):
    """MLP wrapper with sklearn-like interface"""
    
    def __init__(self, input_dim=None, num_classes=10, epochs=10, batch_size=32, 
                 learning_rate=0.001, device='cpu'):
        self.input_dim = input_dim
        self.num_classes   = num_classes
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.device        = device
        self.model         = None
        self.classes_      = None
        
    def _create_data_loader(self, X, y, shuffle=True):
        """Convert numpy arrays to PyTorch DataLoader"""
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.LongTensor(y)
            
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Train the MLP model"""
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)
        
        # Initialize model
        self.model = MLPClassifier(input_dim=self.input_dim, num_classes=self.num_classes)
        
        # Create data loaders
        train_loader = self._create_data_loader(X, y, shuffle=True)
        
        if X_val is not None and y_val is not None:
            val_loader = self._create_data_loader(X_val, y_val, shuffle=False)
        else:
            val_loader = None
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Train model
        self.model = train_model(
            self.model, train_loader, val_loader, 
            criterion, optimizer, 
            epochs=self.epochs, 
            device=self.device
        )
        
        return self
    
    def predict(self, X):
        """Generate class predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Convert to DataLoader if needed
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        # Create dummy dataset for prediction
        dummy_y = torch.zeros(X.shape[0])
        dataset = TensorDataset(X, dummy_y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        predictions, _ = predict(self.model, loader, device=self.device)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """Generate probability predictions"""
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
            
        # Convert to DataLoader if needed
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        # Create dummy dataset for prediction
        dummy_y = torch.zeros(X.shape[0])
        dataset = TensorDataset(X, dummy_y)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        probabilities = predict_proba(self.model, loader, device=self.device)
        return np.array(probabilities)

