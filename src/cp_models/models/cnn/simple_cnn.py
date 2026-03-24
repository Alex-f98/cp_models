import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.utils.data import DataLoader, TensorDataset
from ..utils.train import train_model
from ..utils.predict import predict, predict_proba


class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for image classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# CNN (Convolutional Neural Network) class
class CNN(nn.Module):
    """
    CNN for image classification
    
    Args:
        num_classes (int): Number of classes for classification
    """
    def _auto_reshape(self, x):
        """Detectar y reformatear para capas convolucionales"""
        dim = x.shape
        if len(dim) == 2:
            raise ValueError(f"Expected 2D input, got {len(dim)}D, shape: {dim}")
        
        if len(dim) == 3:  # [N, H, W]
            return x.unsqueeze(1)  # -> [N, 1, H, W]
        else:
            return x  # Ya está bien

    def __init__(self, input_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels , 32, 3, padding=1)  # Input has 3 channels (e.g., RGB images)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)          # Max pooling layer to downsample spatial dimensions
        self.fc1  = nn.Linear(128 * 3 * 3, 64)  # Fully connected layer after the convolutional layers
        self.fc2  = nn.Linear(64, 10)           # Output layer for classification (10 classes assumed)

    def forward(self, x):
        x = self._auto_reshape(x)
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))

        # Flatten the output from convolutional layers for the fully connected layers
        x = x.reshape(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)  # Final output for classification
        return x



class GenericCNN(BaseEstimator, ClassifierMixin):
    """CNN wrapper with sklearn-like interface"""
    
    def __init__(self, input_channels=None, num_classes=10, epochs=10, batch_size=32, 
                 learning_rate=0.001, device='cpu'):
        self.input_channels = input_channels
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
        """Train the CNN model"""
        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)
        
        # Initialize model
        self.model = CNN(input_channels=self.input_channels, num_classes=self.num_classes)
        
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

