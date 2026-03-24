"""Prediction utilities for neural network models"""
import torch

def predict(model, data_loader, device='cpu'):
    """Generate predictions for a PyTorch model"""
    model.eval()
    model.to(device)
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
    
    return predictions, true_labels


def predict_proba(model, data_loader, device='cpu'):
    """Generate probability predictions for a PyTorch model"""
    model.eval()
    model.to(device)
    
    probabilities = []
    
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            probabilities.extend(probs.cpu().numpy())
    
    return probabilities
