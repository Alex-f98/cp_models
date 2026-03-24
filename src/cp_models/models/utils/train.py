"""Training utilities for neural network models"""

import torch


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
    """Generic training function for PyTorch models"""
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss   = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        if val_loader:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
        
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            
            accuracy = 100. * correct / total
            print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
                  f'Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.2f}%')
        else:
            print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}')
    
    return model
