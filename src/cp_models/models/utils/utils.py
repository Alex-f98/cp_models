
import torch
from torchvision import datasets, transforms

def get_data(source = "mnist", flatten = True, size_calib = 50):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if source == "mnist":
        train_dataset = datasets.MNIST(root='./cp_models/data', train=True , download=True, transform=transform)
        test_dataset  = datasets.MNIST(root='./cp_models/data', train=False, download=True, transform=transform)
    elif source == "fashion":
        train_dataset = datasets.FashionMNIST(root='./cp_models/data', train=True , download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(root='./cp_models/data', train=False, download=True, transform=transform)
    else:
        print("Error")

    X_train, y_train = train_dataset.data.float() / 255.0, train_dataset.targets
    X_test , y_test  = test_dataset.data.float()  / 255.0, test_dataset.targets

    if flatten:
        X_train = X_train.flatten(1)
        X_test = X_test.flatten(1)

    X_calib, X_train = X_train[-size_calib:], X_train[:-size_calib]
    y_calib, y_train = y_train[-size_calib:], y_train[:-size_calib]

    return X_train, y_train, X_test, y_test, X_calib, y_calib
