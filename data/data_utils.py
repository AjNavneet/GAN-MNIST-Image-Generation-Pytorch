# Importing required libraries
import torch

# Import MNIST dataset from torchvision.datasets
from torchvision.datasets import MNIST 

# Import DataLoader from torch.utils.data to load the data
from torch.utils.data import DataLoader

# Import transforms from torchvision to transform the data
from torchvision import transforms

# Function to transform and load the data using Torch
def get_dl(batchsize):
    # Define a transformation to convert the data into Tensors
    train_transforms = transforms.Compose([transforms.ToTensor()])
    
    # Download the MNIST train and test datasets and transform them into Tensors
    train_data = MNIST(root="./train.", train=True, download=True, transform=train_transforms)
    test_data = MNIST(root="./test.", train=True, download=True, transform=train_transforms)
    
    # Create DataLoader objects to efficiently load the training and test data in batches
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False, drop_last=True)
    
    # Return the DataLoader objects containing the train and test data
    return train_loader, test_loader
