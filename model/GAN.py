import torch
import torch.nn as nn
import numpy as np

# Generator network
class Generator(nn.Module):
    def __init__(self, batch_size, input_dim):
        super().__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)  # Fully connected layer 1
        self.LRelu = nn.LeakyReLU()  # Leaky ReLU activation function
        self.fc2 = nn.Linear(128, 1 * 28 * 28)  # Fully connected layer 2
        self.tanH = nn.Tanh()  # Hyperbolic Tangent activation function

    # Function for forward propagation
    def forward(self, x):
        layer1 = self.LRelu(self.fc1(x))  # Apply Leaky ReLU to the first fully connected layer
        layer2 = self.tanH(self.fc2(layer1))  # Apply Tanh to the second fully connected layer
        out = layer2.view(self.batch_size, 1, 28, 28)  # Reshape the output to match image dimensions
        return out

# Discriminator network
class Discriminator(nn.Module):
    def __init__(self, batch_size):
        super().__init()
        self.batch_size = batch_size
        self.fc1 = nn.Linear(1 * 28 * 28, 128)  # Fully connected layer 1
        self.LReLu = nn.LeakyReLU()  # Leaky ReLU activation function
        self.fc2 = nn.Linear(128, 1)  # Fully connected layer 2
        self.SigmoidL = nn.Sigmoid()  # Sigmoid activation function

    # Function for forward propagation
    def forward(self, x):
        flat = x.view(self.batch_size, -1)  # Flatten the input image
        layer1 = self.LReLu(self.fc1(flat))  # Apply Leaky ReLU to the first fully connected layer
        out = self.SigmoidL(self.fc2(layer1))  # Apply Sigmoid to the second fully connected layer
        return out.view(-1, 1).squeeze(1)  # Flatten the output and remove unnecessary dimension
