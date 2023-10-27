import torch

# Import the Discriminator and Generator classes from the GAN module
from model.GAN import Discriminator 
from model.GAN import Generator

# Import the get_dl function from the data_utils module
from data.data_utils import get_dl

# Import the train_model function from the train module
from train import train_model

# Set a manual random seed for reproducibility
torch.manual_seed(4)

# Define batch size, number of training epochs, and input size
batch_size = 128
no_of_epochs = 5
input_size = 100

# Get training and validation data loaders using the get_dl function
train_loader, test_loader = get_dl(batch_size)

# Create a dictionary to store data loaders for training and validation
dl = {}
dl['train'] = train_loader
dl['valid'] = test_loader

# Initialize the Discriminator and Generator models with the specified batch size and input size
disc = Discriminator(batch_size)
gen = Generator(batch_size, input_size)

# Define optimizers for the Discriminator and Generator
optimD = torch.optim.Adam(disc.parameters(), lr=0.001, weight_decay=1e-05)
optimG = torch.optim.Adam(gen.parameters(), lr=0.001, weight_decay=1e-05)

# Define the loss function for training
loss_fn = torch.nn.BCELoss()

# Check if a CUDA-compatible GPU is available and set the device accordingly
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Move the Discriminator and Generator models to the chosen device
disc.to(device)
gen.to(device)

# Start the training process by calling the train_model function
train_model(no_of_epochs, disc, gen, optimD, optimG, dl, loss_fn, input_size, batch_size)
