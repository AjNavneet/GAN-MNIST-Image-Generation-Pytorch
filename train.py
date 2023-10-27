import torch

# Function to train the GAN model
def train_model(no_of_epochs, disc, gen, optimD, optimG, dataloaders, loss_fn, input_size, batch_size):
    """
    Train a GAN model.

    Parameters:
    - no_of_epochs: Number of training epochs.
    - disc: Discriminator model.
    - gen: Generator model.
    - optimD: Optimizer for the Discriminator.
    - optimG: Optimizer for the Generator.
    - dataloaders: Dictionary containing data loaders for training and validation.
    - loss_fn: Loss function for training.
    - input_size: Dimension of the random noise input to the Generator.
    - batch_size: Batch size for training.

    Note: The 'dataloaders' dictionary should have keys 'train' and 'valid' for training and validation data.

    """
    
    # Set the device as CUDA or CPU based on availability
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    reall = 1  # Real label
    fakel = 0  # Fake label

    # Run training for each epoch
    for epoch in range(no_of_epochs):
        print('Epoch {}/{}'.format(epoch + 1, no_of_epochs))
        running_loss_D = 0
        running_loss_G = 0
        
        # Loop through the training phase
        for phase in ["train"]:
            # Iterate over batches in the data loader
            for inputs, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                
                # Convert labels into torch tensors with the proper size as per the batch size
                real_label = torch.full((batch_size,), reall, dtype=inputs.dtype, device=device)
                fake_label = torch.full((batch_size,), fakel, dtype=inputs.dtype, device=device)

                # Zero the gradients of the Discriminator optimizer
                optimD.zero_grad()
                
                # Compute output from the Discriminator
                output = disc(inputs)
                
                # Discriminator real loss
                D_real_loss = loss_fn(output, real_label)
                D_real_loss.backward()
                
                # Generate random noise data as input to the Generator
                noise = torch.randn(batch_size, input_size, device=device)
                
                # Generate fake images using the Generator
                fake = gen(noise)
                
                # Pass fake images through the Discriminator with gradient detachment
                output = disc(fake.detach())
                
                # Discriminator fake loss
                D_fake_loss = loss_fn(output, fake_label)
                D_fake_loss.backward()

                # Total loss for the Discriminator
                Disc_loss = D_real_loss + D_fake_loss
                running_loss_D = running_loss_D + Disc_loss
                
                # Update Discriminator's parameters
                optimD.step()

                # Zero the gradients of the Generator optimizer
                optimG.zero_grad()
                
                # Pass fake images obtained from the Generator to the Discriminator
                output = disc(fake)
                
                # Calculate Generator loss by giving fake images as input but providing real labels
                Gen_loss = loss_fn(output, real_label)
                running_loss_G = running_loss_G + Gen_loss
                
                # Backpropagation for the Generator
                Gen_loss.backward()
                
                # Update Generator's parameters
                optimG.step()
        
        # Print the losses for the current epoch
        print("Discriminator Loss : {}".format(running_loss_D))
        print("Generator Loss : {}".format(running_loss_G))
