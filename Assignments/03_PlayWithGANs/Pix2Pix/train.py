import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
from GAN_network import GanNetwork
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): Optimizer for updating model parameters.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.train()
    running_loss = 0.0
    optimizer_generator, optimizer_discriminator = optimizer
    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Train Discriminator
        batch = image_semantic.shape[0]

        for var in model.discriminator.parameters():
            var.requires_grad = True
        optimizer_discriminator.zero_grad()

        real_labels = torch.ones(batch, 60, 60).to(device)
        real_data = torch.cat([image_semantic, image_rgb], dim=1)
        fake_labels = torch.zeros(batch, 60, 60).to(device)
        fake_data = torch.cat([image_semantic, model.generator(image_semantic).detach()], dim=1)

        lossDiscriminator = criterion(model.discriminator(real_data).squeeze(), real_labels) + criterion(model.discriminator(fake_data).squeeze(), fake_labels)

        lossDiscriminator.backward()
        optimizer_discriminator.step()

        # Train Generator
        optimizer_generator.zero_grad()

        for var in model.discriminator.parameters():
            var.requires_grad = False
        
        gen = model.generator(image_semantic)
        fake_data = torch.cat([image_semantic, gen], dim=1)

        image_L1 = 7*torch.mean(torch.abs(gen - image_rgb))
        lossGenerator = criterion(model.discriminator(fake_data).squeeze(), real_labels) + image_L1

        lossGenerator.backward()
        optimizer_generator.step()


        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_rgb, image_semantic, model.generator(image_semantic), 'train_results', epoch)

        running_loss += lossGenerator.item() + lossDiscriminator.item()


        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {running_loss:.4f}, LossGenerator: {lossGenerator.item():.4f}, LossDiscriminator: {lossDiscriminator.item():.4f}')

def validate(model, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)

            # Compute the loss
            loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    model = GanNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer_generator = optim.Adam(model.generator.parameters(), lr=0.001, betas=(0.5, 0.999)) 
    optimizer_discriminator = optim.Adam(model.discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    scheduler_generator = StepLR(optimizer_generator, step_size=200, gamma=0.2)
    scheduler_discriminator = StepLR(optimizer_discriminator, step_size=200, gamma=0.2)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(model, train_loader, [optimizer_generator, optimizer_discriminator], criterion, device, epoch, num_epochs)
        # validate(model, val_loader, criterion, device, epoch, num_epochs)

        # Step the scheduler after each epoch
        scheduler_generator.step()
        scheduler_discriminator.step()

        # Save model checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f'checkpoints/pix2pix_model_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
