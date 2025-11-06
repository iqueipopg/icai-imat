"""
Script for generating the train loop, and functions of train and test.
You can implement the loss function of the gan in a single method for both models or two
different, it's up to you.
But remember they have different ground truths, and different backwards calls!
Another thing: make the split of the train and test. (In the loop you must evaluate the
validation error).
"""

import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Own modules import can change depending on your architecture
from src.model import Generator, Discriminator
from utils import set_seed, get_noise

set_seed(42)  # set for testing purposes

# DEFINE FUNCTIONS


if __name__ == "__main__":
    # Set your parameters [Optionally can reduce]
    criterion = None
    n_epochs = 200
    z_dim = 64
    display_step = 500
    batch_size = 128
    lr = 0.00001

    # Load MNIST dataset as tensors
    dataloader = DataLoader(
        MNIST(".", download=False, transform=transforms.ToTensor()),
        batch_size=batch_size,
        shuffle=True,
    )

    device = "cuda"
