"""
Script for some utility functions.
You can add or use less than what we have included.
"""

# standard libraries
import os
import random

# 3pp
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np


def show_tensor_images(
    image_tensor: torch.Tensor, num_images: int = 25, size: tuple = (1, 28, 28)
):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed : Seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def save_generated_samples(
    epochs: list[int],
    samples: list[torch.Tensor],
    samples_per_epoch: int = 8,
    path: str = "generated_samples_epoch_0.png",
) -> None:
    """
    Creates a figure with the same latent values but in different training epochs.

    Args:
        epochs (list[int]): List of epoch numbers to display.
        samples (list[torch.Tensor]): List in which each element is a batch of generated images of
            dimensions [batch, channels, height, width].
        samples_per_epoch (int, optional): Number of samples to show per epoch. Defaults to 8.
        path (str, optional): Path to save the generated figure. Defaults to "generated_samples_epoch_0.png".

    Returns:
        None
    """

    fig, axs = plt.subplots(
        len(epochs),
        samples_per_epoch,
        figsize=(4 * len(epochs), samples_per_epoch),
    )
    assert isinstance(axs, np.ndarray)
    fig.tight_layout(rect=(0.1, 0, 1, 0.95))
    fig.suptitle("Images Generated", fontsize=35)
    for i, epoch in enumerate(epochs):
        for j, img in enumerate(samples[epoch][:samples_per_epoch]):
            img = img.detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))
            axs[i, j].xaxis.set_visible(False)
            axs[i, j].yaxis.set_visible(False)
            axs[i, j].imshow(img, cmap=plt.get_cmap("gray"))
        fig.text(
            0.04,
            1 - (i + 0.5) / len(epochs),
            f"Epoch {epoch}",
            ha="center",
            va="center",
            fontsize=25,
        )

    fig.savefig(path)
    plt.close(fig)


def save_gan_losses(
    d_real_losses: list[float],
    d_fake_losses: list[float],
    d_losses: list[float],
    g_losses: list[float],
    path: str,
) -> None:
    """Saves the evolution of the losses in the GAN.

    Args:
        d_real_losses (List[float]): Losses of the discriminator when predicting real images.
        d_fake_losses (List[float]): Losses of the discriminator when predicting fake images.
        d_losses (List[float]): Losses of the discriminator when predicting real and fake images
            (is the sum of d_real_losses and d_fake_losses).
        g_losses (List[float]): Losses of the generator.
        path (str): Path to save the generated plot.
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    assert isinstance(axs, np.ndarray)

    axs[0].plot(d_real_losses, linewidth=0.5, label="Total")
    axs[0].plot(d_fake_losses, linewidth=0.5, label="Fake")
    axs[0].plot(d_losses, linewidth=0.5, label="Total")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Discriminator")
    axs[0].legend()

    axs[1].plot(g_losses)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Generator")

    fig.savefig(path)
    plt.close(fig)


def get_noise(n_samples: int, z_dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn((n_samples, z_dim), device=device)
