"""
Script to create the Discriminator and Generator for the GAN.

Optionally include the loss here, or in new file loss.py
"""

import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),  # 28x28 -> 14x14
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                out_channels, out_channels * 2, kernel_size=4, stride=2, padding=1
            ),  # 14x14 -> 7x7
            nn.BatchNorm2d(out_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(out_channels * 2 * 7 * 7, 1),
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)


class Generator(nn.Module):
    def __init__(self, latent_dim=64, out_channels=1):
        super().__init__()
        self.out_channels = 1  # fuerza la salida a 1 canal, pase lo que pase
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                64, self.out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


def gan_loss(disc_output, real_images=True, criterion=None):
    """
    GAN loss function - binary cross-entropy for real vs fake classification.
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    labels = (
        torch.ones_like(disc_output) if real_images else torch.zeros_like(disc_output)
    )
    loss = criterion(disc_output.squeeze(), labels.squeeze())
    return loss


def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    real = real.to(device)

    noise = torch.randn(num_images, z_dim, device=device)
    fake = gen(noise).detach()

    disc_real = disc(real)
    disc_fake = disc(fake)

    real_loss = gan_loss(disc_real, real_images=True, criterion=criterion)
    fake_loss = gan_loss(disc_fake, real_images=False, criterion=criterion)

    return (real_loss + fake_loss) / 2.0
