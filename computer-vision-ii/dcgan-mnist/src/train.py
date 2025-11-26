"""
Script for generating the train loop, and functions of train and test.
You can implement the loss function of the gan in a single method for both models or two
different, it's up to you.
But remember they have different ground truths, and different backwards calls!
Another thing: make the split of the train and test. (In the loop you must evaluate the
validation error).
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from .model import Generator, Discriminator, gan_loss
from .utils import set_seed, get_noise, save_generated_samples, save_gan_losses

set_seed(42)  # reproducibilidad


def train_epoch(
    discriminator, generator, dataloader, d_optimizer, g_optimizer, z_dim, device
):
    d_losses, g_losses = [], []
    for real_images, _ in dataloader:
        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        # ----- Discriminator -----
        d_optimizer.zero_grad()
        real_logits = discriminator(real_images)
        real_loss = gan_loss(real_logits, True)

        noise = get_noise(batch_size, z_dim, device)
        fake_images = generator(noise)
        fake_logits = discriminator(fake_images.detach())
        fake_loss = gan_loss(fake_logits, False)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # ----- Generator -----
        g_optimizer.zero_grad()
        fake_logits = discriminator(fake_images)
        g_loss = gan_loss(fake_logits, True)
        g_loss.backward()
        g_optimizer.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
    return d_losses, g_losses


def test_epoch(generator, fixed_noise, device):
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_noise.to(device)).cpu()
    generator.train()
    return samples


def train():
    n_epochs = 1  # test rápido (sube si quieres entrenamiento real)
    z_dim = 64
    lr = 0.00001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    display_step = 500

    generator = Generator(latent_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999)
    )

    full_dataset = MNIST(".", download=True, transform=transforms.ToTensor())
    train_len = int(0.9 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    dataset_train, dataset_val = random_split(full_dataset, [train_len, val_len])
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=128, shuffle=False)

    for epoch in range(n_epochs):
        train_epoch(
            discriminator,
            generator,
            dataloader_train,
            d_optimizer,
            g_optimizer,
            z_dim,
            device,
        )
    return discriminator, generator


# Puedes conservar tu bloque de 'main' si quieres para ejecutar el entrenamiento real
if __name__ == "__main__":
    train()
