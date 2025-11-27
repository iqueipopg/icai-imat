"""
Single unified training script for VQ-VAE - Incomplete version.

You need to implement the main training loop and data loading.
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import VQVAEConfig
from .training import eval_one_epoch, train_one_epoch
from .vq_vae import VQVAE


def create_dataloader(
    dataset_name: str = "CIFAR10", batch_size: int = 256, data_path: str = "./data"
) -> Tuple[DataLoader, DataLoader, float]:
    """Create training and validation dataloaders.

    This function sets up the data pipeline for training:
    - Applies appropriate transforms (ToTensor, Normalize)
    - Creates train/validation splits
    - Computes data variance for loss normalization
    - Returns configured DataLoaders

    Args:
        dataset_name: Name of dataset to load (currently supports "CIFAR10")
        batch_size: Batch size for training
        data_path: Path to store/load data

    Returns:
        Tuple of (train_loader, val_loader, data_variance)
    """
    # Define transforms and load dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))]
    )

    if dataset_name == "CIFAR10":
        training_data = datasets.CIFAR10(
            root=data_path, train=True, download=True, transform=transform
        )
        validation_data = datasets.CIFAR10(
            root=data_path, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Create train and validation DataLoaders
    train_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    val_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # Compute data variance
    # Get sample of data to compute variance
    data_sample = []
    for i, (data, _) in enumerate(train_loader):
        data_sample.append(data)
        if i >= 10:  # Use first 10 batches
            break
    data_sample = torch.cat(data_sample, dim=0)
    data_variance = torch.var(data_sample).item()

    # Return loaders and variance
    return train_loader, val_loader, data_variance


def fit(config: VQVAEConfig) -> Tuple[VQVAE, List[float], List[float]]:
    """Train VQ-VAE model using configuration.

    This is the main training function that:
    - Sets up data loaders and model
    - Runs the training loop for specified iterations
    - Saves checkpoints and tracks metrics
    - Returns trained model and training history

    Args:
        config: VQVAEConfig containing all training parameters

    Returns:
        Tuple of (trained_model, reconstruction_errors, perplexities)
    """
    # Setup data loaders, model, and optimizer
    device = config.device

    train_loader, val_loader, data_variance = create_dataloader(
        config.dataset_name, config.batch_size, config.data_path
    )

    model = VQVAE(
        config.num_hiddens,
        config.num_residual_layers,
        config.num_residual_hiddens,
        config.num_embeddings,
        config.embedding_dim,
        config.commitment_cost,
        config.decay,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, amsgrad=False)

    # Training tracking
    train_res_recon_error: List[float] = []
    train_res_perplexity: List[float] = []

    # Main training loop with validation and checkpointing
    iteration = 0

    while iteration < config.num_training_updates:
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Usamos la función train_one_epoch sobre un solo batch manualmente
            # para mantener la lógica de iteraciones (no por épocas completas)
            model.train()
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = model(data)
            recon_error = torch.mean((data_recon - data) ** 2) / data_variance
            loss = recon_error + vq_loss

            loss.backward()
            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            iteration += 1

            if iteration % 100 == 0:
                print(
                    f"Iteration {iteration}: "
                    f"recon_error={recon_error.item():.3f}, "
                    f"perplexity={perplexity.item():.3f}"
                )

            if iteration >= config.num_training_updates:
                break

    # Save final model and return results
    # OJO: usamos config.save_path y nombre model_15000.pt para el test
    checkpoint_dir = Path(config.save_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "iteration": config.num_training_updates,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "recon_error": train_res_recon_error,
            "perplexity": train_res_perplexity,
        },
        checkpoint_dir / "model_15000.pt",
    )

    return model, train_res_recon_error, train_res_perplexity


if __name__ == "__main__":
    config = VQVAEConfig()
    print("Training on device:", config.device)
    fit(config)
