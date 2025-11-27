"""
Training and testing functions for VQ-VAE - Incomplete version.

Students need to implement the training and evaluation loops.
"""

import torch
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .loss import reconstruction_loss


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device | None,
    data_variance: float,
    use_amp: bool = True,
) -> dict[str, float]:
    """Train model for one epoch.

    This function implements one complete training epoch, including:
    - Forward pass through the model
    - Loss computation (reconstruction + VQ loss)
    - Backward pass and optimization
    - Metric tracking and averaging

    Args:
        model: VQ-VAE model to train
        dataloader: Training data loader
        optimizer: Model optimizer (typically Adam)
        device: Training device (CPU or CUDA)
        data_variance: Data variance for loss normalization
        use_amp: Whether to use Automatic Mixed Precision

    Returns:
        Dictionary with training metrics:
        - 'loss': Average total loss
        - 'recon_loss': Average reconstruction loss
        - 'perplexity': Average codebook perplexity
    """
    # Set model to training mode and initialize metrics
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    # Initialize GradScaler for AMP if using CUDA
    # Only create scaler if use_amp=True and device is CUDA
    scaler = None
    if use_amp and device is not None and device.type == "cuda":
        scaler = GradScaler("cuda")

    # Training loop with forward/backward passes
    for data, _ in dataloader:
        if device is not None:
            data = data.to(device)

        optimizer.zero_grad()

        # Forward pass with optional AMP
        if scaler is not None:
            with autocast("cuda"):
                vq_loss, data_recon, perplexity = model(data)
                recon_loss = reconstruction_loss(data_recon, data, data_variance)
                loss = recon_loss + vq_loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward/backward without AMP
            vq_loss, data_recon, perplexity = model(data)
            recon_loss = reconstruction_loss(data_recon, data, data_variance)
            loss = recon_loss + vq_loss

            loss.backward()
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_perplexity += perplexity.item()
        num_batches += 1

    # Return averaged metrics
    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
    }


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device | None,
    data_variance: float,
) -> dict[str, float]:
    """Evaluate model for one epoch.

    This function implements one complete evaluation epoch:
    - Set model to evaluation mode
    - Disable gradients for efficiency
    - Forward pass and loss computation
    - Metric tracking and averaging

    Args:
        model: VQ-VAE model to evaluate
        dataloader: Validation/test data loader
        device: Evaluation device
        data_variance: Data variance for loss normalization

    Returns:
        Dictionary with evaluation metrics:
        - 'loss': Average total loss
        - 'recon_loss': Average reconstruction loss
        - 'perplexity': Average codebook perplexity
    """
    # Set model to evaluation mode and initialize metrics
    model.eval()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_perplexity = 0.0
    num_batches = 0

    # Evaluation loop
    with torch.no_grad():
        for data, _ in dataloader:
            if device is not None:
                data = data.to(device)

            # Forward pass
            vq_loss, data_recon, perplexity = model(data)
            recon_loss = reconstruction_loss(data_recon, data, data_variance)
            loss = recon_loss + vq_loss

            # Track metrics
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_perplexity += perplexity.item()
            num_batches += 1

    # Return averaged metrics
    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "perplexity": total_perplexity / num_batches,
    }
