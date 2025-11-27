"""
Loss functions for VQ-VAE.
Students need to implement the core loss computations for vector quantization.
"""

import torch
import torch.nn.functional as F


def vq_loss(
    quantized: torch.Tensor, inputs: torch.Tensor, commitment_cost: float
) -> torch.Tensor:
    """
    Vector quantization loss with commitment term.

    The VQ loss consists of two components:
    1. Codebook loss: Updates the embeddings to be closer to encoder outputs
    2. Commitment loss: Encourages encoder outputs to commit to embeddings

    Total: codebook_loss + commitment_cost * commitment_loss
    """
    # sg[x] = x.detach(): no grad para ese término
    codebook_loss = F.mse_loss(quantized, inputs.detach())
    commitment_loss = F.mse_loss(inputs, quantized.detach())
    return codebook_loss + commitment_cost * commitment_loss


def vq_ema_loss(
    quantized: torch.Tensor, inputs: torch.Tensor, commitment_cost: float
) -> torch.Tensor:
    """
    Vector quantization loss for EMA version.

    For EMA quantizers, only the commitment loss is needed since
    the codebook is updated via exponential moving averages.
    """
    commitment_loss = F.mse_loss(inputs, quantized.detach())
    return commitment_cost * commitment_loss


def reconstruction_loss(
    recon: torch.Tensor, target: torch.Tensor, data_variance: float = 1.0
) -> torch.Tensor:
    """
    Reconstruction loss normalized by data variance.

    This loss measures how well the decoder reconstructs the original input.
    Normalizing by data variance helps with training stability.
    """
    mse = F.mse_loss(recon, target)
    return mse / data_variance
