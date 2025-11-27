"""
VQ-VAE model with refactored quantizers - Incomplete version.

You need to implement the core vector quantization logic and model forward passes.
"""

from typing import Tuple

import torch
from torch import nn

from .ema_update import update_ema_weights
from .loss import vq_ema_loss, vq_loss
from .networks import Decoder, Encoder


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with commitment loss.

    This layer performs vector quantization by finding the closest embedding
    for each input vector and applying straight-through estimation.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, commitment_cost: float
    ) -> None:
        """Initialize vector quantizer.

        Args:
            num_embeddings: Size of the codebook (number of discrete codes)
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for the commitment loss term
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # Initialize embedding layer
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )

        self._commitment_cost = commitment_cost

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through vector quantizer.

        This method should:
        1. Reshape inputs from [B, D, H, W] to [B*H*W, D] for processing
        2. Compute distances between inputs and all embeddings
        3. Find closest embeddings and create one-hot encodings
        4. Quantize inputs using closest embeddings
        5. Apply straight-through estimation
        6. Compute perplexity and return results

        Args:
            inputs: Input tensor [B, D, H, W] where D is embedding_dim

        Returns:
            Tuple of (loss, quantized, perplexity, encodings)
            - loss: VQ loss including commitment term
            - quantized: Quantized tensor same shape as input
            - perplexity: Measure of codebook usage
            - encodings: One-hot encoding matrix [B*H*W, num_embeddings]
        """
        # Reshape inputs and compute distances to embeddings
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input: [B*H*W, D]
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2*x^T*e
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Find closest embeddings and create encodings
        # encoding_indices: [B*H*W]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and apply straight-through estimation
        # Quantize the input using the codebook
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Compute VQ loss and perplexity
        loss = vq_loss(quantized, inputs, self._commitment_cost)

        # Straight-through estimator: copy gradients from quantized to inputs
        quantized = inputs + (quantized - inputs).detach()

        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Calculate perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # Return results
        return loss, quantized, perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    """Vector Quantization layer with Exponential Moving Average updates.

    This version updates the codebook using EMA instead of gradient descent,
    which often leads to more stable training.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize EMA vector quantizer.

        Args:
            num_embeddings: Size of the codebook
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for the commitment loss term
            decay: EMA decay rate (typically 0.99)
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        # Initialize embedding layer
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()

        self._commitment_cost = commitment_cost

        # Register EMA buffers and parameters
        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through EMA vector quantizer.

        Similar to VectorQuantizer but uses EMA updates during training.

        Args:
            inputs: Input tensor [B, D, H, W]

        Returns:
            Tuple of (loss, quantized, perplexity, encodings)
        """
        # Implement quantization with EMA updates
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            update_ema_weights(
                self._ema_cluster_size,
                self._ema_w,
                self._embedding.weight,
                encodings,
                flat_input,
                self._decay,
                self._epsilon,
                self._num_embeddings,
            )

        # Loss
        loss = vq_ema_loss(quantized, inputs, self._commitment_cost)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        # Convert quantized from BHWC -> BCHW
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings


class VQVAE(nn.Module):
    """Complete VQ-VAE model combining encoder, quantizer, and decoder."""

    def __init__(
        self,
        num_hiddens: int,
        num_residual_layers: int,
        num_residual_hiddens: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float,
        decay: float = 0,
    ) -> None:
        """Initialize VQ-VAE model.

        Args:
            num_hiddens: Number of hidden channels in encoder/decoder
            num_residual_layers: Number of residual layers
            num_residual_hiddens: Hidden channels in residual blocks
            num_embeddings: Size of codebook
            embedding_dim: Dimension of embeddings
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate (0 for standard VQ, >0 for EMA)
        """
        super().__init__()

        # Initialize encoder, pre-quantization conv, quantizer, and decoder
        self._encoder = Encoder(
            3, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            num_hiddens, embedding_dim, kernel_size=1, stride=1
        )

        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(
                num_embeddings, embedding_dim, commitment_cost, decay
            )
        else:
            self._vq_vae = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )

        self._decoder = Decoder(
            embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VQ-VAE.

        The forward pass should:
        1. Encode input to latent representation
        2. Apply pre-quantization convolution
        3. Quantize the latent representation
        4. Decode quantized representation to reconstruction

        Args:
            x: Input images [B, C, H, W]

        Returns:
            Tuple of (vq_loss, reconstruction, perplexity)
        """
        # Encode, quantize, and decode
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity
