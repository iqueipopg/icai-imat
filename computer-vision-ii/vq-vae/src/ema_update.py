"""
EMA update functions for VQ-VAE quantizers - Incomplete version.

Students need to implement the exponential moving average updates for the codebook.
"""

import torch
import torch.nn as nn


def update_ema_weights(
    ema_cluster_size: torch.Tensor,
    ema_w: nn.Parameter,
    embedding_weight: torch.Tensor,
    encodings: torch.Tensor,
    flat_input: torch.Tensor,
    decay: float,
    epsilon: float,
    num_embeddings: int,
) -> None:
    """Update EMA weights for vector quantizer.

    This function implements the exponential moving average updates for the
    VQ-VAE codebook. It updates both the cluster sizes and the embedding weights
    using EMA, which often leads to more stable training than gradient-based updates.

    The algorithm:
    1. Update cluster sizes using EMA
    2. Apply Laplace smoothing to prevent dead codes
    3. Update embedding weights using EMA
    4. Normalize embeddings by cluster sizes

    Args:
        ema_cluster_size: EMA cluster size buffer [num_embeddings]
        ema_w: EMA weight parameter [num_embeddings, embedding_dim]
        embedding_weight: Embedding weight parameter [num_embeddings, embedding_dim]
        encodings: One-hot encodings [batch_size, num_embeddings]
        flat_input: Flattened input tensor [batch_size, embedding_dim]
        decay: EMA decay rate (typically 0.99)
        epsilon: Smoothing parameter for Laplace smoothing
        num_embeddings: Number of embeddings in codebook
    """
    # 1. Update cluster sizes using EMA
    # Calculate how many vectors are assigned to each embedding in this batch
    cluster_counts = encodings.sum(dim=0)  # [num_embeddings]

    # EMA update: N_i^(t) = decay * N_i^(t-1) + (1-decay) * n_i^(t)
    ema_cluster_size.data.mul_(decay).add_(cluster_counts, alpha=1 - decay)

    # 2. Apply Laplace smoothing to prevent dead codes
    # This adds a small constant to prevent division by zero for unused embeddings
    n = ema_cluster_size.sum()
    ema_cluster_size_stable = (
        (ema_cluster_size + epsilon) / (n + num_embeddings * epsilon) * n
    )

    # 3. Update embedding weights using EMA
    # Calculate sum of encoder outputs for each embedding
    # encodings.t() is [num_embeddings, batch_size]
    # flat_input is [batch_size, embedding_dim]
    # Result: [num_embeddings, embedding_dim]
    cluster_sum = encodings.t() @ flat_input

    # EMA update: m_i^(t) = decay * m_i^(t-1) + (1-decay) * sum(z_e)
    ema_w.data.mul_(decay).add_(cluster_sum, alpha=1 - decay)

    # 4. Normalize embedding weights by cluster sizes
    # e_i^(t) = m_i^(t) / N_i^(t)
    # unsqueeze(1) adds a dimension for broadcasting: [num_embeddings, 1]
    embedding_weight.data.copy_(ema_w / ema_cluster_size_stable.unsqueeze(1))
