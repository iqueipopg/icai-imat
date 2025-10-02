"""
This script contains the code for the Integrated Gradients.
"""

from typing import Literal

import torch
from torch import nn

from src.utils import normalize_tensor, saliency_map


class IntegratedGradients(nn.Module):
    """
    Computes Integrated Gradients for model interpretability.

    This class implements the Integrated Gradients method, which attributes the
    prediction of a model to its input features by integrating gradients along a
    straight path from a baseline to the input.

    More details can be found in: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model

    def _initialize_baseline(
        self,
        channels: int,
        height: int,
        width: int,
        baseline: Literal["zero", "random"],
    ) -> torch.Tensor:
        """
        Initializes the baseline tensor.

        Args:
            channels: Number of channels of the image.
            height: Height of the image.
            width: Width of the image.
            baseline: Type of baseline to use.

        Returns:
            Baseline tensor of shape (channels, height, width).

        Raises:
            ValueError: If the value of 'baseline' is not valid.
        """

        match baseline:
            case "zero":
                return torch.zeros((channels, height, width))
            case "random":
                return torch.randn((channels, height, width))
            case _:
                raise ValueError("Please introduce a correct value for the baseline.")

    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        n_steps: int = 10,
        baseline: Literal["zero", "random"] = "zero",
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor. Dimensions: [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            n_steps: Number of steps for the integration path from baseline to input.
            baseline: Type of baseline to use.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """

        # TODO
