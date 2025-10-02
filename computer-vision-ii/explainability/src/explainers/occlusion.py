"""
This script contains the code for the Occlusion.
"""

import torch
from torch import nn

from src.utils import normalize_tensor


class Occlusion(nn.Module):
    """
    Computes Occlusion for model interpretability.

    This class provides methods to generate feature or pixel-level importance
    maps by systematically occluding parts of the input and measuring the
    impact on the model's output.

    More details can be found in: https://arxiv.org/abs/1311.2901
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor of the class.

        Args:
            model: Model to explain.
        """

        super().__init__()

        self.model = model

    def _get_target_outputs(
        self, original_output: torch.Tensor, target_class: int | None
    ):
        """
        Gets the output values and target indices for the specified class.

        Args:
            original_output: The model's output tensor.
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.

        Returns:
            Output values for the target class and target class indices as a tensor.

        Raises:
            ValueError: If target_class is not None or an integer.
        """

        # TODO

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        target_class: int | None = None,
        mask_size: int | tuple[int, int] | None = None,
        stride: int | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, channels, height, width].
            target_class: Class index for which the explanation is computed. If None, it
                uses the class with the highest score for each sample.
            mask_size: Shape of the occlusion mask. If int, creates a square mask. If
                tuple, uses (width, height). If None, it uses a mask ten times smaller
                than the width and height.
            stride: Stride for moving the occlusion mask. If None, it uses half the mask
                size.

        Returns:
            Explanation. Dimensions: [batch, height, width].
        """

        # TODO
