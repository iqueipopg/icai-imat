"""
Script to define the YOLO loss.
"""

import torch
from torch import nn

from src.constants import EPSILON, LAMBDA_COORD, LAMBDA_NOOBJ, C
from src.utils import iou


class YOLOLoss(nn.Module):
    """
    Implementation of YOLO loss function from the original paper. Some parts are
    hardcoded because we assume B = 2.
    """

    def __init__(self) -> None:
        """
        Constructor of the class.
        """

        super().__init__()

        self.mse = nn.MSELoss()

    def _coordinates_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the predicted coordinates of the bounding boxes.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].
            best_box: Binary tensor indicating the best box for each cell, which is the
                one with more confidence. For position (i, j): 0 -> first bounding box,
                1 -> second bounding box, because we assume B = 2). Dimensions:
                [batch_size, S, S, 1].

        Returns:
            Loss for the predicted coordinates. It is a scalar.
        """

        # TODO

    def _object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
        best_box: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the prediction of if a cell contains an object.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].
            best_box: Binary tensor indicating the best box for each cell, which is the
                one with more confidence. For position (i, j): 0 -> first bounding box,
                1 -> second bounding box, because we assume B = 2). Dimensions:
                [batch_size, S, S, 1].

        Returns:
            Loss for the prediction of if a cell contains an object. It is a scalar.
        """

        # TODO

    def _no_object_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the prediction of if a cell does not contain an object.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S].

        Returns:
            Loss for the prediction of if a cell does not contain an object. It is a
            scalar.
        """

        # TODO

    def _class_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        one_i_obj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Obtains the loss for the class of the objects.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].
            one_i_obj: Binary tensor indicating if there is an object in each cell.
                Dimensions: [batch, S, S, 1].

        Returns:
            Loss for the prediction of the class of the objects. It is a scalar.
        """

        # TODO

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].

        Returns:
            YOLO loss and each of the sub-losses.
        """

        # TODO
