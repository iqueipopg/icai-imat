"""
Script to define the YOLO loss.
"""

import torch
from torch import nn

from src.constants import EPSILON, LAMBDA_COORD, LAMBDA_NOOBJ, C, B
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

        best_box = best_box.float()
        one_i_obj = one_i_obj.float()
        pred_coords_1 = predictions[..., C + 1 : C + 5]
        pred_coords_2 = predictions[..., C + 6 : C + 10]
        target_coords = one_i_obj * targets[..., C + 1 : C + 5]

        pred_coords = one_i_obj * ((1 - best_box) * pred_coords_1 + best_box * pred_coords_2)

        transformed_coords = torch.sqrt(torch.abs(pred_coords[..., 2:4]) + EPSILON) * torch.sign(pred_coords[..., 2:4])
        pred_coords = torch.cat((pred_coords[..., :2], transformed_coords), dim=-1)

        target_coords[..., 2:4] = torch.sqrt(torch.abs(target_coords[..., 2:4]) + EPSILON)
        pred = torch.cat((pred_coords[..., :2], pred_coords[..., 2:4]), dim=-1)
        target = torch.cat((target_coords[..., :2], target_coords[..., 2:4]), dim=-1)

        loss = self.mse(pred, target)
        return LAMBDA_COORD * loss



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
        pred_conf_all = predictions[..., C : C + 5 * B : 5]

        pred_conf = torch.gather(
            pred_conf_all, -1, best_box  # [batch, S, S, 1]
        ).squeeze(-1)  # [batch, S, S]
        
        target_conf = targets[..., C]  # [batch, S, S]
        
        loss = self.mse(
            one_i_obj.squeeze(-1) * pred_conf,
            one_i_obj.squeeze(-1) * target_conf
        )
        return loss



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
        pred_conf_all = predictions[..., C:C + 5*B:5]  # [batch, S, S, B]
        target_conf_all = targets[..., C].unsqueeze(-1).expand_as(pred_conf_all)  # [batch, S, S, B]
        noobj_mask = (1 - one_i_obj).expand_as(pred_conf_all)

        # Only take MSE of cells with no objects
        loss = self.mse(noobj_mask * pred_conf_all, noobj_mask * target_conf_all) 

        return 2*LAMBDA_NOOBJ * loss


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
        pred_class = predictions[..., :C]  # [batch, S, S, C]
        target_class = targets[..., :C]  # [batch, S, S, C

        loss = self.mse(one_i_obj * pred_class, one_i_obj * target_class)
        return loss



    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            predictions: Predictions of the model. Dimensions: [batch, S, S, 5 * B + C].
            targets: Ground truths. Dimensions: [batch, S, S, 5 * B + C].

        Returns:
            Each of the sub-losses in the order they are coded and final YOLO loss.
        """

        # TODO
        one_i_obj = targets[..., C].unsqueeze(-1)


        pred_box_1 = predictions[..., C + 1 : C + 5]
        pred_box_2 = predictions[..., C + 6 : C + 10]

        target_box = targets[..., C + 1 : C + 5]
        iou_1 = iou(pred_box_1, target_box)
        iou_2 = iou(pred_box_2, target_box)

        
        best_box = (iou_2 > iou_1).long()
        
        coord_loss = self._coordinates_loss(predictions, targets, one_i_obj, best_box)
        obj_loss = self._object_loss(predictions, targets, one_i_obj, best_box)
        noobj_loss = self._no_object_loss(predictions, targets, one_i_obj)
        class_loss = self._class_loss(predictions, targets, one_i_obj)

        total_loss = coord_loss + obj_loss + noobj_loss + class_loss

        return coord_loss, obj_loss, noobj_loss, class_loss, total_loss
