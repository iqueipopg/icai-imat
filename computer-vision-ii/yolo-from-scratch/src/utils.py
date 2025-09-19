"""
Script to write some necessary functions not related with the rest of the scripts.
"""

import os
from typing import Literal

import torch
from torch import nn
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from src.constants import EPSILON, TARGET_SIZE_IMG, B, C, S


def iou(
    boxes_preds: torch.Tensor,
    boxes_labels: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates Intersection over Union for each pair of the batch.

    Args:
        boxes_preds: Predictions of bounding boxes, Dimensions: [batch_size, S, S, 4].
        boxes_labels: Correct bounding boxes.  Dimensions: [batch_size, S, S, 4].

    Returns:
        Intersection over Union for each pair of the batch. Dimensions:
        [batch_size, S, S].
    """

    # TODO


def nms(
    predicted_boxes: list[torch.Tensor],
    threshold_confidence: float = 0.5,
    threshold_repeated: float = 0.5,
) -> list[torch.Tensor]:
    """
    Applies Non Max Suppression given the predicted boxes. When updating the list, if
    two boxes have different classes, we do not delete them even if their IoU is high.

    Parameters:
        predicted_boxes: List with the bounding boxes predicted. They have to be
            relative to the hole image, not to each cell. Each box has dimension C + 5.
        threshold_confidence: Threshold to remove predicted bounding boxes with low
            confidence.
        threshold_repeated: Threshold to remove predicted boxes because we consider they
            refer to the same box.

    Returns:
        Predicted bounding boxes after NMS. Each box has dimension C + 5.
    """

    # TODO


def to_image_coords(boxes: torch.Tensor, img_w: int, img_h: int) -> list[torch.Tensor]:
    """
    Transforms the coordinates of the bounding boxes from relative to each cell to
    relative to the hole image.

    Args:
        boxes: Predicted bounding boxes. Dimensions: [S, S, 5 * B + C].
        img_w: Width of the image.
        img_h: Height of the image.

    Returns:
        List with the scaled bounding boxes to the hole image. Each tensor has dimension
        C + 5.
    """

    # TODO


# This last part is just to calculate the mAP. It is a little bit annoying.


def mean_average_precision(subset: Literal["train", "test"], model: nn.Module) -> float:
    """
    Calculates mAP in the test set.

    Args:
        subset: Subset where the mAP is calculated.
        model: YOLO.

    Returns:
        Calculated mAP.
    """

    targets = _get_targets_map(subset)
    predictions = _get_predictions_map(subset, model)

    metric = MeanAveragePrecision(
        iou_type="bbox", iou_thresholds=[0.5], class_metrics=True
    )
    metric.update(predictions, targets)
    result = metric.compute()

    return float(result["map"])


def _get_targets_map(subset: Literal["train", "test"]) -> list[dict[str, torch.Tensor]]:
    """
    Gets the targets in the correct format for mAP calculation.

    Args:
        subset: Subset where the mAP is calculated.

    Returns:
        List of dictionaries with the form: {"boxes": ..., "labels": ...}.
    """

    targets = []

    for i in range(len(os.listdir(f"data/labels_resized/{subset}"))):
        temp_box = []
        temp_label = []
        with open(f"data/labels_resized/{subset}/{i}.txt", encoding="utf8") as f:
            for label in f.readlines():
                label, c_x, c_y, width, height = [  # type: ignore
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]
                temp_label.append(label)
                # Change coordinates to (x1, y1, x2, y2)
                temp_box.append(
                    torch.tensor(
                        [
                            (c_x - width / 2) * TARGET_SIZE_IMG[0],
                            (c_y - height / 2) * TARGET_SIZE_IMG[1],
                            (c_x + width / 2) * TARGET_SIZE_IMG[0],
                            (c_y + height / 2) * TARGET_SIZE_IMG[1],
                        ]
                    )
                )
        boxes = torch.stack(temp_box)
        labels = torch.tensor(temp_label)
        targets.append({"boxes": boxes, "labels": labels})

    return targets


def _get_predictions_map(
    subset: Literal["train", "test"], model: nn.Module
) -> list[dict[str, torch.Tensor]]:
    """
    Gets the predictions in the correct format for mAP calculation.

    Args:
        subset: Subset where the mAP is calculated.
        model: YOLO model.

    Returns:
        List of dictionaries with the form: {"boxes": ..., "scores": ...,
        "labels": ...}.
    """

    dir_path = f"data/images/{subset}"
    predictions = []

    for i in range(len(os.listdir(dir_path))):
        bounding_boxes = model.predict(f"{dir_path}/raccoon-{i}.jpg")  # type: ignore
        boxes = []
        scores = []
        labels = []
        if len(bounding_boxes) > 0:
            for bounding_box in bounding_boxes:
                c_x, c_y, width, height = bounding_box[C + 1 :]
                # Change coordinates to (x1, y1, x2, y2)
                normalized_bb = torch.tensor(
                    [
                        float(c_x - width / 2),
                        float(c_y - height / 2),
                        float(c_x + width / 2),
                        float(c_y + height / 2),
                    ]
                )
                boxes.append(normalized_bb)
                scores.append(torch.min(torch.tensor(1.0), bounding_box[C]))
                labels.append(torch.argmax(bounding_box[:C]))
            predictions.append(
                {
                    "boxes": torch.stack(boxes),
                    "scores": torch.stack(scores),
                    "labels": torch.stack(labels),
                }
            )
        else:  # no predictions were made
            predictions.append(
                {
                    "boxes": torch.zeros((0, 4), dtype=torch.float32),
                    "scores": torch.zeros((0,), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                }
            )

    return predictions
