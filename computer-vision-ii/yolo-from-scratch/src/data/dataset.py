"""
Creates a Pytorch dataset to load the dataset used.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import B, C, S


class RaccoonDataset(Dataset):
    """
    Adapted Dataset class for our dataset.
    """

    def __init__(
        self,
        img_dir: str,
        label_dir: str,
    ) -> None:
        """
        Constructor of the class.

        Args:
            img_dir: Path to the directory of the images.
            label_dir: Path to the directory of the labels.
        """

        self.imgs_dir = img_dir
        self.labels_dir = label_dir

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            Length of the dataset.
        """

        return len(os.listdir(self.labels_dir))

    def _generate_label(self, boxes: list[list[int | float]]) -> torch.Tensor:
        """
        Generates the labels with coordinates respect to the cells, not the hole image.
        We will assume at most we can have one object per cell. Therefore, the
        dimensions of the generated label will be [S, S, 5 * C + B]. If we wanted to
        detect at most n objects the dimensions would be [S, S, n, 5 * C + B].

        Args:
            boxes: Labels with respect to the hole image. Each label has 5 elements:
                class, center_x, center_y, width and height.

        Returns:
            Labels with respect to the cells. Dimensions: [S, S, 5 * C + B].
        """

        # TODO

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the image and its labels.

        Args:
            index: Index of the image and labels.

        Returns:
            The image and its labels. Dimensions of the image: [channels, height,
            width]. Dimensions of the labels: [S, S, 5 * C + B].
        """

        # TODO
