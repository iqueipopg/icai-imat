"""
Creates a Pytorch dataset to load the dataset used.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.constants import B, C, S, TARGET_SIZE_IMG


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
        # Tensor de ceros: shape [S, S, 5*B + C]
        label = torch.zeros((S, S, 5 * B + C), dtype=torch.float32)
        for box in boxes:
            cls, x, y, w, h = box
            # Celda de la cuadrícula
            j = int(S * x)  # columna
            i = int(S * y)  # fila
            # Solo un objeto por celda
            if label[i, j, C] == 0.0:  # Si no hay objeto, añade
                label[i, j, 1] = 1.0  # one-hot de clase 1 (índice 1 en tu test)
                label[i, j, C] = 1.0  # flag "has object"
                x_cell = S * x - j
                y_cell = S * y - i
                w_cell = w * S
                h_cell = h * S
                label[i, j, C + 1 : C + 5] = torch.tensor([x_cell, y_cell, w_cell, h_cell])
        return label

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

        # Construir rutas de imagen y label asumiendo formato con índice
        img_path = os.path.join(self.imgs_dir, f"raccoon-{index}.jpg")
        label_path = os.path.join(self.labels_dir, f"{index}.txt")

        # Abrir la imagen y convertir a RGB
        image = Image.open(img_path).convert("RGB")

        # Leer las cajas del archivo label
        boxes = []
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Primer valor es clase int, el resto float
                    cls = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    boxes.append([cls] + coords)

        # Transformación de imagen: resize a TARGET_SIZE_IMG y tensor (C,H,W)
        transform = transforms.Compose([
            transforms.Resize(TARGET_SIZE_IMG),  # Cambia con la tuya, ej (400,400)
            transforms.ToTensor(),
        ])
        image = transform(image)

        # Generar tensor label con la función que ya tienes
        label = self._generate_label(boxes)

        return image, label
        
