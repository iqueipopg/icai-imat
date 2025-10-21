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
        batch_size = original_output.shape[0]
        if target_class is None:
            # argmax por fila, devolver indices shape [B, 1]
            target_class_idxs = original_output.argmax(dim=1, keepdim=True)
        elif isinstance(target_class, int):
            # repetir el entero en forma [B, 1]
            target_class_idxs = torch.full((batch_size, 1), target_class, dtype=torch.long, device=original_output.device)
        else:
            raise ValueError("target_class must be None or int.")

        # coge los valores original_output[batch_idx, class_idx] para cada batch_idx
        out_vals = original_output.gather(1, target_class_idxs).squeeze(1)
        return out_vals, target_class_idxs

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
        self.model.eval()
        batch_size, channels, height, width = x.shape

        # Determinar tamaño del parche de oclusión
        if mask_size is None:
            mask_h = max(height // 10, 1)
            mask_w = max(width // 10, 1)
        elif isinstance(mask_size, int):
            mask_h = mask_w = mask_size
        else:
            mask_h, mask_w = mask_size

        # Determinar stride (asegurar que sea > 0)
        if stride is None:
            stride = max(mask_h // 2, 1)
        if stride <= 0:
            stride = 1

        # Obtener la predicción original
        base_output = self.model(x)
        base_scores, target_class_idxs = self._get_target_outputs(base_output, target_class)

        # Crear mapa de importancia
        importance_map = torch.zeros((batch_size, height, width), device=x.device)

        # Mover la máscara
        for i in range(0, height - mask_h + 1, stride):
            for j in range(0, width - mask_w + 1, stride):
                x_occluded = x.clone()
                x_occluded[:, :, i:i + mask_h, j:j + mask_w] = 0  # Aplicar oclusión

                # Evaluar modelo con entrada ocluida
                occluded_output = self.model(x_occluded)
                occluded_scores, _ = self._get_target_outputs(occluded_output, target_class)

                # Calcular diferencia de puntuación (mantener dimensiones correctas)
                diff = (base_scores - occluded_scores).view(-1, 1, 1)
                importance_map[:, i:i + mask_h, j:j + mask_w] += diff

        # Normalizar mapa de importancia
        importance_map = normalize_tensor(importance_map)

        return importance_map
