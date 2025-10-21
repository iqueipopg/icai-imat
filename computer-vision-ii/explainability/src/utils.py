"""
Script to define some utility functions.
"""

import os

import torch
from torch import nn


def saliency_map(
    model: nn.Module, x: torch.Tensor, target_class: int | None
) -> torch.Tensor:
    """
    Computes gradients of the model output with respect to input. As we want to return a
    saliency map, we must obtain the maximum of the absolute values along the channels
    dimension.

    Args:
        model: Model we want to explain.
        x: Input tensor. Dimensions: [batch, channels, height, width].
        target_class: Class for which backward pass is computed. If None, then we assume
            it is the one with maximum score.

    Returns:
        Saliency map of the model output with respect to input. Dimensions: [batch,
        height, width].

    Raises:
        RuntimeError: If the gradients calculated with the backward are None.
    """

    # TODO
    # Poner el modelo en modo evaluación
    model.eval()

    # Habilitar gradientes para la entrada
    x = x.clone().detach().requires_grad_(True)

    # Forward pass
    outputs = model(x)

    # Seleccionar la clase objetivo
    if target_class is None:
        target_class = outputs.argmax(dim=1)

    # Seleccionar la salida correspondiente a cada clase objetivo
    selected_outputs = outputs[torch.arange(outputs.size(0)), target_class]

    # Backward pass
    selected_outputs.backward(torch.ones_like(selected_outputs))

    # Obtener gradientes
    gradients = x.grad
    if gradients is None:
        raise RuntimeError("Gradients are None. Check if input requires_grad=True.")

    # Calcular el mapa de saliencia como el máximo del valor absoluto por canal
    saliency = gradients.abs().max(dim=1)[0]

    return saliency
    

def normalize_tensor(explanation: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the explanation tensor.

    Args:
        explanation: Explanation tensor with dimensions [batch, height, width].

    Returns:
        Normalized explanation with the same dimensions.
    """

    max_ = torch.amax(explanation, dim=(1, 2), keepdim=True)
    min_ = torch.amin(explanation, dim=(1, 2), keepdim=True)

    return (explanation - min_) / (max_ - min_ + 1e-8)


def clear_folder(folder_path: str, extension: str = ".png") -> None:
    """
    Clears all the images that match an extension in a folder.

    Ags:
        folder_path: Path of the folder.
        extension: Extension of the files we want to remove.
    """

    for file in os.listdir(folder_path):
        if file.endswith(extension):
            os.remove(os.path.join(folder_path, file))
