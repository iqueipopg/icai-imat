# deep learning libraries
import torch
import torch.nn as nn

# own modules
from src.utils import get_dropout_random_indexes
import torchvision.models as models


class Dropout(torch.nn.Module):
    """
    This the Dropout class.

    Attr:
        p: probability of the dropout.
        inplace: indicates if the operation is done in-place.
            Defaults to False.
    """

    def __init__(self, p: float, inplace: bool = False) -> None:
        """
        This function is the constructor of the Dropout class.

        Args:
            p: probability of the dropout.
            inplace: if the operation is done in place.
                Defaults to False.
        """

        # TODO
        # Inicializa clase padre
        super(Dropout, self).__init__()

        # Almacena prob
        self.p = p

        # Indica si operación in-place
        self.inplace = inplace

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method computes the forwward pass.

        Args:
            inputs: inputs tensor. Dimensions: [*].

        Returns:
            outputs. Dimensions: [*], same as inputs tensor.
        """

        # TODO
        # Si el modelo está evaluación o no Dropout, retornamos los datos
        if not self.training or self.p <= 0:
            return inputs

        # Generamos los índices de los valores que serán descartados
        drop_mask_indices = get_dropout_random_indexes(inputs.shape, self.p)

        # Calculamos el factor de escalado para mantener la magnitud de las activaciones
        scaling_factor = 1.0 / (1.0 - self.p)

        # Aplicamos Dropout dependiendo del modo de operación
        if self.inplace:
            # En modo inplace, escalamos primero todo el tensor
            inputs.mul_(scaling_factor)
            # Luego, ponemos a cero los elementos seleccionados en la máscara
            inputs[drop_mask_indices.bool()] = 0
            return inputs
        else:
            # En modo no inplace, creamos una nueva máscara con valores 1 y 0
            dropout_mask = torch.ones_like(inputs)
            dropout_mask[drop_mask_indices.bool()] = 0
            # Retornamos los datos con la máscara aplicada y el factor de escalado
            return inputs * dropout_mask * scaling_factor


import torch
from torch import nn
from torchvision import models


class CNNModel(torch.nn.Module):
    # TODO
    def __init__(
        self, num_classes: int = 10, freeze_base: bool = True, unfreeze_layers: int = 0
    ) -> None:
        """
        Constructor para el modelo basado en transferencia de aprendizaje.

        Args:
            num_classes: Cantidad de clases en la salida (por defecto 10 para Imagenette).
            freeze_base: Si se establece como True, congela los pesos del modelo base excepto las capas especificadas.
            unfreeze_layers: Número de capas finales a descongelar (0 = solo capa totalmente conectada (FC), 1 = layer4, 2 = layer3+layer4, etc.)
        """
        super().__init__()
        # Usar weights='IMAGENET1K_V1' para evitar advertencias en lugar de usar pretrained=True
        self.model_base = models.resnet18(weights="IMAGENET1K_V1")

        if freeze_base:
            for param in self.model_base.parameters():
                param.requires_grad = False

            if unfreeze_layers >= 1:
                for param in self.model_base.layer4.parameters():
                    param.requires_grad = True

            if unfreeze_layers >= 2:
                for param in self.model_base.layer3.parameters():
                    param.requires_grad = True

            if unfreeze_layers >= 3:
                for param in self.model_base.layer2.parameters():
                    param.requires_grad = True

            if unfreeze_layers >= 4:
                for param in self.model_base.layer1.parameters():
                    param.requires_grad = True

        in_features = self.model_base.fc.in_features
        self.model_base.fc = nn.Sequential(nn.Linear(in_features, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagación hacia adelante del modelo.

        Args:
            x: Tensor de entrada con dimensiones [batch_size, canales, altura, anchura].

        Returns:
            Logits con dimensiones [batch_size, num_classes].
        """
        return self.model_base(x)
